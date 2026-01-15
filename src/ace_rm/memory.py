import os
import sqlite3
import json
import numpy as np
import faiss
from typing import List, Optional, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from filelock import FileLock
from . import config
from . import constants

class ACE_Memory:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id

        if self.session_id:
            # Session-specific paths in a dedicated folder
            os.makedirs(config.USER_DATA_DIR, exist_ok=True)
            self.db_path = os.path.join(config.USER_DATA_DIR, f"ace_memory_{self.session_id}.db")
            self.index_path = os.path.join(config.USER_DATA_DIR, f"ace_memory_{self.session_id}.faiss")
        else:
            # Global paths in the project root
            self.db_path = config.DB_PATH
            self.index_path = f"{config.DB_PATH}_idx_{self.session_id}.index"
        self.index_lock_path = f"{self.index_path}.lock"

        # Track last index loaded time
        self.last_index_mtime = 0.0

        # Configure distance threshold and metric from environment or use default
        self.distance_metric = config.DISTANCE_METRIC
        self.distance_threshold = config.DISTANCE_THRESHOLD

        # Initialize embedding model (configurable via environment variables)
        self.encoder_name = config.ENCODER_NAME
        self.dimension = config.EMBEDDING_DIMENSION

        # Performance optimization: use GPU if available for ruri models
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = SentenceTransformer(self.encoder_name, device=device)

        # Prefix configuration for models like ruri-v3
        self.use_prefixes = "ruri" in self.encoder_name.lower()

        self._init_db()
        self._load_or_build_index()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for concurrency
            conn.execute("PRAGMA journal_mode=WAL;")

            # Main table with metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    entities TEXT,
                    problem_class TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Full Text Search table
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(content, entities, problem_class, content='documents', content_rowid='id')")
            # Triggers to keep FTS in sync
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN INSERT INTO documents_fts(rowid, content, entities, problem_class) VALUES (new.id, new.content, new.entities, new.problem_class); END;")
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN INSERT INTO documents_fts(documents_fts, rowid, content, entities, problem_class) VALUES('delete', old.id, old.content, old.entities, old.problem_class); END;")
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN INSERT INTO documents_fts(documents_fts, rowid, content, entities, problem_class) VALUES('delete', old.id, old.content, old.entities, old.problem_class); INSERT INTO documents_fts(rowid, content, entities, problem_class) VALUES (new.id, new.content, new.entities, new.problem_class); END;")

            # Persistent Task Queue for Background Worker
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS task_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    agent_output TEXT,
                    status TEXT DEFAULT '{constants.TASK_STATUS_PENDING}',
                    retries INTEGER DEFAULT 0,
                    error_msg TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _load_or_build_index(self):
        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    if self.distance_metric == 'cosine':
                        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
                    else:
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                    self._rebuild_vectors_from_db()
            else:
                if self.distance_metric == 'cosine':
                    self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
                else:
                    self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                self._rebuild_vectors_from_db()
                faiss.write_index(self.index, self.index_path)

            if os.path.exists(self.index_path):
                self.last_index_mtime = os.path.getmtime(self.index_path)

    def _rebuild_vectors_from_db(self):
        # This function should be called within a lock
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content FROM documents")
            rows = cursor.fetchall()
            if rows:
                contents = [r[1] for r in rows]
                if self.use_prefixes:
                    contents = [constants.EMBEDDING_DOCUMENT_PREFIX + c for c in contents]
                embeddings = self.encoder.encode(contents)
                if self.distance_metric == 'cosine':
                    faiss.normalize_L2(embeddings)
                ids = np.array([r[0] for r in rows])
                self.index.add_with_ids(np.array(embeddings).astype('float32'), ids)

    def add(self, content: str, entities: List[str] = [], problem_class: str = ""):
        # Simple insertion without deduplication (logic moved to BackgroundWorker)
        entities_json = json.dumps(entities)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (content, entities, problem_class) VALUES (?, ?, ?)",
                (content, entities_json, problem_class)
            )
            doc_id = cursor.lastrowid

        # Encode with prefix if needed
        encoded_content = constants.EMBEDDING_DOCUMENT_PREFIX + content if self.use_prefixes else content
        vector = self.encoder.encode([encoded_content])
        with FileLock(self.index_lock_path):
            # Read fresh, add, write
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    if self.distance_metric == 'cosine':
                        self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
                    else:
                        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

            if self.distance_metric == 'cosine':
                faiss.normalize_L2(vector)
            self.index.add_with_ids(np.array(vector).astype('float32'), np.array([doc_id]))
            faiss.write_index(self.index, self.index_path)
            self.last_index_mtime = os.path.getmtime(self.index_path)

    def find_similar_vectors(self, content: str, threshold: float = 0.3) -> List[Tuple[int, float]]:
        """Find similar documents based on vector distance."""
        # Encode with query prefix if needed
        encoded_content = constants.EMBEDDING_QUERY_PREFIX + content if self.use_prefixes else content
        vector = self.encoder.encode([encoded_content])

        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                 try:
                    # Check for updates from other processes
                    current_mtime = os.path.getmtime(self.index_path)
                    if current_mtime > self.last_index_mtime:
                        self.index = faiss.read_index(self.index_path)
                        self.last_index_mtime = current_mtime
                 except Exception:
                    pass

            if self.index.ntotal > 0:
                # Search slightly more candidates to filter by strict threshold
                D, I = self.index.search(np.array(vector).astype('float32'), 3)
                results = []
                for i in range(len(I[0])):
                    idx = I[0][i]
                    dist = D[0][i]
                    if idx >= 0:
                        # For IP (cosine), distance is similarity, so we want dist > threshold
                        if self.distance_metric == 'cosine':
                            if dist > threshold:
                                results.append((int(idx), float(dist)))
                        else:
                            if dist < threshold:
                                results.append((int(idx), float(dist)))
                return results
        return []

    def get_document_by_id(self, doc_id: int) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_document(self, doc_id: int, content: str, entities: List[str], problem_class: str):
        """Updates content, entities, problem_class and re-indexes vector."""
        entities_json = json.dumps(entities)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE documents SET content = ?, entities = ?, problem_class = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                (content, entities_json, problem_class, doc_id)
            )

        # Update Vector Index
        # FAISS IndexIDMap doesn't support simple update. We must remove and add.
        # However, remove_ids requires a selector.

        # Encode with prefix if needed
        encoded_content = constants.EMBEDDING_DOCUMENT_PREFIX + content if self.use_prefixes else content
        vector = self.encoder.encode([encoded_content])

        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)

            # Remove old vector
            self.index.remove_ids(np.array([doc_id]).astype('int64'))

            # Add new vector
            if self.distance_metric == 'cosine':
                faiss.normalize_L2(vector)
            self.index.add_with_ids(np.array(vector).astype('float32'), np.array([doc_id]))

            faiss.write_index(self.index, self.index_path)
            self.last_index_mtime = os.path.getmtime(self.index_path)

    def search(self, query: str, k: int = 3, distance_threshold: float = None) -> List[str]:
        """
        Search for relevant documents using hybrid retrieval (vector + keyword search).
        """
        # --- Fix for Stale Index ---
        # Check if index file has changed on disk since last load
        if os.path.exists(self.index_path):
            current_mtime = os.path.getmtime(self.index_path)
            if current_mtime > self.last_index_mtime:
                # Reload index safely
                with FileLock(self.index_lock_path):
                     try:
                        self.index = faiss.read_index(self.index_path)
                        self.last_index_mtime = current_mtime
                        print(f"[Memory] Index reloaded from disk (mtime={current_mtime})", flush=True)
                     except Exception:
                        pass # Keep current index if reload fails

        # Use instance threshold if not specified
        if distance_threshold is None:
            distance_threshold = self.distance_threshold

        results = {}
        # 1. Vector Search with relevance filtering
        if self.index.ntotal > 0:
            # Encode with query prefix if needed
            encoded_query = constants.EMBEDDING_QUERY_PREFIX + query if self.use_prefixes else query
            query_vec = self.encoder.encode([encoded_query])
            # Search for more candidates to allow for filtering
            search_k = min(k * 3, self.index.ntotal)
            D, I = self.index.search(np.array(query_vec).astype('float32'), search_k)

            # Filter by distance/similarity threshold
            found_ids = []
            for distance, idx in zip(D[0], I[0]):
                if idx >= 0:
                    if self.distance_metric == 'cosine':
                        if distance > distance_threshold:
                            found_ids.append(int(idx))
                    else:
                        if distance < distance_threshold:
                            found_ids.append(int(idx))

            # Limit to k results after filtering
            found_ids = found_ids[:k]

            if found_ids:
                placeholders = ','.join('?' * len(found_ids))
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT content FROM documents WHERE id IN ({placeholders})", found_ids)
                    for row in cursor.fetchall():
                        results[row[0]] = row[0]

        # 2. Keyword Search (FTS5) - only if no vector results
        # This ensures we don't overwhelm with irrelevant keyword matches
        if len(results) < k:
            with sqlite3.connect(self.db_path) as conn:
                try:
                    cursor = conn.cursor()
                    remaining = k - len(results)
                    cursor.execute("SELECT content FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?", (query, remaining))
                    for row in cursor.fetchall():
                        if row[0] not in results:
                            results[row[0]] = row[0]
                except Exception:
                    pass # FTS syntax error or other issues
        return list(results.values())

    def clear(self):
        """Clears all memory (DB and Index) for the current session or globally."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.index_lock_path):
            try:
                os.remove(self.index_lock_path)
            except OSError:
                pass # File might be locked by another process
        self.__init__(session_id=self.session_id)

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieves all documents from the DB."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, entities, problem_class, timestamp FROM documents ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    def get_tasks(self) -> List[Dict[str, Any]]:
        """Retrieves all tasks from the task queue."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_input, status, created_at, updated_at, error_msg FROM task_queue ORDER BY id DESC LIMIT 20")
            return [dict(row) for row in cursor.fetchall()]

    # --- Queue Operations ---
    def enqueue_task(self, user_input: str, agent_output: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO task_queue (user_input, agent_output) VALUES (?, ?)",
                (user_input, agent_output)
            )

    def fetch_pending_task(self) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch one pending task (FIFO)
            cursor.execute("SELECT * FROM task_queue WHERE status = ? ORDER BY id ASC LIMIT 1", (constants.TASK_STATUS_PENDING,))
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def mark_task_processing(self, task_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (constants.TASK_STATUS_PROCESSING, task_id))

    def mark_task_complete(self, task_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (constants.TASK_STATUS_DONE, task_id))

    def mark_task_failed(self, task_id: int, error_msg: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = ?, error_msg = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (constants.TASK_STATUS_FAILED, error_msg, task_id))
