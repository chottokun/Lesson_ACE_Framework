import os
import sqlite3
import json
import numpy as np
import faiss
from typing import List, Optional, Tuple, Dict, Any
from filelock import FileLock

from ace_rm.config import (
    DB_PATH, DISTANCE_METRIC, DISTANCE_THRESHOLD, EMBEDDING_MODEL_NAME
)
from ace_rm.utils.embedding_manager import get_embedding_model

class ACE_Memory:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id

        if self.session_id:
            data_dir = "user_data"
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, f"ace_memory_{self.session_id}.db")
            self.index_path = os.path.join(data_dir, f"ace_memory_{self.session_id}.faiss")
        else:
            self.db_path = DB_PATH
            self.index_path = f"{DB_PATH}_idx_{self.session_id}.index"
        
        self.index_lock_path = f"{self.index_path}.lock"
        self.last_index_mtime = 0.0

        self.distance_metric = DISTANCE_METRIC
        self.distance_threshold = DISTANCE_THRESHOLD

        self.encoder_name = EMBEDDING_MODEL_NAME
        # Use shared embedding model
        self.encoder = get_embedding_model()
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.use_prefixes = "ruri" in self.encoder_name.lower()

        self._init_db()
        self._load_or_build_index()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    entities TEXT,
                    problem_class TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(content, entities, problem_class, content='documents', content_rowid='id')")
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN INSERT INTO documents_fts(rowid, content, entities, problem_class) VALUES (new.id, new.content, new.entities, new.problem_class); END;")
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN INSERT INTO documents_fts(documents_fts, rowid, content, entities, problem_class) VALUES('delete', old.id, old.content, old.entities, old.problem_class); END;")
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN INSERT INTO documents_fts(documents_fts, rowid, content, entities, problem_class) VALUES('delete', old.id, old.content, old.entities, old.problem_class); INSERT INTO documents_fts(rowid, content, entities, problem_class) VALUES (new.id, new.content, new.entities, new.problem_class); END;")

    def _load_or_build_index(self):
        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    self._create_empty_index()
                    self._rebuild_vectors_from_db()
            else:
                self._create_empty_index()
                self._rebuild_vectors_from_db()
                faiss.write_index(self.index, self.index_path)
            
            if os.path.exists(self.index_path):
                self.last_index_mtime = os.path.getmtime(self.index_path)

    def _create_empty_index(self):
        if self.distance_metric == 'cosine':
            self.index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dimension))
        else:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

    def _rebuild_vectors_from_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content FROM documents")
            rows = cursor.fetchall()
            if rows:
                contents = [r[1] for r in rows]
                if self.use_prefixes:
                    contents = ["検索文書: " + c for c in contents]
                embeddings = self.encoder.encode(contents)
                if self.distance_metric == 'cosine':
                    faiss.normalize_L2(embeddings)
                ids = np.array([r[0] for r in rows])
                self.index.add_with_ids(np.array(embeddings).astype('float32'), ids)

    def add(self, content: str, entities: List[str] = [], problem_class: str = ""):
        entities_json = json.dumps(entities)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (content, entities, problem_class) VALUES (?, ?, ?)",
                (content, entities_json, problem_class)
            )
            doc_id = cursor.lastrowid

        encoded_content = "検索文書: " + content if self.use_prefixes else content
        vector = self.encoder.encode([encoded_content])
        
        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    self._create_empty_index()
            
            if self.distance_metric == 'cosine':
                faiss.normalize_L2(vector)
            self.index.add_with_ids(np.array(vector).astype('float32'), np.array([doc_id]))
            faiss.write_index(self.index, self.index_path)
            self.last_index_mtime = os.path.getmtime(self.index_path)

    def add_batch(self, items: List[Dict[str, Any]]):
        """Optimized batch insertion."""
        if not items: return
        
        contents = [item['content'] for item in items]
        entities_list = [json.dumps(item.get('entities', [])) for item in items]
        p_classes = [item.get('problem_class', '') for item in items]
        
        # 1. DB write in one transaction
        doc_ids = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for content, ent_json, p_class in zip(contents, entities_list, p_classes):
                cursor.execute(
                    "INSERT INTO documents (content, entities, problem_class) VALUES (?, ?, ?)",
                    (content, ent_json, p_class)
                )
                doc_ids.append(cursor.lastrowid)

        # 2. Batch Encoding
        prefixed_contents = ["検索文書: " + c for c in contents] if self.use_prefixes else contents
        vectors = self.encoder.encode(prefixed_contents)
        if self.distance_metric == 'cosine':
            faiss.normalize_L2(vectors)
        
        # 3. Batch Index update
        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    self._create_empty_index()
            
            self.index.add_with_ids(np.array(vectors).astype('float32'), np.array(doc_ids))
            faiss.write_index(self.index, self.index_path)
            self.last_index_mtime = os.path.getmtime(self.index_path)

    def find_similar_vectors(self, content: str, threshold: float = 0.3) -> List[Tuple[int, float]]:
        encoded_content = "検索クエリ: " + content if self.use_prefixes else content
        vector = self.encoder.encode([encoded_content])
        
        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                 try:
                    current_mtime = os.path.getmtime(self.index_path)
                    if current_mtime > self.last_index_mtime:
                        self.index = faiss.read_index(self.index_path)
                        self.last_index_mtime = current_mtime
                 except Exception:
                    pass

            if self.index.ntotal > 0:
                D, I = self.index.search(np.array(vector).astype('float32'), 3)
                results = []
                for i in range(len(I[0])):
                    idx = I[0][i]
                    dist = D[0][i]
                    if idx >= 0:
                        if self.distance_metric == 'cosine':
                            if dist > threshold: results.append((int(idx), float(dist)))
                        else:
                            if dist < threshold: results.append((int(idx), float(dist)))
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
        entities_json = json.dumps(entities)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE documents SET content = ?, entities = ?, problem_class = ?, timestamp = CURRENT_TIMESTAMP WHERE id = ?",
                (content, entities_json, problem_class, doc_id)
            )

        encoded_content = "検索文書: " + content if self.use_prefixes else content
        vector = self.encoder.encode([encoded_content])
        
        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            
            self.index.remove_ids(np.array([doc_id]).astype('int64'))
            if self.distance_metric == 'cosine':
                faiss.normalize_L2(vector)
            self.index.add_with_ids(np.array(vector).astype('float32'), np.array([doc_id]))
            faiss.write_index(self.index, self.index_path)
            self.last_index_mtime = os.path.getmtime(self.index_path)

    def search(self, query: str, k: int = 3, distance_threshold: float = None) -> List[str]:
        if os.path.exists(self.index_path):
            current_mtime = os.path.getmtime(self.index_path)
            if current_mtime > self.last_index_mtime:
                with FileLock(self.index_lock_path):
                     try:
                        self.index = faiss.read_index(self.index_path)
                        self.last_index_mtime = current_mtime
                     except Exception:
                        pass
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
            
        results = {}
        if self.index.ntotal > 0:
            encoded_query = "検索クエリ: " + query if self.use_prefixes else query
            query_vec = self.encoder.encode([encoded_query])
            search_k = min(k * 3, self.index.ntotal)  
            D, I = self.index.search(np.array(query_vec).astype('float32'), search_k)
            
            found_ids = []
            for distance, idx in zip(D[0], I[0]):
                if idx >= 0:
                    if self.distance_metric == 'cosine':
                        if distance > distance_threshold: found_ids.append(int(idx))
                    else:
                        if distance < distance_threshold: found_ids.append(int(idx))
            found_ids = found_ids[:k]
            
            if found_ids:
                placeholders = ','.join('?' * len(found_ids))
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT content FROM documents WHERE id IN ({placeholders})", found_ids)
                    for row in cursor.fetchall():
                        results[row[0]] = row[0]

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
                    pass
        return list(results.values())
    
    def clear(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.index_lock_path):
            try: os.remove(self.index_lock_path)
            except OSError: pass
        self.__init__(session_id=self.session_id)

    def get_all(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, entities, problem_class, timestamp FROM documents ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]
