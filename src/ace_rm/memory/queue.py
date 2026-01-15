import os
import sqlite3
from typing import List, Optional, Dict, Any

DB_PATH = os.environ.get("ACE_DB_PATH", "ace_memory.db")

class TaskQueue:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id
        if self.session_id:
            data_dir = "user_data"
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, f"ace_memory_{self.session_id}.db")
        else:
            self.db_path = DB_PATH
        
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    agent_output TEXT,
                    status TEXT DEFAULT 'pending', -- pending, processing, done, failed
                    retries INTEGER DEFAULT 0,
                    error_msg TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

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
            cursor.execute("SELECT * FROM task_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def mark_task_processing(self, task_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (task_id,))

    def mark_task_complete(self, task_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = 'done', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (task_id,))
    
    def mark_task_failed(self, task_id: int, error_msg: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = 'failed', error_msg = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (error_msg, task_id))

    def get_tasks(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, user_input, status, created_at, updated_at, error_msg FROM task_queue ORDER BY id DESC LIMIT 20")
            return [dict(row) for row in cursor.fetchall()]

    def clear(self):
        """Note: This is usually handled by memory.clear() if they share the same DB file."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM task_queue")
