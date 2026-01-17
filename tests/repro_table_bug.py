import os
import sqlite3
import time
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue

def reproduce_bug():
    session_id = "test_repro"
    print(f"Initializing Memory and Queue for session: {session_id}")
    memory = ACE_Memory(session_id=session_id)
    queue = TaskQueue(session_id=session_id)
    
    # Check if table exists
    with sqlite3.connect(queue.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_queue'")
        if cursor.fetchone():
            print("✓ Table 'task_queue' exists.")
        else:
            print("✗ Table 'task_queue' does NOT exist.")
            return

    print("Clearing memory...")
    memory.clear()
    
    # Check if table still exists
    print("Checking if 'task_queue' table exists after memory.clear()...")
    with sqlite3.connect(queue.db_path) as conn:
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='task_queue'")
            if cursor.fetchone():
                print("✓ Table 'task_queue' still exists.")
            else:
                print("✗ Table 'task_queue' disappeared! (Bug reproduced)")
        except Exception as e:
            print(f"Error checking table: {e}")

if __name__ == "__main__":
    reproduce_bug()
