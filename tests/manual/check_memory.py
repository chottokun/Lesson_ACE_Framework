#!/usr/bin/env python3
import sqlite3
import sys
import glob

db_files = glob.glob("user_data/ace_memory_*.db")
if not db_files:
    print("No database files found")
    sys.exit(1)

# Check the most recent DB (largest file)
db_files.sort(key=lambda x: -len(open(x, 'rb').read()))
latest_db = db_files[0]

print(f"Checking: {latest_db}\n")

with sqlite3.connect(latest_db) as conn:
    cursor = conn.cursor()
    
    # Check documents
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    print(f"Documents in memory: {doc_count}")
    
    if doc_count > 0:
        cursor.execute("SELECT id, content, problem_class FROM documents LIMIT 3")
        print("\nSample documents:")
        for row in cursor.fetchall():
            print(f"  ID {row[0]}: {row[2]} - {row[1][:100]}...")
    
    # Check tasks
    cursor.execute("SELECT COUNT(*), status FROM task_queue GROUP BY status")
    print("\nTask queue status:")
    for row in cursor.fetchall():
        print(f"  {row[1]}: {row[0]} tasks")
    
    # Check recent tasks
    cursor.execute("SELECT id, user_input, status, error_msg FROM task_queue ORDER BY id DESC LIMIT 3")
    recent_tasks = cursor.fetchall()
    if recent_tasks:
        print("\nRecent tasks:")
        for row in recent_tasks:
            status = f"{row[2]}"
            if row[3]:
                status += f" (error: {row[3][:50]})"
            print(f"  Task {row[0]}: {status}")
            print(f"    Input: {row[1][:80]}...")
