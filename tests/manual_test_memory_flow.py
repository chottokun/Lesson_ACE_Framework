
"""
Manual Test Script for ACE Framework Memory Flow (Advanced Version)

This script verifies the end-to-end cognitive loop of the ACE agent:
1. Message processing with advanced prompts.
2. Curator: Abstract + Specific search query generation and context retrieval.
3. Agent: Context-aware response generation.
4. Reflector: Structural analysis (MFR) and generalized knowledge storage.

Usage:
    uv run python tests/manual_test_memory_flow.py
"""

import os
import sys
import shutil
import sqlite3
import time
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# Ensure src is in path if running directly from project root or tests/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(os.path.join(project_root, "src"))

from ace_rm.ace_framework import build_ace_agent, ACE_Memory, MODEL_NAME, BASE_URL, BackgroundWorker

# Load environment variables
load_dotenv()

def cleanup_db():
    """Removes existing memory files to ensure a clean test state."""
    print("\n[Setup] Cleaning up existing memory files...")
    files_to_remove = ["ace_memory.db", "ace_memory.faiss"]
    for file in files_to_remove:
        path = os.path.join(project_root, file)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed: {file}")

def run_test():
    cleanup_db()
    
    print("\n--- Starting ACE Memory Flow Test ---")
    
    # 1. Initialize Components
    memory = ACE_Memory()
    api_key = os.environ.get("SAKURA_API_KEY", "dummy")
    
    llm = ChatOpenAI(
        model=MODEL_NAME, 
        api_key=api_key, 
        base_url=BASE_URL,
        temperature=0
    )
    
    ace_app = build_ace_agent(llm, memory)

    # Start Background Worker for Async Reflection
    worker = BackgroundWorker(memory, llm)
    worker.start()
    
    # Step 1: Structural Learning (Storage)
    # ------------------------------------
    print("\n" + "="*60)
    print("[Step 1] Learning: Solving the 3L/5L Water Jug Problem")
    print("="*60)
    
    user_msg_1 = "3リットルと5リットルのバケツを使って、正確に4リットルの水を量る方法を教えてください。"
    print(f"User: {user_msg_1}")
    
    state_1 = {
        "messages": [HumanMessage(content=user_msg_1)], 
        "retry_count": 0,
        "context_docs": [],
        "extracted_entities": [],
        "problem_class": ""
    }
    
    try:
        final_state_1 = ace_app.invoke(state_1)
        print(f"\nAI Response: {final_state_1['messages'][-1].content[:200]}...")
        print(f"\n[Reflector] Should Store: {final_state_1.get('should_store')}")
        if final_state_1.get('should_store'):
            print(f"[Reflector] Lesson Learned (First 150 chars):\n{final_state_1.get('lesson_learned')[:150]}...")
            print("\n[Test] Waiting 5 seconds for background worker to process...")
            time.sleep(5) # Allow worker to process
    except Exception as e:
        print(f"ERROR in Step 1: {e}")
        worker.stop()
        return

    # Step 2: Transfer Learning (Retrieval & Application)
    # --------------------------------------------------
    print("\n" + "="*60)
    print("[Step 2] Retrieval: Applying strategy to 5L/8L jugs")
    print("="*60)
    
    user_msg_2 = "同じ考え方を使って、5Lと8Lのバケツで2Lを量ることはできますか？手順も教えてください。"
    print(f"User: {user_msg_2}")
    
    # Simulate a new state but memory is persistent in the DB
    state_2 = {
        "messages": [HumanMessage(content=user_msg_2)], 
        "retry_count": 0,
        "context_docs": [],
        "extracted_entities": [],
        "problem_class": ""
    }
    
    try:
        final_state_2 = ace_app.invoke(state_2)
        
        # Verify Retrieval
        retrieved = final_state_2.get('context_docs', [])
        print(f"\n[Curator] Context Docs Retrieved: {len(retrieved)}")
        if retrieved:
            print(f"[Curator] Found knowledge about: {final_state_2.get('problem_class')}")
        
        print(f"\nAI Response: {final_state_2['messages'][-1].content[:300]}...")
        
    except Exception as e:
        print(f"ERROR in Step 2: {e}")
    finally:
        worker.stop()

    print("\n--- Test Completed ---")

if __name__ == "__main__":
    run_test()
