
import os
import sys
import time
import uuid

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_openai import ChatOpenAI

from ace_rm.config import MODEL_NAME, BASE_URL, OPENAI_API_KEY, LLM_TEMPERATURE
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue
from ace_rm.workers.background import BackgroundWorker

# Simple LLM Mock or Tracker if needed, but for manual test we just run it and check logs/DB
# Ideally we want to see only 1 LLM call.

def test_background_optimization():
    session_id = f"test_bg_opt_{uuid.uuid4()}"
    print(f"Session ID: {session_id}")
    
    memory = ACE_Memory(session_id=session_id)
    queue = TaskQueue(session_id=session_id)
    
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
        temperature=LLM_TEMPERATURE
    )
    
    worker = BackgroundWorker(llm, memory, queue, interval=0.5)
    worker.start()
    
    try:
        # 1. Add a tasks that should trigger a NEW document
        print("\n--- Test 1: New Knowledge ---")
        user_input = "Pythonのリスト内包表記について教えて"
        agent_output = "リスト内包表記は、既存のリストから新しいリストを作る簡潔な方法です。[expression for item in list]"
        
        queue.enqueue_task(user_input, agent_output)
        
        # Wait for processing
        print("Waiting for worker...")
        time.sleep(5)
        
        # Check Task Status
        tasks = queue.get_tasks()
        latest_task = tasks[0]
        print(f"Task Status: {latest_task['status']}")
        if latest_task['status'] == 'failed':
            print(f"Error: {latest_task['error_msg']}")
        
        # Check Memory
        docs = memory.get_all()
        print(f"Docs in memory: {len(docs)}")
        if docs:
            print(f"Doc Content: {docs[0]['content'][:100]}...")

        if len(docs) == 0:
            print("Failed to add document.")
        else:
            print("Successfully added document.")

        # 2. Add a task that should trigger an UPDATE (refinement)
        print("\n--- Test 2: Update Knowledge ---")
        user_input_2 = "リスト内包表記で条件分岐を使うには？"
        agent_output_2 = "if文を後ろに追加します。[x for x in list if condition]"
        
        queue.enqueue_task(user_input_2, agent_output_2)
        
        time.sleep(5)
        
        docs_after = memory.get_all()
        print(f"Docs in memory after update: {len(docs_after)}")
        # Ideally count shouldn't increase if it was an UPDATE, or if it was New it will be 2.
        # The prompt decides.
        for d in docs_after:
            print(f"Doc ID {d['id']}: {d['content'][:100]}...")

    finally:
        worker.stop()
        worker.join()
        # memory.clear() # Keep for inspection if needed, or clear

if __name__ == "__main__":
    # Ensure prompts are loaded correctly for the test
    # (In isolated script run, we might need to rely on installed package or path insertion)
    print(f"Testing with language: {os.environ.get('ACE_LANG', 'ja')}")
    test_background_optimization()
