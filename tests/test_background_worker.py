import pytest
import json
import sqlite3
import uuid
from unittest.mock import MagicMock
from ace_rm.ace_framework import ACE_Memory, TaskQueue, BackgroundWorker
from langchain_core.messages import AIMessage

@pytest.fixture
def memory_and_queue():
    # Use a unique session ID for each test run to ensure isolation
    session_id = f"test_worker_{uuid.uuid4()}"
    mem = ACE_Memory(session_id=session_id)
    queue = TaskQueue(session_id=session_id)
    yield mem, queue
    # Teardown
    mem.clear()

def test_queue_operations(memory_and_queue):
    mem, queue = memory_and_queue
    # 1. Enqueue
    queue.enqueue_task("User says hi", "Agent says hello")
    
    # 2. Fetch
    task = queue.fetch_pending_task()
    assert task is not None
    assert task['user_input'] == "User says hi"
    assert task['agent_output'] == "Agent says hello"
    assert task['status'] == 'pending'
    
    # 3. Mark Processing
    queue.mark_task_processing(task['id'])
    # fetch_pending_task only fetches 'pending', so it should return None now
    assert queue.fetch_pending_task() is None
    
    # 4. Mark Complete
    queue.mark_task_complete(task['id'])
    
    # Verify via direct DB access
    with sqlite3.connect(queue.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM task_queue WHERE id = ?", (task['id'],))
        row = cursor.fetchone()
        assert row[0] == 'done'

def test_background_worker_process_success(memory_and_queue):
    mem, queue = memory_and_queue
    # Mock LLM
    mock_llm = MagicMock()
    # Mock response for Reflector
    mock_response_content = json.dumps({
        "analysis": "Test Lesson Learned",
        "entities": ["TestEntity"],
        "problem_class": "TestClass",
        "should_store": True
    })
    mock_llm.invoke.return_value = AIMessage(content=mock_response_content)
    
    worker = BackgroundWorker(llm=mock_llm, memory=mem, task_queue=queue)
    
    # Enqueue a task
    queue.enqueue_task("Calculate 1+1", "The answer is 2")
    task = queue.fetch_pending_task()
    
    # Manually trigger process_task
    worker.process_task(task)
    
    # Verify LLM was called
    mock_llm.invoke.assert_called()
    
    # Verify Memory was updated
    docs = mem.search("Test Lesson", k=1)
    assert len(docs) > 0
    assert "Test Lesson Learned" in docs[0]
    
    # Verify Task is marked done
    with sqlite3.connect(queue.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM task_queue WHERE id = ?", (task['id'],))
        row = cursor.fetchone()
        assert row[0] == 'done'

def test_background_worker_process_failure(memory_and_queue):
    """Test that task is marked 'failed' when LLM raises an exception."""
    mem, queue = memory_and_queue
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = Exception("API Connection Error")
    
    worker = BackgroundWorker(llm=mock_llm, memory=mem, task_queue=queue)
    
    queue.enqueue_task("Fail me", "Ok")
    task = queue.fetch_pending_task()
    
    worker.process_task(task)
    
    # Verify Task is marked failed
    with sqlite3.connect(queue.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, error_msg FROM task_queue WHERE id = ?", (task['id'],))
        row = cursor.fetchone()
        assert row[0] == 'failed'
        assert "API Connection Error" in row[1]

def test_background_worker_invalid_json(memory_and_queue):
    """Test handling of invalid JSON response from LLM."""
    mem, queue = memory_and_queue
    mock_llm = MagicMock()
    # Simulate Non-JSON output
    mock_llm.invoke.return_value = AIMessage(content="This is not JSON.")
    
    worker = BackgroundWorker(llm=mock_llm, memory=mem, task_queue=queue)
    
    queue.enqueue_task("Bad JSON", "Ok")
    task = queue.fetch_pending_task()
    
    worker.process_task(task)
    
    # Verify Task is marked failed (due to JSON decode error)
    with sqlite3.connect(queue.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, error_msg FROM task_queue WHERE id = ?", (task['id'],))
        row = cursor.fetchone()
        assert row[0] == 'failed'
        # Error message should mention JSON or something related
        assert row[1] is not None