import os
import pytest
import time
import json
import sqlite3
import unittest
import uuid
from unittest.mock import MagicMock
from ace_rm.ace_framework import ACE_Memory, BackgroundWorker
from langchain_core.messages import HumanMessage, AIMessage

@pytest.fixture
def memory():
    # Use a unique session ID for each test run to ensure isolation
    session_id = f"test_worker_{uuid.uuid4()}"
    mem = ACE_Memory(session_id=session_id)
    yield mem
    # Teardown
    mem.clear()

def test_queue_operations(memory):
    # 1. Enqueue
    memory.enqueue_task("User says hi", "Agent says hello")
    
    # 2. Fetch
    task = memory.fetch_pending_task()
    assert task is not None
    assert task['user_input'] == "User says hi"
    assert task['agent_output'] == "Agent says hello"
    assert task['status'] == 'pending'
    
    # 3. Mark Processing
    memory.mark_task_processing(task['id'])
    # fetch_pending_task only fetches 'pending', so it should return None now
    assert memory.fetch_pending_task() is None
    
    # 4. Mark Complete
    memory.mark_task_complete(task['id'])
    
    # Verify via direct DB access
    with sqlite3.connect(memory.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM task_queue WHERE id = ?", (task['id'],))
        row = cursor.fetchone()
        assert row[0] == 'done'

def test_background_worker_process_success(memory):
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
    
    worker = BackgroundWorker(llm=mock_llm, memory_session_id=memory.session_id)
    
    # Enqueue a task
    memory.enqueue_task("Calculate 1+1", "The answer is 2")
    task = memory.fetch_pending_task()
    
    # Manually trigger process_task
    worker.process_task(task)
    
    # Verify LLM was called
    mock_llm.invoke.assert_called_once()
    
    # Verify Memory was updated
    docs = memory.search("Test Lesson", k=1)
    assert len(docs) > 0
    assert "Test Lesson Learned" in docs[0]
    
    # Verify Task is marked done
    with sqlite3.connect(memory.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status FROM task_queue WHERE id = ?", (task['id'],))
        row = cursor.fetchone()
        assert row[0] == 'done'

def test_background_worker_process_failure(memory):
    """Test that task is marked 'failed' when LLM raises an exception."""
    mock_llm = MagicMock()
    
    # We patch call_llm_with_retry to avoid tenacity retries and directly raise
    with unittest.mock.patch('ace_rm.utils.call_llm_with_retry') as mock_call:
        mock_call.side_effect = Exception("API Connection Error")
        
        worker = BackgroundWorker(llm=mock_llm, memory_session_id=memory.session_id)
        
        memory.enqueue_task("Fail me", "Ok")
        task = memory.fetch_pending_task()
        
        worker.process_task(task)
        
        # Verify Task is marked failed
        with sqlite3.connect(memory.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, error_msg FROM task_queue WHERE id = ?", (task['id'],))
            row = cursor.fetchone()
            assert row[0] == 'failed'
            assert "API Connection Error" in row[1]

def test_background_worker_invalid_json(memory):
    """Test handling of invalid JSON response from LLM."""
    mock_llm = MagicMock()
    # Simulate Non-JSON output
    mock_llm.invoke.return_value = AIMessage(content="This is not JSON.")
    
    worker = BackgroundWorker(llm=mock_llm, memory_session_id=memory.session_id)
    
    memory.enqueue_task("Bad JSON", "Ok")
    task = memory.fetch_pending_task()
    
    worker.process_task(task)
    
    # Verify Task is marked failed (due to JSON decode error)
    with sqlite3.connect(memory.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT status, error_msg FROM task_queue WHERE id = ?", (task['id'],))
        row = cursor.fetchone()
        assert row[0] == 'failed'
        # Error message should mention JSON or something related
        assert row[1] is not None