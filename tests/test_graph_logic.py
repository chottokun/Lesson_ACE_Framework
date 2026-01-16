import pytest
import uuid
import json
from ace_rm.ace_framework import build_ace_agent, ACE_Memory, TaskQueue
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from unittest.mock import MagicMock

@pytest.fixture
def memory():
    session_id = f"test_graph_{uuid.uuid4()}"
    mem = ACE_Memory(session_id=session_id)
    yield mem
    mem.clear()

def test_curator_node_fast_path(memory):
    mock_llm = MagicMock()
    # Fast path should skip LLM
    import os
    os.environ["ACE_CURATOR_SKIP_SIMPLE"] = "true"

    graph = build_ace_agent(mock_llm, memory, use_tools=False)

    # Simple message
    state = {"messages": [HumanMessage(content="Hello")], "retry_count": 0}
    final_state = graph.invoke(state)

    # Should not have called LLM for intent analysis (but agent will be called)
    # Actually build_ace_agent compiles a graph. curator is the first node.
    # In my build_ace_agent, curator is always called.

    # Check if LLM was called only once (by agent, not curator)
    assert mock_llm.invoke.call_count == 1
    os.environ["ACE_CURATOR_SKIP_SIMPLE"] = "false"

def test_curator_node_full_path(memory):
    mock_llm = MagicMock()
    # Mock curator response
    curator_res = json.dumps({
        "entities": ["test"],
        "problem_class": "testing",
        "search_query": "test query"
    })
    # Mock agent response
    agent_res = "I am an agent"

    mock_llm.invoke.side_effect = [
        AIMessage(content=curator_res),
        AIMessage(content=agent_res)
    ]

    graph = build_ace_agent(mock_llm, memory, use_tools=False)

    state = {"messages": [HumanMessage(content="A complex question about testing")], "retry_count": 0}
    final_state = graph.invoke(state)

    assert final_state["extracted_entities"] == ["test"]
    assert final_state["problem_class"] == "testing"
    assert mock_llm.invoke.call_count == 2

def test_system_message_no_accumulation(memory):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Agent Response")

    # Manually add a curator system message
    memory.add("Previous Context")

    graph = build_ace_agent(mock_llm, memory, use_tools=False)

    # Initial state with a system message that looks like curator's
    sys_msg = SystemMessage(content="--- Retrieved Context ---\nPrevious Context\n-----------------------")
    state = {
        "messages": [sys_msg, HumanMessage(content="Hello")],
        "retry_count": 0
    }

    # Mock curator to return no new docs for simplicity (fast path)
    import os
    os.environ["ACE_CURATOR_SKIP_SIMPLE"] = "true"

    final_state = graph.invoke(state)

    # The curator node should have filtered out the old system message.
    # But since it's a simple message and we have docs in memory, it might add a new one.

    # Count system messages in final state
    sys_msgs = [m for m in final_state["messages"] if isinstance(m, SystemMessage)]
    # Should only have one (the new one or none if not found)
    assert len(sys_msgs) <= 1
    os.environ["ACE_CURATOR_SKIP_SIMPLE"] = "false"
