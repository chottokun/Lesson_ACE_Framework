import pytest
import os
import uuid
from ace_rm.ace_framework import build_ace_agent, ACE_Memory
from langchain_openai import ChatOpenAI

def test_graph_build():
    session_id = f"test_graph_{uuid.uuid4()}"
    memory = None
    
    # Mock Memory and LLM
    try:
        memory = ACE_Memory(session_id=session_id)
        llm = ChatOpenAI(api_key="dummy", base_url="http://localhost")
        
        graph = build_ace_agent(llm, memory)
        assert graph is not None
    except Exception as e:
        pytest.fail(f"Graph build failed: {e}")
    finally:
        if memory:
            memory.clear()
