import pytest
import os
from ace_rm.ace_framework import build_ace_agent, ACE_Memory
from langchain_openai import ChatOpenAI

def test_graph_build():
    db_path = "test_graph.db"
    index_path = "test_graph.faiss"
    
    # Mock Memory and LLM
    try:
        memory = ACE_Memory(db_path=db_path, index_path=index_path)
        llm = ChatOpenAI(api_key="dummy", base_url="http://localhost")
        
        graph = build_ace_agent(llm, memory)
        assert graph is not None
    except Exception as e:
        pytest.fail(f"Graph build failed: {e}")
    finally:
        if os.path.exists(db_path): os.remove(db_path)
        if os.path.exists(index_path): os.remove(index_path)
