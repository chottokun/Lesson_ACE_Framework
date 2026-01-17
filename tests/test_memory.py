import pytest
import uuid
from ace_rm.ace_framework import ACE_Memory

@pytest.fixture
def memory():
    session_id = f"test_memory_{uuid.uuid4()}"
    mem = ACE_Memory(session_id=session_id)
    yield mem
    # Teardown
    mem.clear()

def test_add_and_search(memory):
    memory.add("The capital of France is Paris.", entities=["France", "Paris"], problem_class="Geography")
    memory.add("The capital of Japan is Tokyo.", entities=["Japan", "Tokyo"], problem_class="Geography")
    
    # Test Vector Search (Semantic)
    results = memory.search("France capital", k=1)
    assert len(results) >= 1
    assert "Paris" in results[0]
    
    # Test Keyword Search (FTS)
    results_fts = memory.search("Tokyo", k=1)
    assert len(results_fts) >= 1
    assert "Tokyo" in results_fts[0]
    
    # Test Get All
    all_docs = memory.get_all()
    assert len(all_docs) == 2
