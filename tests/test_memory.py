import os
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

def test_add_batch(memory):
    items = [
        {"content": "Batch item 1", "entities": ["E1"], "problem_class": "P1"},
        {"content": "Batch item 2", "entities": ["E2"], "problem_class": "P2"}
    ]
    memory.add_batch(items)

    all_docs = memory.get_all()
    assert len(all_docs) == 2

    results = memory.search("Batch item", k=2)
    assert len(results) == 2

def test_update_document(memory):
    memory.add("Original content", entities=["old"], problem_class="old_p")
    doc_id = memory.get_all()[0]['id']

    memory.update_document(doc_id, "New content", entities=["new"], problem_class="new_p")

    doc = memory.get_document_by_id(doc_id)
    assert doc['content'] == "New content"
    assert "new" in doc['entities']

    # Vector search should find new content
    results = memory.search("New content", k=1)
    assert "New content" in results[0]

    # And not the old one (ideally, but similarity might still be high, so check specifically)
    results_old = memory.search("Original content", k=1)
    if results_old:
        assert "New content" in results_old[0] # Because the doc was updated

def test_fts_sanitization(memory):
    # This should not crash
    results = memory.search("unexpected'quote\"", k=1)
    assert isinstance(results, list)
