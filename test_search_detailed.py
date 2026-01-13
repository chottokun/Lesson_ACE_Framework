#!/usr/bin/env python3
"""
Comprehensive test suite for ACE Memory search functionality
"""
import sys
sys.path.insert(0, 'src')
from ace_rm.ace_framework import ACE_Memory
import json

def test_memory_search():
    # Test with global shared memory
    memory = ACE_Memory()
    
    print("=" * 80)
    print("ACE MEMORY SEARCH DIAGNOSTIC TEST")
    print("=" * 80)
    
    # 1. Check all documents in memory
    all_docs = memory.get_all()
    print(f"\n[1] TOTAL DOCUMENTS IN MEMORY: {len(all_docs)}")
    print("-" * 80)
    
    if all_docs:
        for i, doc in enumerate(all_docs, 1):
            print(f"\nDocument {i} (ID: {doc['id']}):")
            print(f"  Class: {doc['problem_class']}")
            print(f"  Entities: {doc['entities']}")
            print(f"  Content (first 200 chars): {doc['content'][:200]}...")
            print(f"  Timestamp: {doc['timestamp']}")
    else:
        print("  >>> WARNING: No documents found in memory!")
        return
    
    # 2. Test vector search behavior
    print("\n" + "=" * 80)
    print("[2] VECTOR SEARCH TESTS")
    print("-" * 80)
    
    test_queries = [
        ("bottles puzzle", "Specific query about bottles"),
        ("logic reasoning", "General query about logic"),
        ("危険", "Japanese keyword from stored content"),
        ("mathematics", "Unrelated topic"),
        ("constraint satisfaction problem", "Abstract problem class"),
    ]
    
    for query, description in test_queries:
        print(f"\nQuery: '{query}' ({description})")
        results = memory.search(query, k=3)
        print(f"  Results: {len(results)} documents")
        
        if results:
            for j, doc in enumerate(results, 1):
                # Show first 150 chars
                preview = doc.replace("\n", " ")[:150]
                print(f"    {j}. {preview}...")
        else:
            print("    >>> No results")
    
    # 3. Test FTS vs Vector search separately
    print("\n" + "=" * 80)
    print("[3] SEARCH ALGORITHM BREAKDOWN")
    print("-" * 80)
    
    # Check FAISS index status
    print(f"\nFAISS Index:")
    print(f"  Total vectors: {memory.index.ntotal}")
    print(f"  Dimension: {memory.dimension}")
    
    # Test a specific query with detailed breakdown
    test_query = "bottles and poison"
    print(f"\n\nDetailed search for: '{test_query}'")
    
    # Vector search
    if memory.index.ntotal > 0:
        query_vec = memory.encoder.encode([test_query])
        import numpy as np
        D, I = memory.index.search(np.array(query_vec).astype('float32'), 3)
        print(f"\nVector Search Results (distances and IDs):")
        for dist, idx in zip(D[0], I[0]):
            if idx >= 0:
                print(f"  ID={idx}, Distance={dist:.4f}")
    
    # FTS search
    print(f"\nFTS (Full-Text Search) Results:")
    import sqlite3
    with sqlite3.connect(memory.db_path) as conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT rowid, content FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT 3",
                (test_query,)
            )
            for rowid, content in cursor.fetchall():
                print(f"  RowID={rowid}, Content={content[:100]}...")
        except Exception as e:
            print(f"  FTS Error: {e}")
    
    # 4. Check for potential issues
    print("\n" + "=" * 80)
    print("[4] POTENTIAL ISSUES CHECK")
    print("-" * 80)
    
    # Check if all documents have vectors
    print(f"\nVector count vs Document count:")
    print(f"  Documents in DB: {len(all_docs)}")
    print(f"  Vectors in FAISS: {memory.index.ntotal}")
    
    if len(all_docs) != memory.index.ntotal:
        print("  >>> WARNING: Mismatch between documents and vectors!")
    
    # Check for duplicate or very similar content
    if len(all_docs) > 1:
        print(f"\nContent similarity check:")
        for i in range(min(3, len(all_docs))):
            for j in range(i+1, min(3, len(all_docs))):
                doc1 = all_docs[i]['content'][:100]
                doc2 = all_docs[j]['content'][:100]
                if doc1 == doc2:
                    print(f"  >>> WARNING: Doc {i+1} and Doc {j+1} have identical content!")

if __name__ == "__main__":
    test_memory_search()
