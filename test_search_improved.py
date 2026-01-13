#!/usr/bin/env python3
"""
Test the improved search algorithm with distance threshold filtering
"""
import sys
sys.path.insert(0, 'src')
from ace_rm.ace_framework import ACE_Memory

def test_improved_search():
    memory = ACE_Memory()
    
    print("=" * 80)
    print("TESTING IMPROVED SEARCH WITH DISTANCE THRESHOLD")
    print("=" * 80)
    
    # Test queries - some relevant, some not
    test_cases = [
        ("bottles puzzle", "Should find bottle-related documents"),
        ("logic puzzle", "Should find logic/puzzle documents"),
        ("高齢者 労働", "Should find elderly labor documents"),
        ("mathematics", "Should find math-related or return few/no results"),
        ("completely unrelated random text xyz123", "Should return no or very few results"),
        ("constraint satisfaction", "Should find CSP documents"),
        ("Python programming", "Should find Python-related documents"),
    ]
    
    for query, description in test_cases:
        print(f"\nQuery: '{query}'")
        print(f"Expectation: {description}")
        results = memory.search(query, k=3)
        print(f"Results: {len(results)} documents")
        
        if results:
            for i, doc in enumerate(results, 1):
                preview = doc.replace("\n", " ")[:120]
                print(f"  {i}. {preview}...")
        else:
            print("  >>> No relevant results (GOOD if query is unrelated)")
        print("-" * 80)
    
    # Test with different thresholds
    print("\n" + "=" * 80)
    print("TESTING DIFFERENT DISTANCE THRESHOLDS")
    print("=" * 80)
    
    query = "mathematics"
    for threshold in [1.5, 2.0, 2.5, 3.0]:
        results = memory.search(query, k=3, distance_threshold=threshold)
        print(f"\nThreshold={threshold}: {len(results)} results")
        if results:
            for doc in results[:2]:
                print(f"  - {doc[:80]}...")

if __name__ == "__main__":
    test_improved_search()
