#!/usr/bin/env python3
"""
Reproduction script for LTM duplication and stale index issues.
"""
import sys
sys.path.insert(0, 'src')
from ace_rm.ace_framework import ACE_Memory
import time

def test_duplication_and_stale_index():
    print("=" * 80)
    print("TEST: LTM DUPLICATION AND STALE INDEX")
    print("=" * 80)

    # 1. Setup separate memory instances to simulate Main App vs Background Worker
    # We use the same shared DB path
    
    print("\n[Step 1] Initializing Main App Memory (Memory A)...")
    mem_a = ACE_Memory()
    initial_count = len(mem_a.get_all())
    print(f"  Docs in Memory A: {initial_count}")
    
    print("\n[Step 2] Initializing Background Worker Memory (Memory B)...")
    mem_b = ACE_Memory()
    
    # 2. Test Duplication
    print("\n" + "-" * 40)
    print("TEST 1: DUPLICATION CHECK")
    print("-" * 40)
    
    test_content = "Unique content for duplication test " + str(time.time())
    
    print(f"  Adding content via Memory B: '{test_content}'")
    mem_b.add(test_content, entities=["TestEntity"], problem_class="TestClass")
    
    print("  Adding SAME content again via Memory B...")
    mem_b.add(test_content, entities=["TestEntity"], problem_class="TestClass")
    
    # Check count in Memory B (it should have reloaded index/db on add)
    docs_b = mem_b.get_all()
    duplicates = [d for d in docs_b if d['content'] == test_content]
    print(f"  Copies found in Memory B: {len(duplicates)}")
    
    if len(duplicates) > 1:
        print("  >>> FAIL: Content was duplicated!")
    else:
        print("  >>> PASS: Content was deduplicated (unexpected per current analysis).")

    # 3. Test Stale Index (Retrieval)
    print("\n" + "-" * 40)
    print("TEST 2: STALE INDEX / RETRIEVAL CHECK")
    print("-" * 40)
    
    # Memory A hasn't done anything since initialization.
    # It should search for the new content added by Memory B.
    
    print("  Searching in Memory A for the new content...")
    results_a = mem_a.search(test_content, k=5)
    
    # Check if the specific content is found
    # search return lists of content strings
    found = any(test_content in r for r in results_a)
    
    if found:
        print("  >>> PASS: Memory A found the content added by Memory B.")
    else:
        print("  >>> FAIL: Memory A did NOT find the content added by Memory B.")
        print(f"  Results found: {len(results_a)}")
        if results_a:
            print(f"  Top result: {results_a[0][:50]}...")
            
    # Force reload on Memory A and try again
    print("\n  Forcing index reload on Memory A...")
    # Simulate what we might implement: check file vs memory
    # For now, just re-init or call internal reload if possible
    # Hack: Creating NEW instance C to verify it's definitely on disk
    mem_c = ACE_Memory()
    results_c = mem_c.search(test_content, k=5)
    found_c = any(test_content in r for r in results_c)
    
    if found_c:
        print("  >>> VERIFY: New Instance C found the content (confirmed it's on disk).")
    else:
        print("  >>> CRITICAL: Even new instance C couldn't find it. Index functionality might be broken.")

if __name__ == "__main__":
    test_duplication_and_stale_index()
