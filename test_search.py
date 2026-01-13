#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from ace_rm.ace_framework import ACE_Memory

# Test with a specific session that has data
session_id = "307c922b-e995-4687-8b4c-ec961bd38489"
memory = ACE_Memory(session_id=session_id)

print(f"Testing search for session: {session_id}")
print(f"Total documents: {len(memory.get_all())}\n")

# Test various queries
test_queries = [
    "bottles puzzle",
    "logic puzzle",
    "constraint satisfaction",
    "危険",
    "safety puzzle"
]

for query in test_queries:
    results = memory.search(query, k=3)
    print(f"Query: '{query}'")
    print(f"  Results: {len(results)}")
    if results:
        for i, doc in enumerate(results[:2], 1):
            print(f"    {i}. {doc[:100]}...")
    print()
