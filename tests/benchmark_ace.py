
import time
import os
import uuid
from ace_rm.ace_framework import ACE_Memory

def benchmark_memory(n_docs=50):
    print(f"--- Benchmarking ACE_Memory with {n_docs} documents ---")
    session_id = f"bench_{uuid.uuid4()}"
    memory = ACE_Memory(session_id=session_id)
    
    # 1. Benchmark Single Insertion
    start_time = time.time()
    for i in range(n_docs):
        memory.add(
            f"Single insert doc {i}. Logic check.",
            entities=[f"entity_{i}"],
            problem_class="BenchmarkSingle"
        )
    end_time = time.time()
    avg_insert = (end_time - start_time) / n_docs
    print(f"Single Insertion: Total={end_time - start_time:.4f}s, Avg={avg_insert:.4f}s/doc")
    
    # 2. Benchmark Batch Insertion
    batch_items = [
        {
            "content": f"Batch insert doc {i}. Speed check.",
            "entities": [f"batch_entity_{i}"],
            "problem_class": "BenchmarkBatch"
        }
        for i in range(n_docs)
    ]
    start_time = time.time()
    memory.add_batch(batch_items)
    end_time = time.time()
    avg_batch = (end_time - start_time) / n_docs
    print(f"Batch Insertion: Total={end_time - start_time:.4f}s, Avg={avg_batch:.4f}s/doc")
    print(f"Speedup Factor: {avg_insert / avg_batch:.2f}x")

    # 3. Benchmark Search
    start_time = time.time()
    for i in range(n_docs):
        results = memory.search(f"insert doc {i}", k=3)
    end_time = time.time()
    avg_search = (end_time - start_time) / n_docs
    print(f"Search: Total={end_time - start_time:.4f}s, Avg={avg_search:.4f}s/query")
    
    # Clean up
    memory.clear()

if __name__ == "__main__":
    benchmark_memory(20)
