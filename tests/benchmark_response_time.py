"""
ACE-RM Response Time and LLM Call Benchmark

This script measures:
1. Total response time for different types of queries
2. Number of LLM calls per interaction
3. Time breakdown by component (Curator, Agent, Reflector)

Usage:
    python tests/benchmark_response_time.py
"""

import os
import sys
import time
import uuid
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ace_rm.config import MODEL_NAME, BASE_URL, OPENAI_API_KEY, LLM_TEMPERATURE
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue
from ace_rm.agent.graph import build_ace_agent


class LLMCallTracker:
    """Tracks LLM invocations for analysis using LangChain callbacks."""
    
    def __init__(self):
        self.call_count = 0
        self.call_times: List[float] = []
        self.total_llm_time = 0.0
        self._current_start: float = 0.0
    
    def on_start(self):
        """Called before LLM invocation."""
        self.call_count += 1
        self._current_start = time.perf_counter()
    
    def on_end(self):
        """Called after LLM invocation."""
        elapsed = time.perf_counter() - self._current_start
        self.call_times.append(elapsed)
        self.total_llm_time += elapsed
    
    def reset(self):
        self.call_count = 0
        self.call_times = []
        self.total_llm_time = 0.0
    
    def report(self) -> Dict[str, Any]:
        return {
            "call_count": self.call_count,
            "total_llm_time": self.total_llm_time,
            "avg_llm_time": self.total_llm_time / self.call_count if self.call_count > 0 else 0,
            "call_times": self.call_times
        }


class TrackingChatOpenAI(ChatOpenAI):
    """ChatOpenAI wrapper that tracks call counts."""
    
    _tracker: LLMCallTracker = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, tracker: LLMCallTracker, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_tracker', tracker)
    
    def invoke(self, *args, **kwargs):
        tracker = object.__getattribute__(self, '_tracker')
        if tracker:
            tracker.on_start()
        result = super().invoke(*args, **kwargs)
        if tracker:
            tracker.on_end()
        return result


def run_benchmark(test_queries: List[Dict[str, str]], num_runs: int = 3) -> Dict[str, Any]:
    """
    Run benchmark with multiple test queries.
    
    Args:
        test_queries: List of {"name": "...", "query": "..."} dicts
        num_runs: Number of runs per query for averaging
    
    Returns:
        Benchmark results
    """
    session_id = f"benchmark_{uuid.uuid4()}"
    
    print(f"[Benchmark] Initializing session: {session_id}")
    print(f"[Benchmark] Using model: {MODEL_NAME}")
    print(f"[Benchmark] Base URL: {BASE_URL}")
    print("-" * 60)
    
    # Setup tracker and LLM
    tracker = LLMCallTracker()
    
    llm = TrackingChatOpenAI(
        tracker=tracker,
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
        temperature=LLM_TEMPERATURE
    )
    
    # Initialize components
    memory = ACE_Memory(session_id=session_id)
    task_queue = TaskQueue(session_id=session_id)
    ace_app = build_ace_agent(llm, memory, task_queue)
    
    results = []
    
    try:
        for test in test_queries:
            test_name = test["name"]
            query = test["query"]
            
            query_results = []
            
            for run in range(num_runs):
                tracker.reset()
                
                initial_state = {
                    "messages": [HumanMessage(content=query)],
                    "retry_count": 0,
                    "context_docs": [],
                    "extracted_entities": [],
                    "problem_class": ""
                }
                
                start_time = time.perf_counter()
                final_state = ace_app.invoke(initial_state)
                total_time = time.perf_counter() - start_time
                
                llm_report = tracker.report()
                
                run_result = {
                    "run": run + 1,
                    "total_time": total_time,
                    "llm_call_count": llm_report["call_count"],
                    "llm_total_time": llm_report["total_llm_time"],
                    "non_llm_time": total_time - llm_report["total_llm_time"]
                }
                query_results.append(run_result)
                
                print(f"[{test_name}] Run {run+1}: {total_time:.2f}s total, {llm_report['call_count']} LLM calls ({llm_report['total_llm_time']:.2f}s LLM time)")
            
            # Calculate averages
            avg_total = sum(r["total_time"] for r in query_results) / len(query_results)
            avg_llm_calls = sum(r["llm_call_count"] for r in query_results) / len(query_results)
            avg_llm_time = sum(r["llm_total_time"] for r in query_results) / len(query_results)
            
            results.append({
                "test_name": test_name,
                "query": query,
                "runs": query_results,
                "avg_total_time": avg_total,
                "avg_llm_calls": avg_llm_calls,
                "avg_llm_time": avg_llm_time
            })
            
            print(f"[{test_name}] Average: {avg_total:.2f}s, {avg_llm_calls:.1f} LLM calls")
            print("-" * 60)
    
    finally:
        # Cleanup
        print("[Benchmark] Cleaning up...")
        memory.clear()
    
    return {
        "session_id": session_id,
        "model": MODEL_NAME,
        "results": results
    }


def print_summary(benchmark_results: Dict[str, Any]):
    """Print a summary table of results."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Model: {benchmark_results['model']}")
    print("-" * 70)
    print(f"{'Test Name':<30} {'Avg Time (s)':<15} {'LLM Calls':<12} {'LLM Time (s)':<12}")
    print("-" * 70)
    
    for result in benchmark_results["results"]:
        print(f"{result['test_name']:<30} {result['avg_total_time']:<15.2f} {result['avg_llm_calls']:<12.1f} {result['avg_llm_time']:<12.2f}")
    
    print("=" * 70)


if __name__ == "__main__":
    # Test queries of different complexity
    test_queries = [
        {
            "name": "simple_greeting",
            "query": "こんにちは"
        },
        {
            "name": "simple_question",
            "query": "今日の天気は？"
        },
        {
            "name": "medium_question",
            "query": "Pythonでリストをソートする方法を教えてください"
        },
        {
            "name": "complex_question",
            "query": "機械学習において、過学習（オーバーフィッティング）を防ぐための一般的な手法と、それぞれの長所・短所について詳しく説明してください"
        },
        {
            "name": "follow_up_context",
            "query": "それを実装する具体的なコード例を示してください"
        }
    ]
    
    print("=" * 70)
    print("ACE-RM Response Time Benchmark")
    print("=" * 70)
    
    results = run_benchmark(test_queries, num_runs=2)
    print_summary(results)
