#!/usr/bin/env python3
"""
Test to verify that retrieved context is actually used in agent responses
"""
import sys
sys.path.insert(0, 'src')
from ace_rm.ace_framework import build_ace_agent, ACE_Memory
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os

def test_context_usage():
    print("=" * 80)
    print("TESTING CONTEXT USAGE IN AGENT RESPONSES")
    print("=" * 80)
    
    # Use shared memory (should have existing documents)
    memory = ACE_Memory()
    
    docs = memory.get_all()
    print(f"\nMemory has {len(docs)} documents")
    
    if len(docs) == 0:
        print("WARNING: No documents in memory to test with")
        return
    
    # Show sample documents
    print("\nSample documents in memory:")
    for i, doc in enumerate(docs[:3], 1):
        print(f"\n{i}. {doc['problem_class']}")
        print(f"   Content preview: {doc['content'][:150]}...")
    
    # Create agent
    llm = ChatOpenAI(
        model=os.environ.get("MODEL_NAME", "gpt-oss-120b"),
        api_key=os.environ.get("SAKURA_API_KEY", "dummy"),
        base_url=os.environ.get("BASE_URL", "https://api.ai.sakura.ad.jp/v1/"),
        temperature=0
    )
    
    ace_app = build_ace_agent(llm, memory)
    
    # Test query related to stored content
    test_query = "bottles puzzle"  # This should match stored documents
    
    print(f"\n{'=' * 80}")
    print(f"TEST QUERY: {test_query}")
    print("=" * 80)
    
    initial_state = {
        "messages": [HumanMessage(content=test_query)],
        "retry_count": 0,
        "context_docs": [],
        "extracted_entities": [],
        "problem_class": ""
    }
    
    # Invoke agent
    print("\nInvoking agent...")
    final_state = ace_app.invoke(initial_state)
    
    # Check results
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print("=" * 80)
    
    print(f"\nExtracted Entities: {final_state.get('extracted_entities', [])}")
    print(f"Problem Class: {final_state.get('problem_class', '')}")
    
    context_docs = final_state.get('context_docs', [])
    print(f"\nRetrieved Context Documents: {len(context_docs)}")
    if context_docs:
        for i, doc in enumerate(context_docs, 1):
            print(f"\n  Context {i}:")
            print(f"  {doc[:200]}...")
    
    # Check if context was injected into messages
    print(f"\n{'=' * 80}")
    print("MESSAGE FLOW ANALYSIS")
    print("=" * 80)
    
    messages = final_state.get('messages', [])
    print(f"\nTotal messages in final state: {len(messages)}")
    
    for i, msg in enumerate(messages, 1):
        msg_type = type(msg).__name__
        content_preview = str(msg.content)[:100]
        print(f"\n{i}. {msg_type}")
        print(f"   Content: {content_preview}...")
        
        # Check if this is the context injection
        if "Retrieved Context" in str(msg.content):
            print("   ✓✓✓ CONTEXT INJECTION FOUND ✓✓✓")
    
    # Final verdict
    print(f"\n{'=' * 80}")
    print("VERDICT")
    print("=" * 80)
    
    has_context_msg = any("Retrieved Context" in str(msg.content) for msg in messages)
    context_retrieved = len(context_docs) > 0
    
    if has_context_msg and context_retrieved:
        print("\n✓ SUCCESS: Context was retrieved AND injected into messages")
        print("✓ The agent SHOULD have access to the context")
    elif context_retrieved and not has_context_msg:
        print("\n⚠️  WARNING: Context was retrieved but NOT found in messages")
        print("⚠️  This indicates a potential bug")
    elif not context_retrieved:
        print("\n⚠️  INFO: No relevant context was found for the query")
        print("   This may be normal if the query doesn't match stored content")
    
    print()

if __name__ == "__main__":
    test_context_usage()
