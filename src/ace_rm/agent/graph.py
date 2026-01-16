import json
import os
from typing import List, TypedDict, Optional, Annotated, Tuple, Any
from typing_extensions import NotRequired
from tenacity import retry, stop_after_attempt, wait_exponential

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from ace_rm import prompts
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue

# --- Agent State ---
class AgentState(TypedDict):
    messages: List[BaseMessage]
    context_docs: List[str]
    extracted_entities: List[str]
    problem_class: str
    retry_count: int
    lesson_learned: NotRequired[str]
    should_store: NotRequired[bool]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
def call_llm_with_retry(llm, messages):
    return llm.invoke(messages)

def build_ace_agent(llm: ChatOpenAI, memory: ACE_Memory, task_queue: Optional[TaskQueue] = None, use_tools: bool = True):
    """
    Builds the ACE Agent graph.
    If task_queue is not provided, reflector will not enqueue tasks.
    """
    
    tools = []
    if use_tools:
        @tool
        def search_memory_tool(query: str):
            """Searches the agent's long-term memory for relevant information, facts, or past experiences."""
            docs = memory.search(query)
            if not docs:
                return "No relevant information found in memory."
            return "\n\n".join(docs)
        tools = [search_memory_tool]

    tool_node_instance = ToolNode(tools) if use_tools else None

    def tool_executor_node(state: AgentState):
        if not tool_node_instance: return {}
        result = tool_node_instance.invoke(state)
        return {"messages": state['messages'] + result['messages']}

    llm_with_tools = llm.bind_tools(tools) if use_tools else llm

    # Configuration for fast-path optimization
    curator_skip_simple = os.environ.get("ACE_CURATOR_SKIP_SIMPLE", "false").lower() == "true"
    simple_patterns = [
        r"^(はい|いいえ|うん|ううん|わかりました|了解|OK|ok|yes|no)\.?$",
        r"^(ありがとう|thanks|thank you|どうも|サンキュー)",
        r"^(こんにちは|こんばんは|おはよう|hello|hi|hey)\.?$",
    ]
    import re
    skip_pattern_compiled = [re.compile(p, re.IGNORECASE) for p in simple_patterns]

    def curator_node(state: AgentState):
        messages = state['messages']
        last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_user_msg:
            return {"context_docs": [], "extracted_entities": [], "problem_class": ""}

        user_input = last_user_msg.content.strip()
        
        # Fast-path: Skip LLM for simple/short queries
        if curator_skip_simple:
            is_simple = len(user_input) < 20 or any(p.match(user_input) for p in skip_pattern_compiled)
            
            if is_simple:
                # Direct vector search without LLM intent analysis
                docs = memory.search(user_input)
                context_msg = []
                if docs:
                    context_str = "\n".join(docs)
                    context_msg = [SystemMessage(content=prompts.RETRIEVED_CONTEXT_TEMPLATE.format(context_str=context_str))]
                
                return {
                    "context_docs": docs,
                    "extracted_entities": [],
                    "problem_class": "",
                    "messages": context_msg + messages if context_msg else messages
                }

        # Full path: LLM-based intent analysis
        history_txt = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages[-5:-1]])
        prompt = prompts.INTENT_ANALYSIS_PROMPT.format(
            user_input=user_input,
            history_txt=history_txt
        )
        
        try:
            res = call_llm_with_retry(llm, [HumanMessage(content=prompt)]).content.strip()
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            elif "```" in res: res = res.split("```")[1].split("```")[0]
            
            data = json.loads(res)
            entities = data.get("entities", [])
            p_class = data.get("problem_class", "")
            query = data.get("search_query", user_input)
            
            docs = memory.search(query)
            
            context_msg = []
            if docs:
                context_str = "\n".join(docs)
                context_msg = [SystemMessage(content=prompts.RETRIEVED_CONTEXT_TEMPLATE.format(context_str=context_str))]
            
            return {
                "context_docs": docs,
                "extracted_entities": entities,
                "problem_class": p_class,
                "messages": context_msg + messages if context_msg else messages
            }
        except Exception as e:
            print(f"[Curator] Error: {e}")
            return {"context_docs": [], "extracted_entities": [], "problem_class": ""}

    def agent_node(state: AgentState):
        messages = state['messages']
        try:
            response = call_llm_with_retry(llm_with_tools, messages)
            return {"messages": messages + [response]}
        except Exception as e:
            return {"messages": messages + [AIMessage(content=f"Error in Agent: {e}")]}

    def reflector_node(state: AgentState):
        if task_queue is None:
            return {"lesson_learned": "Reflector disabled: No task queue provided."}
            
        messages = state['messages']
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage) and m.content]
        
        if not human_msgs or not ai_msgs: return {}

        last_human = human_msgs[-1]
        last_ai = ai_msgs[-1]

        try:
            print(f"[Reflector] Enqueueing interaction...", flush=True)
            task_queue.enqueue_task(last_human.content, last_ai.content)
            return {"lesson_learned": "Analysis queued in background.", "should_store": True}
        except Exception as e:
            print(f"[Reflector] Error enqueueing: {e}", flush=True)
            return {"should_store": False, "lesson_learned": f"Error: {e}"}

    # --- Graph ---
    workflow = StateGraph(AgentState)
    workflow.add_node("curator", curator_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("reflector", reflector_node)

    if use_tools:
        workflow.add_node("tool_executor", tool_executor_node)

    workflow.set_entry_point("curator")
    workflow.add_edge("curator", "agent")
    
    if use_tools:
        def check_tool_call(state: AgentState):
            last_message = state['messages'][-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tool_executor"
            return "reflector"

        workflow.add_conditional_edges("agent", check_tool_call, {"tool_executor": "tool_executor", "reflector": "reflector"})
        workflow.add_edge("tool_executor", "agent")
    else:
        workflow.add_edge("agent", "reflector")

    workflow.add_edge("reflector", END)
    return workflow.compile()
