from typing import List, Dict, Any
import json
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from .memory import ACE_Memory
from . import prompts
from .typing import AgentState
from . import utils

def build_ace_agent(llm: ChatOpenAI, memory: ACE_Memory, use_tools: bool = True):
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
        """Executes tools and appends output to history."""
        if not tool_node_instance:
            return {}
        result = tool_node_instance.invoke(state)
        return {"messages": state['messages'] + result['messages']}

    llm_with_tools = llm.bind_tools(tools) if use_tools else llm

    def curator_node(state: AgentState):
        """Analyzes intent and retrieves context."""
        messages = state['messages']
        last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_user_msg:
            return {"context_docs": [], "extracted_entities": [], "problem_class": ""}

        history_txt = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages[-5:-1]])
        prompt = prompts.INTENT_ANALYSIS_PROMPT.format(user_input=last_user_msg.content, history_txt=history_txt)

        try:
            res = utils.call_llm_with_retry(llm, [HumanMessage(content=prompt)]).content.strip()
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            elif "```" in res: res = res.split("```")[1].split("```")[0]

            data = json.loads(res)
            entities = data.get("entities", [])
            p_class = data.get("problem_class", "")
            query = data.get("search_query", last_user_msg.content)

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
        """Generates response or calls tools."""
        messages = state['messages']
        try:
            response = utils.call_llm_with_retry(llm_with_tools, messages)
            return {"messages": messages + [response]}
        except Exception as e:
            return {"messages": messages + [AIMessage(content=f"Error in Agent: {e}")]}

    def reflector_node(state: AgentState):
        """Queues the interaction for background analysis."""
        messages = state['messages']
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage) and m.content]

        if not human_msgs or not ai_msgs:
            return {}

        last_human = human_msgs[-1]
        last_ai = ai_msgs[-1]

        try:
            print(f"[Reflector] Enqueueing interaction for background processing...", flush=True)
            memory.enqueue_task(last_human.content, last_ai.content)
            return {"lesson_learned": "Analysis queued in background.", "should_store": True}
        except Exception as e:
            print(f"[Reflector] Error enqueueing task: {e}", flush=True)
            return {"should_store": False, "lesson_learned": "Error enqueueing reflection."}

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
            messages = state['messages']
            last_message = messages[-1]
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tool_executor"
            return "reflector"

        workflow.add_conditional_edges("agent", check_tool_call, {"tool_executor": "tool_executor", "reflector": "reflector"})
        workflow.add_edge("tool_executor", "agent")
    else:
        workflow.add_edge("agent", "reflector")

    workflow.add_edge("reflector", END)

    return workflow.compile()
