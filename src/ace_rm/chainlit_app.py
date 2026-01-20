import os
import uuid
import json
import chainlit as cl
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from ace_rm.ace_framework import (
    build_ace_agent, ACE_Memory, TaskQueue
)
from ace_rm.config import (
    MODEL_NAME, BASE_URL, OPENAI_API_KEY, LLM_TEMPERATURE, LTM_MODE
)

# --- Configuration ---
# You can adjust these or use environment variables
RESPONSE_STYLE_DEFAULT = "detailed"

@cl.on_chat_start
async def start():
    # Use a fixed session ID for shared mode, or generate a new one for isolated mode
    session_id = "shared_session" if LTM_MODE == "shared" else str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)

    # Initialize LLM
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
        temperature=LLM_TEMPERATURE,
        streaming=True
    )

    # Initialize Memory and Queue
    memory = ACE_Memory(session_id=session_id if LTM_MODE != "shared" else None)
    queue = TaskQueue(session_id=session_id if LTM_MODE != "shared" else None)
    
    # Build Agent
    agent = build_ace_agent(llm, memory, queue)
    
    cl.user_session.set("agent", agent)
    cl.user_session.set("memory", memory)
    cl.user_session.set("queue", queue)
    cl.user_session.set("history", [])
    cl.user_session.set("stm_model", {"constraints": [], "actions": [], "entities": []})

    # Welcome Message with Actions
    actions = [
        cl.Action(name="reset_memory", payload={"action": "reset"}, label="Reset Memory (⚠️)"),
        cl.Action(name="view_stm", payload={"action": "view"}, label="View World Model")
    ]
    await cl.Message(
        content="🤖 **ACE Framework** へようこそ！\n長期記憶と継続学習を備えたエージェントです。何をお手伝いしましょうか？",
        actions=actions
    ).send()

@cl.action_callback("reset_memory")
async def on_reset(action):
    memory = cl.user_session.get("memory")
    queue = cl.user_session.get("queue")
    memory.clear()
    queue.clear()
    cl.user_session.set("history", [])
    cl.user_session.set("stm_model", {"constraints": [], "actions": [], "entities": []})
    await cl.Message(content="✅ メモリをリセットしました。").send()

@cl.action_callback("view_stm")
async def on_view_stm(action):
    stm_model = cl.user_session.get("stm_model")
    model_json = json.dumps(stm_model, indent=2, ensure_ascii=False)
    await cl.Message(content=f"🧠 **現在の世界モデル (STM)**:\n```json\n{model_json}\n```").send()

@cl.on_message
async def main(message: cl.Message):
    agent = cl.user_session.get("agent")
    memory = cl.user_session.get("memory")
    history = cl.user_session.get("history")
    stm_model = cl.user_session.get("stm_model")

    # Add user message to history
    history.append(HumanMessage(content=message.content))

    # Prepare STM
    stm = {
        "current_time": datetime.now().isoformat(),
        "response_style": RESPONSE_STYLE_DEFAULT,
        "turn_count": len([m for m in history if isinstance(m, HumanMessage)]),
        "model": stm_model
    }

    initial_state = {
        "messages": history,
        "retry_count": 0,
        "context_docs": [],
        "extracted_entities": [],
        "problem_class": "",
        "stm": stm
    }

    final_response = cl.Message(content="")
    
    async for chunk in agent.astream(initial_state, stream_mode="updates"):
        for node_name, output in chunk.items():
            if node_name == "curator":
                async with cl.Step(name="Thinking: Intent Analysis", type="run") as step:
                    entities = output.get("extracted_entities", [])
                    p_class = output.get("problem_class", "")
                    
                    if "stm" in output and "model" in output["stm"]:
                        new_model = output["stm"]["model"]
                        cl.user_session.set("stm_model", new_model)
                        model_json = json.dumps(new_model, indent=2, ensure_ascii=False)
                        step.output = f"**Entities**: {entities}\n**Class**: {p_class}\n\n**World Model Update**:\n```json\n{model_json}\n```"
                    else:
                        step.output = f"**Entities**: {entities}\n**Class**: {p_class}"

            elif node_name == "agent":
                messages = output.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, AIMessage):
                        if last_msg.tool_calls:
                            # If it's a tool call, we don't stream it to the main message
                            pass
                        elif last_msg.content:
                            # Stream final response tokens
                            # Note: In a loop, there might be multiple AIMessages. 
                            # Usually we only stream the one with content.
                            await final_response.stream_token(last_msg.content)
            
            elif node_name == "tool_executor":
                async with cl.Step(name="Memory Access", type="run") as step:
                    messages = output.get("messages", [])
                    if messages and getattr(messages[-1], "type", "") == "tool":
                        step.output = messages[-1].content
                    else:
                        step.output = "Searching long-term memory..."

            elif node_name == "reflector":
                async with cl.Step(name="Background Processing", type="run") as step:
                    lesson = output.get("lesson_learned", "Queuing for reflection...")
                    step.output = lesson

    if not final_response.content:
        # Fallback if no content was streamed (e.g. error)
        final_response.content = "申し訳ありません。応答の生成中に問題が発生しました。"

    await final_response.send()
    
    # Update history
    history.append(AIMessage(content=final_response.content))
    cl.user_session.set("history", history)

    # Periodic LTM Status check
    count = len(memory.get_all())
    # Instead of a full message, we can use a status message or similar if needed.
    # For now, let's just keep it simple.

