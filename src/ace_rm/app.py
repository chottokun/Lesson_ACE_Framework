import gradio as gr
import pandas as pd
from dotenv import load_dotenv
import uuid
import os
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from ace_rm.ace_framework import build_ace_agent, ACE_Memory, MODEL_NAME, BASE_URL, BackgroundWorker
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# --- Mode Configuration ---
LTM_MODE = os.getenv("LTM_MODE", "isolated").lower()  # "isolated" or "shared"
print(f"Running in LTM_MODE: {LTM_MODE}", flush=True)

# --- Global/Session-based Initialization ---
agent_sessions: Dict[str, Any] = {}
shared_agent = None

if LTM_MODE == "shared":
    print("Initializing shared agent...", flush=True)
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.environ.get("SAKURA_API_KEY", "dummy"),
        base_url=BASE_URL,
        temperature=0
    )
    # No session_id provided for a shared memory
    shared_memory = ACE_Memory()
    shared_ace_app = build_ace_agent(llm, shared_memory)

    # One worker for the shared memory
    shared_worker = BackgroundWorker(llm=llm)
    shared_worker.start()

    shared_agent = {
        "memory": shared_memory,
        "app": shared_ace_app,
        "worker": shared_worker
    }

def get_session_agent(session_id: str):
    """
    Creates or retrieves an ACE agent instance based on the LTM_MODE.
    """
    if LTM_MODE == "shared":
        return shared_agent

    # Isolated mode
    if session_id not in agent_sessions:
        print(f"Creating new agent for session: {session_id}", flush=True)
        llm = ChatOpenAI(
            model=MODEL_NAME,
            api_key=os.environ.get("SAKURA_API_KEY", "dummy"),
            base_url=BASE_URL,
            temperature=0
        )
        memory_instance = ACE_Memory(session_id=session_id)
        ace_app_instance = build_ace_agent(llm, memory_instance)
        worker_instance = BackgroundWorker(llm=llm, memory_session_id=session_id)
        worker_instance.start()

        agent_sessions[session_id] = {
            "memory": memory_instance,
            "app": ace_app_instance,
            "worker": worker_instance
        }
    return agent_sessions[session_id]


def process_chat(user_message: str, history: list, session_id: str):
    """
    Executes the ACE Agent for the given session.
    """
    session_agent = get_session_agent(session_id)
    ace_app = session_agent["app"]

    if not user_message:
        # On empty input, just refresh the memory view
        memory_df = get_memory_df(session_agent["memory"])
        return history, "", "", "", "", memory_df

    # Gradio 6.x history is often a list of dictionaries [{'role': 'user', 'content': '...'}, ...]
    # or the previous list of tuples format. We handle both to be safe.
    messages = []
    for m in history:
        if isinstance(m, dict):
            role = m.get("role")
            content = m.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
        elif isinstance(m, (list, tuple)) and len(m) == 2:
            messages.append(HumanMessage(content=m[0]))
            messages.append(AIMessage(content=m[1]))

    messages.append(HumanMessage(content=user_message))

    initial_state = {
        "messages": messages, "retry_count": 0, "context_docs": [],
        "extracted_entities": [], "problem_class": ""
    }
    
    final_state = ace_app.invoke(initial_state)
    
    last_msg = final_state["messages"][-1]
    response_text = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    
    entities_str = f"Entities: {final_state.get('extracted_entities', [])}\nClass: {final_state.get('problem_class', '')}"
    context_list = final_state.get("context_docs", [])
    context_str = "\n---\n".join(context_list) if context_list else "No context retrieved."
    lesson = final_state.get("lesson_learned", "N/A")
    store_status = str(final_state.get("should_store", False))
    
    new_memory_df = get_memory_df(session_agent["memory"])

    # Build new history in Gradio 6.x format
    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response_text}
    ]

    return (
        new_history, entities_str, context_str,
        lesson, store_status, new_memory_df
    )

def get_memory_df(memory_instance: ACE_Memory):
    data = memory_instance.get_all()
    if not data:
        return pd.DataFrame(columns=["id", "content", "entities", "problem_class", "timestamp"])
    return pd.DataFrame(data)

def reset_memory_handler(session_id: str):
    session_agent = get_session_agent(session_id)
    session_agent["memory"].clear()
    return get_memory_df(session_agent["memory"])

# --- UI Layout ---
with gr.Blocks(title="ACE Agent Framework") as demo:
    # Use a fixed session ID for shared mode, or generate a new one for isolated mode
    session_id_val = "shared_session" if LTM_MODE == "shared" else str(uuid.uuid4())
    session_id = gr.State(value=session_id_val)

    gr.Markdown("# 🤖 ACE (Agentic Context Engineering) Framework Demo")
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=600, label="Conversation")
            with gr.Row():
                msg = gr.Textbox(scale=4, placeholder="Type a message...", show_label=False)
                submit_btn = gr.Button("Send", scale=1, variant="primary")
            clear_btn = gr.Button("Clear Chat")
            clear_btn.click(lambda: ([], ""), None, [chatbot, msg])

        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Internals (Debug)")
            with gr.Group():
                curator_intent = gr.Textbox(label="Curator: Intent & Entities", lines=4, interactive=False)
                curator_context = gr.Textbox(label="Curator: Retrieved Context", lines=10, interactive=False)
                reflector_lesson = gr.Textbox(label="Reflector: Learned Lesson", lines=6, interactive=False)
                reflector_status = gr.Textbox(label="Reflector: Stored?", interactive=False)

    with gr.Row():
        with gr.Column():
            ltm_title = "### 🧠 Long-Term Memory (Shared)" if LTM_MODE == 'shared' else "### 🧠 Long-Term Memory (Session-Specific)"
            gr.Markdown(ltm_title)
            memory_table = gr.DataFrame(
                headers=["id", "content", "entities", "problem_class", "timestamp"],
                interactive=False, wrap=True
            )
            refresh_mem_btn = gr.Button("Refresh Memory")
            reset_mem_btn = gr.Button("⚠️ Reset All Memory", variant="stop")

    def on_load(session_id_str: str):
        print(f"UI loaded for session: {session_id_str}", flush=True)
        session_agent = get_session_agent(session_id_str)
        return get_memory_df(session_agent["memory"])

    submit_btn.click(
        process_chat,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, curator_intent, curator_context, reflector_lesson, reflector_status, memory_table]
    ).then(lambda: "", None, msg) # Clear msg AFTER update

    msg.submit(
        process_chat,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, curator_intent, curator_context, reflector_lesson, reflector_status, memory_table]
    ).then(lambda: "", None, msg) # Clear msg AFTER update
    
    def refresh_memory_display(session_id_str: str):
        session_agent = get_session_agent(session_id_str)
        return get_memory_df(session_agent["memory"])

    refresh_mem_btn.click(refresh_memory_display, inputs=[session_id], outputs=[memory_table])
    reset_mem_btn.click(reset_memory_handler, inputs=[session_id], outputs=[memory_table])
    
    demo.load(on_load, inputs=[session_id], outputs=[memory_table])

if __name__ == "__main__":
    print("Starting Gradio app...", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
    print("Gradio app finished.", flush=True)
    
    # Optional: Cleanup workers on exit (though daemon=True handles this)
    if LTM_MODE == "isolated":
        for sid, agent_data in agent_sessions.items():
            print(f"Stopping worker for session {sid}...", flush=True)
            agent_data["worker"].stop()
            agent_data["worker"].join()
    elif shared_agent:
        print("Stopping shared worker...", flush=True)
        shared_agent["worker"].stop()
        shared_agent["worker"].join()
    