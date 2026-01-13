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
LTM_MODE = os.getenv("LTM_MODE", "shared").lower()  # "isolated" or "shared"
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
    
    new_memory_df = get_memory_df(session_agent["memory"])
    new_task_df = get_task_df(session_agent["memory"])

    # Build new history in Gradio 6.x format
    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response_text}
    ]

    # Get meaningful status about background processing and LTM updates
    all_tasks = session_agent["memory"].get_tasks()
    pending_count = len([t for t in all_tasks if t['status'] in ['pending', 'processing']])
    recent_done = [t for t in all_tasks if t['status'] == 'done'][:3]  # Last 3 completed
    recent_failed = [t for t in all_tasks if t['status'] == 'failed'][:1]  # Last failure
    
    # Build status message
    status_parts = []
    if pending_count > 0:
        status_parts.append(f"⏳ {pending_count} task(s) processing")
    else:
        status_parts.append("✓ Queue idle")
    
    if recent_done:
        status_parts.append(f"📊 {len(recent_done)} recently completed")
    
    if recent_failed:
        status_parts.append(f"⚠️ {len(recent_failed)} failed")
    
    reflector_status = " | ".join(status_parts)
    
    # Get LTM update count (compare documents before and after)
    current_doc_count = len(new_memory_df)
    ltm_status = f"📚 Total: {current_doc_count} documents"

    return (
        new_history, entities_str, context_str,
        ltm_status, reflector_status, new_memory_df, new_task_df
    )

def get_memory_df(memory_instance: ACE_Memory):
    data = memory_instance.get_all()
    if not data:
        return pd.DataFrame(columns=["id", "content", "entities", "problem_class", "timestamp"])
    return pd.DataFrame(data)

def get_task_df(memory_instance: ACE_Memory):
    data = memory_instance.get_tasks()
    if not data:
        return pd.DataFrame(columns=["id", "user_input", "status", "created_at", "updated_at", "error_msg"])
    return pd.DataFrame(data)

def reset_memory_handler(session_id: str):
    session_agent = get_session_agent(session_id)
    session_agent["memory"].clear()
    return get_memory_df(session_agent["memory"])

def apply_distance_threshold(session_id: str, threshold: float):
    """Apply new distance threshold to the memory instance"""
    session_agent = get_session_agent(session_id)
    session_agent["memory"].distance_threshold = threshold
    return f"Distance threshold set to {threshold:.1f}"

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
                ltm_status = gr.Textbox(label="LTM Status", interactive=False)
                reflector_status = gr.Textbox(label="Background Processing", interactive=False)
            with gr.Group():
                gr.Markdown("#### Search Settings")
                distance_slider = gr.Slider(
                    minimum=1.0, maximum=3.0, value=1.8, step=0.1,
                    label="Distance Threshold (lower = stricter)",
                    info="Adjust search relevance filtering"
                )
                gr.Markdown("*Lower values return only highly relevant results*")

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
        
        with gr.Column():
            gr.Markdown("### ⏳ Background Reflection Queue")
            task_table = gr.DataFrame(
                headers=["id", "user_input", "status", "created_at", "updated_at", "error_msg"],
                interactive=False, wrap=True
            )
            gr.Markdown("*Queue auto-refreshes every 5 seconds*")

    # Timer for auto-refresh
    timer = gr.Timer(5)

    def on_load(session_id_str: str):
        print(f"UI loaded for session: {session_id_str}", flush=True)
        session_agent = get_session_agent(session_id_str)
        return get_memory_df(session_agent["memory"]), get_task_df(session_agent["memory"])

    def refresh_ui_state(session_id_str: str):
        session_agent = get_session_agent(session_id_str)
        return get_memory_df(session_agent["memory"]), get_task_df(session_agent["memory"])

    submit_btn.click(
        process_chat,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, curator_intent, curator_context, ltm_status, reflector_status, memory_table, task_table]
    ).then(lambda: "", None, msg) # Clear msg AFTER update

    msg.submit(
        process_chat,
        inputs=[msg, chatbot, session_id],
        outputs=[chatbot, curator_intent, curator_context, ltm_status, reflector_status, memory_table, task_table]
    ).then(lambda: "", None, msg) # Clear msg AFTER update

    refresh_mem_btn.click(refresh_ui_state, inputs=[session_id], outputs=[memory_table, task_table])
    reset_mem_btn.click(reset_memory_handler, inputs=[session_id], outputs=[memory_table])
    
    # Distance threshold control
    threshold_status = gr.Textbox(visible=False)  # Hidden status indicator
    distance_slider.change(
        apply_distance_threshold,
        inputs=[session_id, distance_slider],
        outputs=[threshold_status]
    )
    
    # Auto-refresh wiring
    timer.tick(refresh_ui_state, inputs=[session_id], outputs=[memory_table, task_table])
    
    demo.load(on_load, inputs=[session_id], outputs=[memory_table, task_table])

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
    