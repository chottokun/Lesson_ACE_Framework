import gradio as gr
import pandas as pd
import uuid
import os
from datetime import datetime
from typing import Dict, Any

from langchain_core.messages import HumanMessage, AIMessage
from ace_rm.ace_framework import (
    build_ace_agent, ACE_Memory, TaskQueue, BackgroundWorker
)
from ace_rm.config import (
    MODEL_NAME, BASE_URL, OPENAI_API_KEY, LLM_TEMPERATURE, LTM_MODE
)
from langchain_openai import ChatOpenAI

print(f"Running in LTM_MODE: {LTM_MODE}", flush=True)

# --- Global/Session-based Initialization ---
agent_sessions: Dict[str, Any] = {}
shared_agent = None

if LTM_MODE == "shared":
    print("Initializing shared agent...", flush=True)
    llm = ChatOpenAI(
        model=MODEL_NAME,
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL,
        temperature=LLM_TEMPERATURE
    )
    # No session_id provided for a shared memory
    shared_memory = ACE_Memory()
    shared_queue = TaskQueue()
    shared_ace_app = build_ace_agent(llm, shared_memory, shared_queue)

    # shared worker now gets explicit memory/queue instances
    shared_worker = BackgroundWorker(llm=llm, memory=shared_memory, task_queue=shared_queue)
    shared_worker.start()

    shared_agent = {
        "memory": shared_memory,
        "queue": shared_queue,
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
            api_key=OPENAI_API_KEY,
            base_url=BASE_URL,
            temperature=LLM_TEMPERATURE
        )
        memory_instance = ACE_Memory(session_id=session_id)
        queue_instance = TaskQueue(session_id=session_id)
        ace_app_instance = build_ace_agent(llm, memory_instance, queue_instance)
        worker_instance = BackgroundWorker(llm=llm, memory=memory_instance, task_queue=queue_instance)
        worker_instance.start()

        agent_sessions[session_id] = {
            "memory": memory_instance,
            "queue": queue_instance,
            "app": ace_app_instance,
            "worker": worker_instance
        }
    return agent_sessions[session_id]


def process_chat(user_message: str, history: list, session_id: str, response_style: str):
    """
    Executes the ACE Agent for the given session.
    """
    session_agent = get_session_agent(session_id)
    ace_app = session_agent["app"]

    if not user_message:
        # On empty input, just refresh the memory and task views
        memory_df = get_memory_df(session_agent["memory"])
        task_df = get_task_df(session_agent["queue"])
        return history, "", "", "Refreshing...", "Refreshing...", memory_df, task_df

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

    # Build STM (Short-Term Memory) object
    stm = {
        "current_time": datetime.now().isoformat(),
        "response_style": response_style,
        "turn_count": len([m for m in history if isinstance(m, dict) and m.get('role') == 'user']) + 1
    }

    initial_state = {
        "messages": messages, "retry_count": 0, "context_docs": [],
        "extracted_entities": [], "problem_class": "", "stm": stm
    }
    
    final_state = ace_app.invoke(initial_state)
    
    last_msg = final_state["messages"][-1]
    response_text = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    
    entities_str = f"Entities: {final_state.get('extracted_entities', [])}\nClass: {final_state.get('problem_class', '')}"
    context_list = final_state.get("context_docs", [])
    context_str = "\n---\n".join(context_list) if context_list else "No context retrieved."
    
    new_memory_df = get_memory_df(session_agent["memory"])
    new_task_df = get_task_df(session_agent["queue"])

    # Build new history in Gradio 6.x format
    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response_text}
    ]

    # Get meaningful status about background processing and LTM updates
    all_tasks = session_agent["queue"].get_tasks()
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
        ltm_status, reflector_status, memory_df if not user_message else new_memory_df, new_task_df
    )

def get_memory_df(memory_instance: ACE_Memory):
    data = memory_instance.get_all()
    if not data:
        return pd.DataFrame(columns=["id", "content", "entities", "problem_class", "timestamp"])
    return pd.DataFrame(data)

def get_task_df(queue_instance: TaskQueue):
    data = queue_instance.get_tasks()
    if not data:
        return pd.DataFrame(columns=["id", "user_input", "status", "created_at", "updated_at", "error_msg"])
    return pd.DataFrame(data)

def reset_memory_handler(session_id: str):
    session_agent = get_session_agent(session_id)
    session_agent["memory"].clear()
    session_agent["queue"].clear()
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
                # Determine slider range based on distance metric
                distance_metric = os.environ.get('ACE_DISTANCE_METRIC', 'l2').lower()
                is_cosine = (distance_metric == 'cosine')
                
                initial_threshold = float(os.environ.get('ACE_DISTANCE_THRESHOLD', '0.7' if is_cosine else '1.8'))
                slider_min = 0.0 if is_cosine else 1.0
                slider_max = 1.0 if is_cosine else 3.0
                slider_step = 0.05 if is_cosine else 0.1

                distance_slider = gr.Slider(
                    minimum=slider_min, maximum=slider_max, value=initial_threshold, step=slider_step,
                    label=f"Distance Threshold ({distance_metric})",
                    info="Lower = stricter (Cosine: higher similarity, L2: smaller distance)" if not is_cosine else "Higher = stricter (Similarity score)"
                )
                gr.Markdown("*Adjust relevance filtering criteria*")
            with gr.Group():
                gr.Markdown("#### 🎛️ 応答設定 (STM)")
                response_style = gr.Dropdown(
                    choices=[
                        ("簡潔", "concise"),
                        ("詳細", "detailed"),
                        ("根拠重視", "evidence-based"),
                        ("ステップバイステップ", "step-by-step"),
                        ("比較・対照", "comparative"),
                        ("チュートリアル", "tutorial"),
                        ("要約のみ", "summary-only"),
                    ],
                    value="detailed",
                    label="応答スタイル"
                )

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
        return get_memory_df(session_agent["memory"]), get_task_df(session_agent["queue"])

    def refresh_ui_state(session_id_str: str):
        session_agent = get_session_agent(session_id_str)
        return get_memory_df(session_agent["memory"]), get_task_df(session_agent["queue"])

    submit_btn.click(
        process_chat,
        inputs=[msg, chatbot, session_id, response_style],
        outputs=[chatbot, curator_intent, curator_context, ltm_status, reflector_status, memory_table, task_table]
    ).then(lambda: "", None, msg) # Clear msg AFTER update

    msg.submit(
        process_chat,
        inputs=[msg, chatbot, session_id, response_style],
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