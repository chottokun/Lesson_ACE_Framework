import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from ace_rm.ace_framework import build_ace_agent, ACE_Memory, AgentState, MODEL_NAME, BASE_URL, BackgroundWorker
from langchain_openai import ChatOpenAI
import os

# Load environment variables
load_dotenv()

# --- Initialization ---
memory = ACE_Memory()
llm = ChatOpenAI(
    model=MODEL_NAME, 
    api_key=os.environ.get("SAKURA_API_KEY", "dummy"), 
    base_url=BASE_URL,
    temperature=0
)
ace_app = build_ace_agent(llm, memory)

# Start Background Worker
worker = BackgroundWorker(memory, llm)
worker.start()

def process_chat(user_message, history):
    """
    Executes the ACE Agent and returns the response + debug info.
    """
    if not user_message:
        return "", history, "", "", "", "", get_memory_df()

    # 1. Convert history to LangChain messages
    messages = []
    for h in history:
        # history is list of dicts: {'role': 'user'/'assistant', 'content': '...'}
        role = h.get('role')
        content = h.get('content')
        if role == 'user':
            messages.append(HumanMessage(content=content))
        elif role == 'assistant':
            messages.append(AIMessage(content=content))
    
    messages.append(HumanMessage(content=user_message))

    # 2. Invoke Graph
    initial_state = {
        "messages": messages, 
        "retry_count": 0,
        "context_docs": [],
        "extracted_entities": [],
        "problem_class": ""
    }
    
    # We use invoke (synchronous)
    final_state = ace_app.invoke(initial_state)
    
    # 3. Extract Outputs
    final_messages = final_state["messages"]
    last_msg = final_messages[-1]
    response_text = last_msg.content if isinstance(last_msg, AIMessage) else str(last_msg)
    
    # Debug Info
    entities_str = f"Entities: {final_state.get('extracted_entities', [])}\nClass: {final_state.get('problem_class', '')}"
    
    context_list = final_state.get("context_docs", [])
    context_str = "\n---\n".join(context_list) if context_list else "No context retrieved."
    
    lesson = final_state.get("lesson_learned", "N/A")
    store_status = str(final_state.get("should_store", False))

    # Update History
    new_history = history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response_text}
    ]
    
    # Memory Table Update
    new_memory_df = get_memory_df()

    return (
        "", # Clear input
        new_history, # Update Chatbot
        entities_str, # Curator Intent
        context_str, # Curator Context
        lesson, # Reflector Lesson
        store_status, # Reflector Status
        new_memory_df # Memory Table
    )

def get_memory_df():
    data = memory.get_all()
    if not data:
        return pd.DataFrame(columns=["id", "content", "entities", "problem_class", "timestamp"])
    return pd.DataFrame(data)

def reset_memory_handler():
    memory.clear()
    return get_memory_df()

# --- UI Layout ---
with gr.Blocks(title="ACE Agent Framework") as demo:
    gr.Markdown("# 🤖 ACE (Agentic Context Engineering) Framework Demo")
    
    with gr.Row():
        # --- Left: Chat (75%) ---
        with gr.Column(scale=3):
            # IMPORTANT: type="messages" is key for the updated format
            chatbot = gr.Chatbot(height=600, label="Conversation")
            with gr.Row():
                msg = gr.Textbox(scale=4, placeholder="Type a message...", label="User Input")
                submit_btn = gr.Button("Send", scale=1, variant="primary")
            
            clear_btn = gr.ClearButton([msg, chatbot])

        # --- Right: Debug (25%) ---
        with gr.Column(scale=1):
            gr.Markdown("### 🛠️ Internals (Debug)")
            with gr.Group():
                curator_intent = gr.Textbox(label="Curator: Intent & Entities", lines=4, interactive=False)
                curator_context = gr.Textbox(label="Curator: Retrieved Context", lines=10, interactive=False)
                reflector_lesson = gr.Textbox(label="Reflector: Learned Lesson", lines=6, interactive=False)
                reflector_status = gr.Textbox(label="Reflector: Stored?", interactive=False)

    # --- Bottom: Memory ---
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🧠 Long-Term Memory")
            memory_table = gr.DataFrame(
                value=get_memory_df, 
                headers=["id", "content", "entities", "problem_class", "timestamp"],
                label="Memory Store",
                interactive=False,
                wrap=True
            )
            refresh_mem_btn = gr.Button("Refresh Memory")
            reset_mem_btn = gr.Button("⚠️ Reset All Memory", variant="stop")

    # --- Event Wiring ---
    
    # Submit Action
    inputs = [msg, chatbot]
    outputs = [msg, chatbot, curator_intent, curator_context, reflector_lesson, reflector_status, memory_table]
    
    msg.submit(process_chat, inputs, outputs)
    submit_btn.click(process_chat, inputs, outputs)
    
    # Memory Actions
    refresh_mem_btn.click(get_memory_df, None, memory_table)
    reset_mem_btn.click(reset_memory_handler, None, memory_table)

if __name__ == "__main__":
    print("Starting Gradio app...", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=7860)
    print("Gradio app finished.", flush=True)
    
    