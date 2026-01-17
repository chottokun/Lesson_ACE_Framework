import threading
import time
import json
from typing import Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ace_rm import prompts
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue

class BackgroundWorker(threading.Thread):
    """Asynchronous worker that processes the task queue.

    Performs 'Unified Analysis & Synthesis' to extract knowledge from
    recent interactions and update the Long-Term Memory.
    """

    def __init__(self, llm: ChatOpenAI, memory: ACE_Memory, task_queue: TaskQueue, interval: float = 1.0):
        super().__init__(daemon=True)
        self.memory = memory
        self.task_queue = task_queue
        self.llm = llm
        self.interval = interval
        self.running = True

    def run(self):
        print("[BackgroundWorker] Started.", flush=True)
        while self.running:
            try:
                task = self.task_queue.fetch_pending_task()
                if task:
                    self.process_task(task)
                else:
                    time.sleep(self.interval)
            except Exception as e:
                print(f"[BackgroundWorker] Loop Error: {e}", flush=True)
                time.sleep(5.0)

    def stop(self):
        self.running = False

    def process_task(self, task: Dict[str, Any]):
        task_id = task['id']
        print(f"[BackgroundWorker] Processing Task {task_id}...", flush=True)
        self.task_queue.mark_task_processing(task_id)

        user_input = task['user_input']
        agent_output = task['agent_output']
        
        # 0. Quick Filter (Rule-based)
        # Skip trivial interactions like greetings if needed (Already handled by Curator fast-path somewhat, 
        # but good to have here too if direct insertion is tried)
        # For now, we proceed to Unified Analysis.

        # 1. Pre-search for existing context using raw input
        # We use the user input and part of agent output as query
        search_query = f"{user_input}\n{agent_output[:200]}"
        similar_docs = self.memory.find_similar_vectors(search_query, threshold=0.4) # Slightly loose threshold to find candidates
        
        existing_docs_str = "None"
        if similar_docs:
            docs_content = []
            for doc_id, _ in similar_docs[:3]: # Top 3 candidates
                doc = self.memory.get_document_by_id(doc_id)
                if doc:
                    docs_content.append(f"ID: {doc['id']}\nContent: {doc['content']}")
            if docs_content:
                existing_docs_str = "\n---\n".join(docs_content)

        # 2. Unified Analysis & Synthesis (Single LLM Call)
        # The prompts module already handles language selection based on ACE_LANG
        prompt_tmpl = prompts.UNIFIED_ANALYSIS_PROMPT

        prompt = prompt_tmpl.format(
            user_input=user_input, 
            agent_output=agent_output,
            existing_docs=existing_docs_str
        )
        
        try:
            res = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if "```json" in res:
                res = res.split("```json")[1].split("```")[0]
            elif "```" in res:
                res = res.split("```")[1].split("```")[0]
            
            data = json.loads(res)
            
            should_store = data.get('should_store', False)
            if should_store:
                action = data.get('action', 'NEW').upper()
                target_doc_id = data.get('target_doc_id')
                new_content = data.get('analysis', '')
                new_entities = data.get('entities', [])
                new_p_class = data.get('problem_class', '')

                if action == 'UPDATE' and target_doc_id is not None:
                    print(f"[BackgroundWorker] Updating Doc {target_doc_id}", flush=True)
                    self.memory.update_document(target_doc_id, new_content, new_entities, new_p_class)
                elif action == 'KEPT':
                    print("[BackgroundWorker] Knowledge kept (redundant).", flush=True)
                else: # NEW
                    print("[BackgroundWorker] Adding NEW Doc", flush=True)
                    self.memory.add(new_content, new_entities, new_p_class)
            else:
                 print("[BackgroundWorker] Ignored (should_store=False).", flush=True)
            
            self.task_queue.mark_task_complete(task_id)

        except Exception as e:
            print(f"[BackgroundWorker] Task {task_id} Failed: {e}", flush=True)
            self.task_queue.mark_task_failed(task_id, str(e))
