import threading
import time
import json
import os
from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from ace_rm import prompts
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue

class BackgroundWorker(threading.Thread):
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
        # Determine language for prompt
        is_jp = os.environ.get("ACE_LANG") == "ja"
        prompt_tmpl = prompts.UNIFIED_ANALYSIS_PROMPT if is_jp else prompts.UNIFIED_ANALYSIS_PROMPT_EN  # Assuming en.py has UNIFIED.. too, or fallback
        # Fallback if UNIFIED_ANALYSIS_PROMPT_EN is not available in prompts module (it should be)
        if not hasattr(prompts, 'UNIFIED_ANALYSIS_PROMPT_EN'):
             # If strictly separated by files, we might need import adjustment. 
             # For now, we assume prompts.__init__ handles or we use the textual prompts directly.
             pass

        prompt = prompt_tmpl.format(
            user_input=user_input, 
            agent_output=agent_output,
            existing_docs=existing_docs_str
        )
        
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            res = response.content.strip() if response.content else ""
            
            # Handle empty LLM response
            if not res:
                print(f"[BackgroundWorker] Task {task_id}: Empty LLM response, skipping.", flush=True)
                self.task_queue.mark_task_complete(task_id)
                return
            
            # Extract JSON from markdown code blocks
            if "```json" in res: 
                res = res.split("```json")[1].split("```")[0].strip()
            elif "```" in res: 
                res = res.split("```")[1].split("```")[0].strip()
            
            # Handle empty JSON content after extraction
            if not res:
                print(f"[BackgroundWorker] Task {task_id}: No JSON content found, skipping.", flush=True)
                self.task_queue.mark_task_complete(task_id)
                return
            
            try:
                data = json.loads(res)
            except json.JSONDecodeError as je:
                print(f"[BackgroundWorker] Task {task_id}: JSON parse error: {je}. Response was: {res[:100]}...", flush=True)
                self.task_queue.mark_task_complete(task_id)  # Complete instead of fail to avoid retry loops
                return
            
            should_store = data.get('should_store', False)
            if should_store:
                action = data.get('action', 'NEW').upper()
                target_doc_id = data.get('target_doc_id')
                raw_content = data.get('analysis', '')
                new_entities = data.get('entities', [])
                new_p_class = data.get('problem_class', '')

                # Apply LTM_KNOWLEDGE_MODEL_PROMPT to structure the content
                # This transforms raw analysis into a reusable domain knowledge model
                structured_content = self._structure_as_knowledge_model(
                    user_input, agent_output, raw_content
                )
                
                if action == 'UPDATE' and target_doc_id is not None:
                    print(f"[BackgroundWorker] Updating Doc {target_doc_id}", flush=True)
                    self.memory.update_document(target_doc_id, structured_content, new_entities, new_p_class)
                elif action == 'KEPT':
                    print("[BackgroundWorker] Knowledge kept (redundant).", flush=True)
                else: # NEW
                    print("[BackgroundWorker] Adding NEW Doc", flush=True)
                    self.memory.add(structured_content, new_entities, new_p_class)
            else:
                 print("[BackgroundWorker] Ignored (should_store=False).", flush=True)
            
            self.task_queue.mark_task_complete(task_id)

        except Exception as e:
            print(f"[BackgroundWorker] Task {task_id} Failed: {e}", flush=True)
            self.task_queue.mark_task_failed(task_id, str(e))

    def _structure_as_knowledge_model(self, user_input: str, agent_output: str, raw_analysis: str) -> str:
        """
        Apply LTM_KNOWLEDGE_MODEL_PROMPT to transform raw content into a structured domain knowledge model.
        Returns the Markdown-formatted content directly from LLM.
        """
        try:
            # Prepare context for the prompt
            context = f"User Input: {user_input}\n\nAgent Response: {agent_output}\n\nAnalysis:\n{raw_analysis}"
            
            prompt = prompts.LTM_KNOWLEDGE_MODEL_PROMPT.format(context=context)
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            result = response.content.strip() if response.content else ""
            
            if not result:
                print("[BackgroundWorker] LTM structuring: Empty LLM response, using raw analysis.", flush=True)
                return raw_analysis
            
            print("[BackgroundWorker] LTM structuring: Successfully generated knowledge model.", flush=True)
            return result
            
        except Exception as e:
            print(f"[BackgroundWorker] LTM structuring failed: {e}. Falling back to raw analysis.", flush=True)
            return raw_analysis
