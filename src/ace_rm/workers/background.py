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

        lang_instruction = "出力は必ず日本語（Japanese）で行ってください。" if os.environ.get("ACE_LANG") == "ja" else "Ensure the output is in English."
        prompt = prompts.ANALYSIS_PROMPT.format(user_input=user_input, agent_output=agent_output)
        if "Output JSON only:" in prompt:
            prompt = prompt.replace("Output JSON only:", f"{lang_instruction}\nOutput JSON only:")
        
        try:
            # We assume call_llm_with_retry is available or we can use llm.invoke
            res = self.llm.invoke([HumanMessage(content=prompt)]).content.strip()
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            elif "```" in res: res = res.split("```")[1].split("```")[0]
            
            data = json.loads(res)
            
            if data.get('should_store', False):
                new_content = data.get('analysis', '')
                new_entities = data.get('entities', [])
                new_p_class = data.get('problem_class', '')

                similar_docs = self.memory.find_similar_vectors(new_content, threshold=0.5)
                
                action = "NEW"
                target_doc_id = None
                final_content = new_content
                final_entities = new_entities
                final_p_class = new_p_class

                if similar_docs:
                    best_match_id, best_dist = similar_docs[0]
                    existing_doc = self.memory.get_document_by_id(best_match_id)
                    
                    if existing_doc:
                        existing_content = existing_doc['content']
                        synthesizer_prompt = prompts.SYNTHESIZER_PROMPT.format(
                            best_match_id=best_match_id,
                            existing_content=existing_content,
                            new_content=new_content
                        )
                        if "Output JSON only:" in synthesizer_prompt:
                            synthesizer_prompt = synthesizer_prompt.replace("Output JSON only:", f"{lang_instruction}\nOutput JSON only:")
                        
                        try:
                            syn_res = self.llm.invoke([HumanMessage(content=synthesizer_prompt)]).content.strip()
                            if "```json" in syn_res: syn_res = syn_res.split("```json")[1].split("```")[0]
                            elif "```" in syn_res: syn_res = syn_res.split("```")[1].split("```")[0]
                            
                            syn_data = json.loads(syn_res)
                            action = syn_data.get('action', 'NEW').upper()
                            if action == 'UPDATE':
                                target_doc_id = best_match_id
                                final_content = syn_data.get('synthesized_content', new_content)
                                llm_entities = syn_data.get('merged_entities')
                                if llm_entities:
                                    final_entities = llm_entities
                                else:
                                    exist_entities = json.loads(existing_doc['entities'])
                                    final_entities = list(set(exist_entities + new_entities))
                        except Exception as e:
                            print(f"[BackgroundWorker] Synthesizer Error: {e}. Defaulting to NEW.")
                
                if action == "UPDATE" and target_doc_id is not None:
                    self.memory.update_document(target_doc_id, final_content, final_entities, final_p_class)
                elif action == "KEPT":
                    pass
                else:
                    self.memory.add(final_content, final_entities, final_p_class)
            
            self.task_queue.mark_task_complete(task_id)

        except Exception as e:
            print(f"[BackgroundWorker] Task {task_id} Failed: {e}")
            self.task_queue.mark_task_failed(task_id, str(e))
