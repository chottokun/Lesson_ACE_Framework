import os
import sqlite3
import json
import numpy as np
import faiss
import time
import threading
from typing import List, TypedDict, Optional, Annotated, Sequence, Union, Dict, Any
from typing_extensions import NotRequired
from tenacity import retry, stop_after_attempt, wait_exponential
from sentence_transformers import SentenceTransformer
from filelock import FileLock
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool, StructuredTool
from langgraph.prebuilt import ToolNode

# --- Configuration & Setup ---
DB_PATH = "ace_memory.db"
FAISS_INDEX_PATH = "ace_memory.faiss"

# API Setup
SAKURA_MODEL = "gpt-oss-120b"
SAKURA_BASE_URL = "https://api.ai.sakura.ad.jp/v1/"

try:
    OPENAI_API_KEY = os.environ.get('SAKURA_API_KEY')
    if not OPENAI_API_KEY:
        # Placeholder or fallback
        OPENAI_API_KEY = "dummy_key" 
except Exception:
    OPENAI_API_KEY = "dummy_key"

MODEL_NAME = SAKURA_MODEL
BASE_URL = SAKURA_BASE_URL

# --- Agent State ---
class AgentState(TypedDict):
    # Conversation history
    messages: List[BaseMessage]
    # Context injected by Curator
    context_docs: List[str]
    # Extracted entities by Curator
    extracted_entities: List[str]
    # Problem class identified by Curator
    problem_class: str
    # Retry counter
    retry_count: int
    # Reflector output summary
    lesson_learned: NotRequired[str]
    # Reflector storage status
    should_store: NotRequired[bool]

# --- ACE Memory (Persistent Layer) ---
class ACE_Memory:
    def __init__(self, session_id: Optional[str] = None):
        self.session_id = session_id

        if self.session_id:
            # Session-specific paths in a dedicated folder
            data_dir = "user_data"
            os.makedirs(data_dir, exist_ok=True)
            self.db_path = os.path.join(data_dir, f"ace_memory_{self.session_id}.db")
            self.index_path = os.path.join(data_dir, f"ace_memory_{self.session_id}.faiss")
            self.index_lock_path = os.path.join(data_dir, f"ace_memory_{self.session_id}.faiss.lock")
        else:
            # Global paths in the project root
            self.db_path = DB_PATH
            self.index_path = FAISS_INDEX_PATH
            self.index_lock_path = f"{FAISS_INDEX_PATH}.lock"

        # Initialize embedding model (cpu-friendly)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384

        self._init_db()
        self._load_or_build_index()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            # Enable WAL mode for concurrency
            conn.execute("PRAGMA journal_mode=WAL;")
            
            # Main table with metadata
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT,
                    entities TEXT,
                    problem_class TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Full Text Search table
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(content, entities, problem_class, content='documents', content_rowid='id')")
            # Triggers to keep FTS in sync
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents BEGIN INSERT INTO documents_fts(rowid, content, entities, problem_class) VALUES (new.id, new.content, new.entities, new.problem_class); END;")
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents BEGIN INSERT INTO documents_fts(documents_fts, rowid, content, entities, problem_class) VALUES('delete', old.id, old.content, old.entities, old.problem_class); END;")
            conn.execute("CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents BEGIN INSERT INTO documents_fts(documents_fts, rowid, content, entities, problem_class) VALUES('delete', old.id, old.content, old.entities, old.problem_class); INSERT INTO documents_fts(rowid, content, entities, problem_class) VALUES (new.id, new.content, new.entities, new.problem_class); END;")

            # Persistent Task Queue for Background Worker
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT,
                    agent_output TEXT,
                    status TEXT DEFAULT 'pending', -- pending, processing, done, failed
                    retries INTEGER DEFAULT 0,
                    error_msg TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def _load_or_build_index(self):
        with FileLock(self.index_lock_path):
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                    self._rebuild_vectors_from_db()
            else:
                self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))
                self._rebuild_vectors_from_db()
                faiss.write_index(self.index, self.index_path)

    def _rebuild_vectors_from_db(self):
        # This function should be called within a lock
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, content FROM documents")
            rows = cursor.fetchall()
            if rows:
                embeddings = self.encoder.encode([r[1] for r in rows])
                ids = np.array([r[0] for r in rows])
                self.index.add_with_ids(np.array(embeddings).astype('float32'), ids)

    def add(self, content: str, entities: List[str] = [], problem_class: str = ""):
        entities_json = json.dumps(entities)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO documents (content, entities, problem_class) VALUES (?, ?, ?)",
                (content, entities_json, problem_class)
            )
            doc_id = cursor.lastrowid

        vector = self.encoder.encode([content])
        with FileLock(self.index_lock_path):
            # Read the latest index from disk before modifying
            if os.path.exists(self.index_path):
                try:
                    self.index = faiss.read_index(self.index_path)
                except Exception:
                    # If index is corrupt, start fresh
                    self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dimension))

            self.index.add_with_ids(np.array(vector).astype('float32'), np.array([doc_id]))
            faiss.write_index(self.index, self.index_path)

    def search(self, query: str, k: int = 3) -> List[str]:
        results = {}
        # 1. Vector Search
        if self.index.ntotal > 0:
            query_vec = self.encoder.encode([query])
            D, I = self.index.search(np.array(query_vec).astype('float32'), k)
            found_ids = [int(idx) for idx in I[0] if idx >= 0]
            if found_ids:
                placeholders = ','.join('?' * len(found_ids))
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT content FROM documents WHERE id IN ({placeholders})", found_ids)
                    for row in cursor.fetchall():
                        results[row[0]] = row[0]

        # 2. Keyword Search (FTS5)
        with sqlite3.connect(self.db_path) as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT content FROM documents_fts WHERE documents_fts MATCH ? ORDER BY rank LIMIT ?", (query, k))
                for row in cursor.fetchall():
                    if row[0] not in results:
                        results[row[0]] = row[0]
            except Exception:
                pass # FTS syntax error or other issues
        return list(results.values())
    
    def clear(self):
        """Clears all memory (DB and Index) for the current session or globally."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.index_lock_path):
            try:
                os.remove(self.index_lock_path)
            except OSError:
                pass # File might be locked by another process
        self.__init__(session_id=self.session_id)

    def get_all(self) -> List[Dict[str, Any]]:
        """Retrieves all documents from the DB."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT id, content, entities, problem_class, timestamp FROM documents ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]

    # --- Queue Operations ---
    def enqueue_task(self, user_input: str, agent_output: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO task_queue (user_input, agent_output) VALUES (?, ?)",
                (user_input, agent_output)
            )

    def fetch_pending_task(self) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            # Fetch one pending task (FIFO)
            cursor.execute("SELECT * FROM task_queue WHERE status = 'pending' ORDER BY id ASC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return dict(row)
        return None

    def mark_task_processing(self, task_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = 'processing', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (task_id,))

    def mark_task_complete(self, task_id: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = 'done', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (task_id,))
    
    def mark_task_failed(self, task_id: int, error_msg: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE task_queue SET status = 'failed', error_msg = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (error_msg, task_id))

# --- Background Worker ---
class BackgroundWorker(threading.Thread):
    def __init__(self, llm: ChatOpenAI, memory_session_id: Optional[str] = None, interval: float = 1.0):
        super().__init__(daemon=True) # Daemon thread exits when main program exits
        self.memory = ACE_Memory(session_id=memory_session_id)
        self.llm = llm
        self.interval = interval
        self.running = True

    def run(self):
        print("[BackgroundWorker] Started.", flush=True)
        while self.running:
            try:
                task = self.memory.fetch_pending_task()
                if task:
                    self.process_task(task)
                else:
                    time.sleep(self.interval)
            except Exception as e:
                print(f"[BackgroundWorker] Loop Error: {e}", flush=True)
                time.sleep(5.0) # Backoff on error

    def stop(self):
        self.running = False

    def process_task(self, task: Dict[str, Any]):
        task_id = task['id']
        print(f"[BackgroundWorker] Processing Task {task_id}...", flush=True)
        self.memory.mark_task_processing(task_id)

        user_input = task['user_input']
        agent_output = task['agent_output']

        prompt = f"""
        Analyze this interaction. Extract Structural Knowledge (MFR) and General Principles.
        
        1. **Specific Analysis**: If a problem is presented, define:
           - Entities, State Variables, Actions, Constraints.
           - Summary of the solution.

        2. **Abstraction & Generalization**:
           - Abstract the specific details into a general pattern or rule.
           - Identify the underlying problem class (e.g., "Constraint Satisfaction", "Resource Allocation").
           - Define a general strategy derived from this instance.
        
        User: {user_input}
        AI: {agent_output}
        
        Output JSON only:
        {{
            "analysis": "**Specific Model**:\\n[Details...]\\n\\n**Generalization**:\\n[Details...]",
            "entities": ["list", "of", "key", "entities"],
            "problem_class": "Identified Problem Class",
            "should_store": true/false
        }}
        """
        
        try:
            res = call_llm_with_retry(self.llm, [HumanMessage(content=prompt)]).content.strip()
            # Basic cleanup
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            elif "```" in res: res = res.split("```")[1].split("```")[0]
            
            data = json.loads(res)
            
            if data.get('should_store', False):
                print(f"[BackgroundWorker] Storing analysis for Task {task_id}", flush=True)
                self.memory.add(
                    content=data.get('analysis', ''),
                    entities=data.get('entities', []),
                    problem_class=data.get('problem_class', '')
                )
            
            self.memory.mark_task_complete(task_id)
            print(f"[BackgroundWorker] Task {task_id} Completed.", flush=True)

        except Exception as e:
            print(f"[BackgroundWorker] Task {task_id} Failed: {e}", flush=True)
            self.memory.mark_task_failed(task_id, str(e))


# --- 4. ACE Agent Nodes ---

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
def call_llm_with_retry(llm, messages):
    return llm.invoke(messages)

def build_ace_agent(llm: ChatOpenAI, memory: ACE_Memory, use_tools: bool = True):
    
    tools = []
    if use_tools:
        # Tool Definition for the Agent
        @tool
        def search_memory_tool(query: str):
            """Searches the agent's long-term memory for relevant information, facts, or past experiences."""
            docs = memory.search(query)
            if not docs:
                return "No relevant information found in memory."
            return "\n\n".join(docs)
        tools = [search_memory_tool]

    # Use prebuilt ToolNode only if tools are enabled
    tool_node_instance = ToolNode(tools) if use_tools else None

    def tool_executor_node(state: AgentState):
        """Executes tools and appends output to history."""
        if not tool_node_instance:
            return {} # Should not be called if tools are disabled
        result = tool_node_instance.invoke(state)
        return {"messages": state['messages'] + result['messages']}

    llm_with_tools = llm.bind_tools(tools) if use_tools else llm

    def curator_node(state: AgentState):
        """Analyzes intent and retrieves context."""
        messages = state['messages']
        last_user_msg = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        if not last_user_msg:
            return {"context_docs": [], "extracted_entities": [], "problem_class": ""}

        # Construct history text (last 5 messages for context)
        history_txt = "\n".join([f"{type(m).__name__}: {m.content}" for m in messages[-5:-1]])

        # Intent Analysis Prompt
        prompt = f"""
        Based on the conversation history, analyze the user's latest request.
        
        Target TWO things:
        1. Specific entities and facts mentioned in the request.
        2. Abstract problem classes, structural patterns, or general principles relevant to the request.
        
        User Request: "{last_user_msg.content}"
        History: 
        {history_txt}
        
        Output JSON only:
        {{
            "entities": ["entity1", "entity2"],
            "problem_class": "Abstract Problem Class",
            "search_query": "Single effective search query string combining specific entities and abstract concepts"
        }}
        """
        
        try:
            res = call_llm_with_retry(llm, [HumanMessage(content=prompt)]).content.strip()
            # Basic cleanup
            if "```json" in res: res = res.split("```json")[1].split("```")[0]
            elif "```" in res: res = res.split("```")[1].split("```")[0]
            
            data = json.loads(res)
            entities = data.get("entities", [])
            p_class = data.get("problem_class", "")
            query = data.get("search_query", last_user_msg.content)
            
            # Hybrid Retrieval
            docs = memory.search(query)
            
            # Inject Context
            context_msg = []
            if docs:
                context_str = "\n".join(docs)
                context_msg = [SystemMessage(content=f"--- Retrieved Context ---\n{context_str}\n-----------------------")]
            
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
            response = call_llm_with_retry(llm_with_tools, messages)
            return {"messages": messages + [response]}
        except Exception as e:
            return {"messages": messages + [AIMessage(content=f"Error in Agent: {e}")]}

    def reflector_node(state: AgentState):
        """Queues the interaction for background analysis (Reflector)."""
        messages = state['messages']
        # Extract last Human and AI interaction
        human_msgs = [m for m in messages if isinstance(m, HumanMessage)]
        ai_msgs = [m for m in messages if isinstance(m, AIMessage) and m.content]
        
        if not human_msgs or not ai_msgs:
            return {}

        last_human = human_msgs[-1]
        last_ai = ai_msgs[-1]

        # Enqueue for background processing
        try:
            print(f"[Reflector] Enqueueing interaction for background processing...", flush=True)
            memory.enqueue_task(last_human.content, last_ai.content)
            return {
                "lesson_learned": "Analysis queued in background.",
                "should_store": True # Indicates the process was initiated
            }
        except Exception as e:
            print(f"[Reflector] Error enqueueing task: {e}", flush=True)
            return {"should_store": False, "lesson_learned": "Error enqueueing reflection."}

    # --- Graph Definition ---
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

        workflow.add_conditional_edges(
            "agent",
            check_tool_call,
            {
                "tool_executor": "tool_executor",
                "reflector": "reflector"
            }
        )
        workflow.add_edge("tool_executor", "agent")
    else:
        # If no tools, agent always goes to reflector
        workflow.add_edge("agent", "reflector")

    workflow.add_edge("reflector", END)

    return workflow.compile()
