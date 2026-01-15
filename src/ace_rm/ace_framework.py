import os
from dotenv import load_dotenv

# Re-exporting for backward compatibility
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue
from ace_rm.agent.graph import AgentState, build_ace_agent, call_llm_with_retry
from ace_rm.workers.background import BackgroundWorker

# Load environment variables
load_dotenv()

# --- Configuration Constants (Centralized) ---
DB_PATH = os.environ.get("ACE_DB_PATH", "ace_memory.db")
FAISS_INDEX_PATH = os.environ.get("ACE_FAISS_INDEX_PATH", "ace_memory.faiss")

# LLM Configuration
MODEL_NAME = os.environ.get("LLM_MODEL", "gpt-oss-120b")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.ai.sakura.ad.jp/v1/")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))

# API Key handling
OPENAI_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("SAKURA_API_KEY", "dummy_key")
