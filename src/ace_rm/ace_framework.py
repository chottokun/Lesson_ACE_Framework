"""
ACE Framework - Facade module for backward compatibility.
All configuration is now centralized in ace_rm.config.
"""

# Re-exporting core components for backward compatibility
from ace_rm.memory.core import ACE_Memory
from ace_rm.memory.queue import TaskQueue
from ace_rm.agent.graph import AgentState, build_ace_agent, call_llm_with_retry
from ace_rm.workers.background import BackgroundWorker

# Re-exporting configuration constants for backward compatibility
from ace_rm.config import (
    DB_PATH,
    FAISS_INDEX_PATH,
    MODEL_NAME,
    BASE_URL,
    LLM_TEMPERATURE,
    OPENAI_API_KEY,
)
