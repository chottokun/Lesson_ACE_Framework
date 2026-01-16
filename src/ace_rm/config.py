"""
Centralized configuration for the ACE-RM project.
All environment variables and constants are managed here.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
DB_PATH = os.environ.get("ACE_DB_PATH", "ace_memory.db")
FAISS_INDEX_PATH = os.environ.get("ACE_FAISS_INDEX_PATH", "ace_memory.faiss")

# --- LLM Configuration ---
MODEL_NAME = os.environ.get("LLM_MODEL", "gpt-oss-120b")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.ai.sakura.ad.jp/v1/")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.0"))
OPENAI_API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("SAKURA_API_KEY", "dummy_key")

# --- Embedding Model Configuration ---
EMBEDDING_MODEL_NAME = os.environ.get("ACE_EMBEDDING_MODEL", "cl-nagoya/ruri-v3-30m")
ACE_DEVICE = os.environ.get("ACE_DEVICE")  # Default is None for auto-detection

# --- Search Configuration ---
DISTANCE_METRIC = os.environ.get("ACE_DISTANCE_METRIC", "l2").lower()
_default_threshold = "0.7" if DISTANCE_METRIC == "cosine" else "1.8"
DISTANCE_THRESHOLD = float(os.environ.get("ACE_DISTANCE_THRESHOLD", _default_threshold))

# --- Language Configuration ---
ACE_LANG = os.environ.get("ACE_LANG", "en").lower()

# --- LTM Mode Configuration ---
LTM_MODE = os.environ.get("LTM_MODE", "shared").lower()
