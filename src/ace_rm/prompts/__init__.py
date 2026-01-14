import os
from . import en, ja

# Get language from environment variable (default: en)
lang = os.environ.get("ACE_LANG", "en").lower()

if lang == "ja":
    module = ja
else:
    module = en

# Export all prompts from the selected module
ANALYSIS_PROMPT = module.ANALYSIS_PROMPT
SYNTHESIZER_PROMPT = module.SYNTHESIZER_PROMPT
INTENT_ANALYSIS_PROMPT = module.INTENT_ANALYSIS_PROMPT
RETRIEVED_CONTEXT_TEMPLATE = module.RETRIEVED_CONTEXT_TEMPLATE
