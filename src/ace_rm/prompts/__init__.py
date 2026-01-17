import os
from . import en, ja  # noqa: F401

# Select prompts based on the ACE_LANG environment variable
ACE_LANG = os.environ.get("ACE_LANG", "en").lower()

if ACE_LANG == "ja":
    from .ja import *  # noqa: F403
else:
    from .en import *  # noqa: F403
