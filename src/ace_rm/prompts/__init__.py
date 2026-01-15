import os
from . import en, ja

# Select prompts based on the ACE_LANG environment variable
ACE_LANG = os.environ.get("ACE_LANG", "en").lower()

if ACE_LANG == "ja":
    from .ja import *
else:
    from .en import *
