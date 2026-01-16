# tools_prompts.py
# DEPRECATED - This file is deprecated and will be removed in v2.0
# Import from 'prompts' module instead: from prompts import <prompt_name>
#
# This file now serves as a backward compatibility layer that re-exports
# all prompts from the consolidated prompts.py module.
#
# Migration Guide:
# OLD: from tools_prompts import instrument_identifier_prompt
# NEW: from prompts import instrument_identifier_prompt

import warnings
warnings.warn(
    "tools_prompts.py is deprecated and will be removed in v2.0. "
    "Import directly from 'prompts' module instead: from prompts import <prompt_name>",
    DeprecationWarning,
    stacklevel=2
)

# Re-export everything from prompts.py for backward compatibility
from prompts import *
