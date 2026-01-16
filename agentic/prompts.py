# DEPRECATION WARNING: This module is deprecated
# Import prompts directly from 'prompts' module instead of 'agentic.prompts'
# This file will be removed in v2.0
import warnings
warnings.warn(
    "agentic.prompts module is deprecated and will be removed in v2.0. "
    "Import directly from 'prompts' module instead: from prompts import <prompt_name>",
    DeprecationWarning,
    stacklevel=2
)

# Import from consolidated prompts module
from prompts import (
    intent_classifier_prompt,
    feedback_positive_prompt,
    feedback_negative_prompt,
    feedback_comment_prompt as feedback_general_prompt,  # Fixed: was feedback_general_prompt
    validation_prompt,
    schema_prompt,
    additional_requirements_prompt,
    agentic_field_description_prompt as field_description_prompt,  # Fixed: was field_description_prompt
    instrument_identifier_prompt,
    vendor_search_prompt,
    advanced_parameters_discovery_prompt as advanced_parameters_prompt,
    advanced_parameter_selection_prompt as advanced_parameters_selection_prompt,
    image_search_prompt,
    generic_image_prompt,
    analysis_product_images_prompt,
    pdf_search_prompt,
    file_upload_prompt,
    vendor_data_prompt,
    submodel_mapping_prompt,
    price_review_prompt,
    project_management_prompt,
    standardization_prompt,
    strategy_prompt,
    vendor_summary_prompt,  # NEW: Summary Agent prompt
    judge_prompt,
    ranking_agent_prompt,
    comparison_analysis_prompt,  # NEW: Comparison Agent prompt
    grounded_chat_prompt,  # NEW: Grounded Knowledge Chat prompt
    sales_agent_greeting_prompt,  # Sales agent workflow prompts
    sales_agent_initial_input_prompt,
    sales_agent_no_additional_specs_prompt,
    sales_agent_yes_additional_specs_prompt
)

# Alias for backward compatibility if needed
intent_classifier_classifier_prompt = intent_classifier_prompt

# feedback_prompt is used in agentic_main.py, but it was replaced by step-specific ones in backend/prompts.py
from langchain_core.prompts import ChatPromptTemplate
feedback_prompt = ChatPromptTemplate.from_template("""
You are Engenie - a helpful assistant processing user feedback.

Feedback Type: {feedback_type}
User Comment: {comment}
User Query: {user_query}

Based on the feedback type:
- If positive: Thank the user warmly
- If negative: Show empathy, apologize, and acknowledge their input
- If general: Acknowledge and thank them

Keep your response to 1-2 professional sentences.
""")

sales_agent_prompt_template = """
You are Engenie - a helpful sales agent.

Current Step: {step}
Product Type: {product_type}
User Message: {user_message}
Available Parameters: {available_parameters}

Based on the current workflow step, provide an appropriate response that:
1. Maintains conversational flow
2. Guides the user through the product selection process
3. Handles their input professionally
4. Moves the workflow forward appropriately

Respond naturally and helpfully.
"""

# Alias for compatibility with other modules (e.g. agents.py)
sales_agent_prompt = sales_agent_prompt_template
