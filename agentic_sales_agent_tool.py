"""
Agentic Sales Agent Tool
-------------------------
Replica of the /api/sales-agent Flask API endpoint for use in agentic workflows.

This tool handles the complete step-based product selection workflow:
1. initialInput - Initial product requirements
2. awaitAdditionalAndLatestSpecs - Additional specifications
3. awaitAdvancedSpecs - Advanced parameter specifications
4. showSummary - Display requirements summary
5. finalAnalysis - Trigger product analysis

Integrates with:
- Validation Tool (Step 1: Schema generation)
- Advanced Parameters Discovery
- Requirements Collection
"""

import json
import logging
import re
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Import required components
from chaining import setup_ai_components
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import prompts
from advanced_parameters import discover_advanced_parameters

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SalesAgentTool:
    """
    Sales Agent Tool for Agentic Workflows

    Replicates the functionality of /api/sales-agent endpoint
    for step-based product selection workflow.
    """

    def __init__(self):
        """Initialize sales agent tool with AI components"""
        logger.info("[SalesAgentTool] Initializing AI components...")
        self.components = setup_ai_components()
        self.session_state = {}  # In-memory session state for agentic workflow
        logger.info("[SalesAgentTool] Sales agent tool ready")

    def format_available_parameters(self, params: List) -> str:
        """
        Format parameter list for display.

        Args:
            params: List of parameter dictionaries or strings

        Returns:
            Formatted string with bullet points
        """
        formatted = []
        for param in params:
            # Extract parameter name
            if isinstance(param, dict):
                name = param.get('name') or param.get('key') or (list(param.keys())[0] if param else '')
            else:
                name = str(param).strip()

            # Format name
            name = name.replace('_', ' ')
            name = re.split(r'[\(\[\{]', name, 1)[0].strip()
            name = " ".join(name.split())
            name = name.title()

            formatted.append(f"- {name}")

        return "\n".join(formatted)

    def detect_yes_no_response(self, user_message: str) -> Optional[str]:
        """
        Detect if user message is a yes/no response.

        Args:
            user_message: User's input message

        Returns:
            'yes', 'no', or None
        """
        if not user_message:
            return None

        user_lower = user_message.lower().strip()

        affirmative_keywords = ['yes', 'y', 'yeah', 'yep', 'sure', 'ok', 'okay', 'proceed', 'continue', 'go ahead']
        negative_keywords = ['no', 'n', 'nope', 'skip', 'none', 'not needed', 'done', 'not interested']

        for keyword in affirmative_keywords:
            if keyword in user_lower:
                return 'yes'

        for keyword in negative_keywords:
            if keyword in user_lower:
                return 'no'

        return None

    def get_session_state(self, session_id: str, key: str, default: Any = None) -> Any:
        """Get value from session state"""
        session_key = f"{session_id}_{key}"
        return self.session_state.get(session_key, default)

    def set_session_state(self, session_id: str, key: str, value: Any):
        """Set value in session state"""
        session_key = f"{session_id}_{key}"
        self.session_state[session_key] = value

    def handle_knowledge_question(
        self,
        user_message: str,
        current_step: str
    ) -> Dict[str, Any]:
        """
        Handle knowledge questions and provide context-aware responses.

        Args:
            user_message: User's question
            current_step: Current workflow step

        Returns:
            Response with answer and workflow continuation
        """
        try:
            # Determine context hint based on current step
            context_hints = {
                "awaitMissingInfo": "Once you have the information you need, please provide the missing details so we can continue with your product selection or Would you like to continue anyway?",
                "awaitAdditionalAndLatestSpecs": "Now, let's continue - would you like to add additional and latest specifications?",
                "awaitAdvancedSpecs": "Now, let's continue with advanced specifications.",
                "showSummary": "Now, let's proceed with your product analysis.",
            }
            context_hint = context_hints.get(current_step, "Now, let's continue with your product selection.")

            # Build and execute LLM chain
            response_chain = prompts.sales_agent_knowledge_question_prompt | self.components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({
                "user_message": user_message,
                "context_hint": context_hint
            })

            return {
                "success": True,
                "content": llm_response,
                "nextStep": current_step,  # Resume same step
                "maintainWorkflow": True,
                "isKnowledgeQuestion": True
            }

        except Exception as e:
            logger.error(f"[SalesAgentTool] Knowledge question handling failed: {e}")
            return {
                "success": False,
                "content": "I apologize, but I couldn't process your question. Could you rephrase it?",
                "nextStep": current_step,
                "error": str(e)
            }

    def handle_initial_input(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str,
        save_immediately: bool = False
    ) -> Dict[str, Any]:
        """
        Handle initialInput step.

        Args:
            user_message: User's message
            data_context: Context data
            session_id: Session ID
            save_immediately: Whether to save and skip greeting

        Returns:
            Response with next step
        """
        product_type = data_context.get('productType', 'a product')

        if save_immediately:
            # Save product type and skip to next step
            self.set_session_state(session_id, 'product_type', product_type)
            self.set_session_state(session_id, 'current_step', 'initialInput')

            return {
                "success": True,
                "content": f"Saved product type: {product_type}",
                "nextStep": "awaitAdditionalAndLatestSpecs"
            }

        # Generate greeting response
        try:
            prompt_template = prompts.sales_agent_initial_input_prompt
            response_chain = prompt_template | self.components['llm'] | StrOutputParser()

            llm_response = response_chain.invoke({
                "user_input": user_message,
                "product_type": product_type,
                "search_session_id": session_id
            })

            return {
                "success": True,
                "content": llm_response,
                "nextStep": "awaitAdditionalAndLatestSpecs"
            }

        except Exception as e:
            logger.error(f"[SalesAgentTool] Initial input failed: {e}")
            return {
                "success": False,
                "content": "I'm ready to help you find the right product. What are your requirements?",
                "nextStep": "awaitAdditionalAndLatestSpecs",
                "error": str(e)
            }

    def handle_additional_specs(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle awaitAdditionalAndLatestSpecs step.

        Args:
            user_message: User's message
            data_context: Context data
            session_id: Session ID

        Returns:
            Response with next step
        """
        user_lower = user_message.lower().strip()

        # Check session state for interaction stage
        asking_state_key = f'awaiting_additional_specs_yesno'
        is_awaiting_yesno = self.get_session_state(session_id, asking_state_key, True)

        yes_no = self.detect_yes_no_response(user_message)

        if is_awaiting_yesno:
            # Asking yes/no question
            if yes_no == 'no':
                # User says NO → skip to summary
                self.set_session_state(session_id, asking_state_key, False)

                try:
                    prompt_template = prompts.sales_agent_no_additional_specs_prompt
                    response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                    llm_response = response_chain.invoke({
                        "user_input": user_message,
                        "product_type": data_context.get('productType', ''),
                        "search_session_id": session_id
                    })
                except:
                    llm_response = "Understood. Let me prepare the summary of your requirements."

                return {
                    "success": True,
                    "content": llm_response,
                    "nextStep": "showSummary"
                }

            elif yes_no == 'yes':
                # User says YES → ask for specifications
                self.set_session_state(session_id, asking_state_key, False)

                available_parameters = data_context.get('availableParameters', [])

                if available_parameters:
                    params_display = self.format_available_parameters(available_parameters)
                    try:
                        prompt_template = prompts.sales_agent_yes_additional_specs_prompt
                        response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                        llm_response = response_chain.invoke({
                            "user_input": user_message,
                            "product_type": data_context.get('productType', ''),
                            "params_display": params_display,
                            "search_session_id": session_id
                        })
                    except:
                        llm_response = f"Great! Here are the additional specifications:\n\n{params_display}\n\nPlease enter the specifications you'd like to include."
                else:
                    llm_response = "Please enter your additional specifications."

                return {
                    "success": True,
                    "content": llm_response,
                    "nextStep": "awaitAdditionalAndLatestSpecs"
                }

            else:
                # Invalid response
                return {
                    "success": True,
                    "content": "Please respond with yes or no. Additional and latest specifications are available. Would you like to add them?",
                    "nextStep": "awaitAdditionalAndLatestSpecs"
                }

        else:
            # Collecting actual specifications
            self.set_session_state(session_id, 'additional_specs_input', user_message)
            self.set_session_state(session_id, asking_state_key, True)  # Reset for next time

            try:
                prompt_template = prompts.sales_agent_acknowledge_additional_specs_prompt
                response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                llm_response = response_chain.invoke({
                    "user_input": user_message,
                    "product_type": data_context.get('productType', ''),
                    "search_session_id": session_id
                })
            except:
                llm_response = "Thank you for providing the additional specifications. Moving to advanced parameters..."

            return {
                "success": True,
                "content": llm_response,
                "nextStep": "awaitAdvancedSpecs",
                "additional_specs_provided": user_message
            }

    def handle_advanced_specs(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle awaitAdvancedSpecs step with parameter discovery.

        Args:
            user_message: User's message
            data_context: Context data
            session_id: Session ID

        Returns:
            Response with next step
        """
        user_lower = user_message.lower().strip()
        product_type = data_context.get('productType') or self.get_session_state(session_id, 'product_type')
        available_parameters = data_context.get('availableParameters', [])
        selected_parameters = data_context.get('selectedParameters', {})
        total_selected = data_context.get('totalSelected', 0)

        yes_no = self.detect_yes_no_response(user_message)

        # Check if parameters need to be discovered
        if not available_parameters or len(available_parameters) == 0:
            logger.info(f"[SalesAgentTool] Discovering advanced parameters for: {product_type}")

            # Check if this is an error/retry scenario
            parameter_error = data_context.get('parameterError', False)
            no_params_found = data_context.get('no_params_found', False)

            # Handle discovery
            if product_type:
                try:
                    # Discover advanced parameters
                    parameters_result = discover_advanced_parameters(product_type)
                    discovered_params = (
                        parameters_result.get('unique_parameters') or
                        parameters_result.get('unique_specifications', [])
                    )
                    discovered_params = discovered_params[:15] if discovered_params else []

                    # Store in data context
                    data_context['availableParameters'] = discovered_params
                    data_context['no_params_found'] = len(discovered_params) == 0

                    if len(discovered_params) == 0:
                        # No parameters found
                        llm_response = (
                            "No advanced parameters were found. "
                            "Do you want to proceed to summary?"
                        )
                        next_step = "awaitAdvancedSpecs"
                    else:
                        # Parameters found
                        params_display = self.format_available_parameters(discovered_params)
                        llm_response = (
                            f"These advanced parameters were identified:\n\n"
                            f"{params_display}\n\n"
                            "Do you want to add these advanced parameters?"
                        )
                        next_step = "awaitAdvancedSpecs"

                    return {
                        "success": True,
                        "content": llm_response,
                        "nextStep": next_step,
                        "discoveredParameters": discovered_params,
                        "dataContext": data_context
                    }

                except Exception as e:
                    logger.error(f"[SalesAgentTool] Parameter discovery failed: {e}")
                    return {
                        "success": False,
                        "content": "I encountered an issue discovering advanced parameters. Would you like to skip this step?",
                        "nextStep": "awaitAdvancedSpecs",
                        "error": str(e)
                    }
            else:
                return {
                    "success": False,
                    "content": "I'm having trouble accessing advanced parameters because the product type isn't clear. Would you like to skip this step?",
                    "nextStep": "awaitAdvancedSpecs"
                }

        # Parameters already available - handle user response
        if yes_no == 'yes':
            # User wants to add parameters
            try:
                prompt_template = prompts.sales_agent_advanced_specs_yes_prompt
                response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                params_display = self.format_available_parameters(available_parameters)
                llm_response = response_chain.invoke({
                    "user_input": user_message,
                    "product_type": product_type,
                    "params_display": params_display,
                    "search_session_id": session_id
                })
            except:
                params_display = self.format_available_parameters(available_parameters)
                llm_response = f"Here are the parameters:\n\n{params_display}\n\nPlease specify the values you need."

            return {
                "success": True,
                "content": llm_response,
                "nextStep": "awaitAdvancedSpecs"
            }

        elif yes_no == 'no':
            # User declines advanced parameters
            try:
                prompt_template = prompts.sales_agent_advanced_specs_no_prompt
                response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                llm_response = response_chain.invoke({
                    "user_input": user_message,
                    "product_type": product_type,
                    "search_session_id": session_id
                })
            except:
                llm_response = "Understood. Let me prepare the summary of your requirements."

            return {
                "success": True,
                "content": llm_response,
                "nextStep": "showSummary"
            }

        elif total_selected > 0 or user_message.strip():
            # User provided parameter values
            selected_names = [
                param.replace('_', ' ').title()
                for param in selected_parameters.keys()
            ] if selected_parameters else []

            if selected_names:
                selected_display = ", ".join(selected_names)
                llm_response = f"**Added Advanced Parameters:** {selected_display}\n\nProceeding to the summary now."
            else:
                llm_response = "Thank you for providing the advanced specifications. Proceeding to the summary now."

            return {
                "success": True,
                "content": llm_response,
                "nextStep": "showSummary",
                "selectedParameters": selected_parameters
            }

        else:
            # Default: ask again
            return {
                "success": True,
                "content": "Please respond with yes or no. These additional advanced parameters were identified. Would you like to add them?",
                "nextStep": "awaitAdvancedSpecs"
            }

    def handle_show_summary(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle showSummary step.

        Args:
            user_message: User's message
            data_context: Context data
            session_id: Session ID

        Returns:
            Response with next step
        """
        user_lower = user_message.lower().strip()
        yes_no = self.detect_yes_no_response(user_message)

        # Check for proceed confirmation
        proceed_patterns = ['run', 'analyze', 'search', 'find', 'start', 'do it', 'let\'s go', 'confirm']
        wants_to_proceed = yes_no == 'yes' or any(pattern in user_lower for pattern in proceed_patterns)

        if wants_to_proceed:
            try:
                prompt_template = prompts.sales_agent_show_summary_proceed_prompt
                response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                llm_response = response_chain.invoke({
                    "user_input": user_message,
                    "product_type": data_context.get('productType', ''),
                    "search_session_id": session_id
                })
            except:
                llm_response = "Starting the product analysis now. Please wait while I search for matching products..."

            return {
                "success": True,
                "content": llm_response,
                "nextStep": "finalAnalysis",
                "proceed": True
            }

        else:
            # Show summary intro
            try:
                prompt_template = prompts.sales_agent_show_summary_intro_prompt
                response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                llm_response = response_chain.invoke({
                    "user_input": user_message,
                    "product_type": data_context.get('productType', ''),
                    "search_session_id": session_id
                })
            except:
                llm_response = "Here is your requirements summary. Would you like to proceed with the product search?"

            return {
                "success": True,
                "content": llm_response,
                "nextStep": "showSummary",
                "showSummary": True
            }

    def handle_final_analysis(
        self,
        user_message: str,
        data_context: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """
        Handle finalAnalysis step.

        Args:
            user_message: User's message
            data_context: Context data
            session_id: Session ID

        Returns:
            Response with completion
        """
        ranked_products = data_context.get('analysisResult', {}).get('overallRanking', {}).get('rankedProducts', [])
        matching_products = [p for p in ranked_products if p.get('requirementsMatch') is True]
        count = len(matching_products)

        try:
            prompt_template = prompts.sales_agent_final_analysis_prompt
            response_chain = prompt_template | self.components['llm'] | StrOutputParser()
            llm_response = response_chain.invoke({
                "user_input": user_message,
                "product_type": data_context.get('productType', ''),
                "count": count,
                "search_session_id": session_id
            })
        except:
            llm_response = f"Analysis complete. Found {count} matching products."

        return {
            "success": True,
            "content": llm_response,
            "nextStep": None,  # End of workflow
            "complete": True,
            "matchingProductsCount": count
        }

    def process_step(
        self,
        step: str,
        user_message: str,
        data_context: Optional[Dict[str, Any]] = None,
        intent: Optional[str] = None,
        session_id: str = "default",
        save_immediately: bool = False
    ) -> Dict[str, Any]:
        """
        Main entry point - process a workflow step.

        Args:
            step: Current workflow step
            user_message: User's message
            data_context: Context data
            intent: Classified intent
            session_id: Session identifier
            save_immediately: Whether to save immediately (initialInput)

        Returns:
            Response with content and next step
        """
        try:
            logger.info(f"[SalesAgentTool] Processing step: {step}, Session: {session_id}")

            data_context = data_context or {}

            # Detect short yes/no responses and override intent
            short_yesno_re = re.compile(r"^\s*(?:yes|y|yeah|yep|sure|ok|okay|no|n|nope|skip)\b[\.\!\?\s]*$", re.IGNORECASE)
            if user_message and short_yesno_re.match(user_message):
                if intent == 'knowledgeQuestion':
                    logger.info(f"[SalesAgentTool] Overriding intent for short reply: {user_message}")
                intent = 'workflow'

            # Handle knowledge questions
            if intent == "knowledgeQuestion":
                return self.handle_knowledge_question(user_message, step)

            # Route to appropriate step handler
            if step == 'initialInput':
                result = self.handle_initial_input(user_message, data_context, session_id, save_immediately)

            elif step == 'awaitAdditionalAndLatestSpecs':
                result = self.handle_additional_specs(user_message, data_context, session_id)

            elif step == 'awaitAdvancedSpecs':
                result = self.handle_advanced_specs(user_message, data_context, session_id)

            elif step == 'showSummary':
                result = self.handle_show_summary(user_message, data_context, session_id)

            elif step == 'finalAnalysis':
                result = self.handle_final_analysis(user_message, data_context, session_id)

            elif step == 'greeting':
                try:
                    prompt_template = prompts.sales_agent_greeting_prompt
                    response_chain = prompt_template | self.components['llm'] | StrOutputParser()
                    llm_response = response_chain.invoke({
                        "user_input": user_message,
                        "product_type": data_context.get('productType', ''),
                        "search_session_id": session_id
                    })
                except:
                    llm_response = "Hello! I'm Engenie, your AI sales agent. How can I help you find the right industrial equipment today?"

                result = {
                    "success": True,
                    "content": llm_response,
                    "nextStep": "initialInput"
                }

            else:
                # Default fallback
                logger.warning(f"[SalesAgentTool] Unknown step: {step}")
                result = {
                    "success": True,
                    "content": "I'm here to help you find the right product. What are your requirements?",
                    "nextStep": "initialInput"
                }

            # Update session state
            if result.get('nextStep'):
                self.set_session_state(session_id, 'current_step', result['nextStep'])
                self.set_session_state(session_id, 'current_intent', 'workflow')

            return result

        except Exception as e:
            logger.error(f"[SalesAgentTool] Process step failed: {e}", exc_info=True)
            return {
                "success": False,
                "content": "I apologize, but I'm having technical difficulties. Please try again.",
                "nextStep": step,
                "error": str(e)
            }


# Convenience function
def process_sales_agent_step(
    step: str,
    user_message: str,
    data_context: Optional[Dict[str, Any]] = None,
    intent: Optional[str] = None,
    session_id: str = "default",
    save_immediately: bool = False
) -> Dict[str, Any]:
    """
    Convenience function for sales agent workflow.

    Args:
        step: Current workflow step
        user_message: User's message
        data_context: Context data
        intent: Classified intent
        session_id: Session ID
        save_immediately: Save immediately flag

    Returns:
        Response dictionary
    """
    tool = SalesAgentTool()
    return tool.process_step(step, user_message, data_context, intent, session_id, save_immediately)


# Example usage
if __name__ == "__main__":
    print("\n" + "="*70)
    print("AGENTIC SALES AGENT TOOL - STEP-BY-STEP WORKFLOW")
    print("="*70)

    # Initialize tool
    agent = SalesAgentTool()

    # Example workflow
    session_id = "test_session_001"

    # Step 1: Initial Input
    print("\n--- Step 1: Initial Input ---")
    result1 = agent.process_step(
        step="initialInput",
        user_message="I need a pressure transmitter",
        data_context={"productType": "Pressure Transmitter"},
        session_id=session_id
    )
    print(f"Response: {result1['content'][:200]}...")
    print(f"Next Step: {result1['nextStep']}")

    # Step 2: Additional Specs
    print("\n--- Step 2: Additional Specs (User says 'yes') ---")
    result2 = agent.process_step(
        step="awaitAdditionalAndLatestSpecs",
        user_message="yes",
        data_context={
            "productType": "Pressure Transmitter",
            "availableParameters": ["accuracy", "outputSignal", "processConnection"]
        },
        session_id=session_id
    )
    print(f"Response: {result2['content'][:200]}...")
    print(f"Next Step: {result2['nextStep']}")

    # Step 3: Advanced Specs
    print("\n--- Step 3: Advanced Specs ---")
    result3 = agent.process_step(
        step="awaitAdvancedSpecs",
        user_message="show me",
        data_context={"productType": "Pressure Transmitter"},
        session_id=session_id
    )
    print(f"Response: {result3['content'][:200]}...")
    print(f"Next Step: {result3['nextStep']}")

    print("\n✓ Sales Agent Tool workflow demonstration complete")
