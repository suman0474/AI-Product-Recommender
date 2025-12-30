# agentic/router_agent.py
# Master Router for Intelligent Workflow Selection
# Routes user input to the appropriate agentic workflow based on intent

import re
import logging
from typing import Dict, Any, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .models import RoutingDecision, WorkflowType, AmbiguityLevel
from datetime import datetime
import os
from llm_fallback import create_llm_with_fallback

logger = logging.getLogger(__name__)


# ============================================================================
# WORKFLOW ROUTER CLASS
# ============================================================================

class WorkflowRouter:
    """
    Master router for intelligent workflow selection.
    Uses hybrid classification: rule-based for speed, LLM for flexibility.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.llm = create_llm_with_fallback(
            model=model_name,
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        # Define workflow patterns for rule-based classification
        # 3 Main Workflows: Solution, Instrument Detail, Grounded Chat
        self.patterns = {
            "instrument_detail": {
                "keywords": ["project", "BOM", "bill of materials", "identify instruments",
                            "instrumentation", "crude unit", "refinery", "plant", "facility",
                            "water treatment", "distillation", "reactor", "accessories",
                            "instrument list", "equipment list"],
                "context_words": ["for", "in the", "for the", "needed for"],
                "weight": 1.0
            },
            "grounded_chat": {
                "keywords": ["what is", "explain", "tell me about", "how does", "why",
                            "difference between", "can you explain", "what are",
                            "define", "meaning of", "help me understand"],
                "question_words": ["?", "question", "clarify", "wondering"],
                "weight": 0.9
            },
            "solution": {
                "keywords": ["need", "looking for", "find me", "search", "buy", "purchase",
                            "recommend", "suggest", "design", "solution", "system",
                            "best approach", "configure", "set up"],
                "product_types": ["transmitter", "sensor", "valve", "pump", "meter", "gauge",
                                 "switch", "controller", "actuator", "pressure", "flow",
                                 "temperature", "level"],
                "weight": 0.8
            }
        }

    def _check_domain_validity(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Check if input is within industrial/procurement domain.
        Returns invalid intent if query is clearly out of domain.

        Returns:
            Dict with workflow='invalid' if out of domain, None otherwise
        """
        lowered = user_input.lower()

        # Out-of-domain indicators (non-industrial topics)
        invalid_keywords = [
            # General topics
            "weather", "forecast", "temperature today", "climate",
            # Entertainment
            "joke", "funny", "meme", "game", "movie", "film", "music", "song",
            "tv show", "series", "anime", "video game", "gaming",
            # Food & lifestyle
            "recipe", "cooking", "food", "restaurant", "menu", "dinner",
            "breakfast", "lunch", "diet", "nutrition",
            # Travel & hospitality
            "flight", "hotel", "vacation", "trip", "travel", "booking",
            "airport", "airline", "resort",
            # Finance & markets (non-procurement)
            "stock market", "cryptocurrency", "bitcoin", "trading", "forex",
            "investment portfolio", "stock price",
            # News & current events
            "news", "headline", "breaking news", "politics", "election",
            # Sports
            "football", "soccer", "basketball", "baseball", "cricket",
            "sports score", "match result",
            # Personal services
            "doctor appointment", "medical advice", "health insurance",
            "lawyer", "legal advice",
            # Technology (consumer)
            "smartphone", "iphone", "android app", "social media",
            "facebook", "instagram", "twitter", "tiktok",
            # Education (non-technical)
            "homework help", "essay writing", "school assignment"
        ]

        # Industrial/procurement domain indicators
        industrial_keywords = [
            # Instruments
            "instrument", "transmitter", "sensor", "detector", "gauge",
            "meter", "analyzer", "controller", "actuator", "positioner",
            # Equipment
            "valve", "pump", "compressor", "heat exchanger", "vessel",
            "tank", "reactor", "separator", "filter", "strainer",
            # Process variables
            "pressure", "flow", "temperature", "level", "density",
            "viscosity", "ph", "conductivity", "humidity",
            # Process applications
            "crude oil", "refinery", "petrochemical", "chemical plant",
            "water treatment", "wastewater", "pharmaceutical",
            "food processing", "mining", "pulp and paper",
            # Industrial systems
            "process control", "automation", "scada", "dcs", "plc",
            "instrumentation", "control system", "safety system",
            # Industrial standards
            "HART", "4-20mA", "Modbus", "Profibus", "Foundation Fieldbus",
            "SIL", "ATEX", "IECEx", "API", "ASME", "ISO",
            # Project terms
            "BOM", "bill of materials", "specification", "datasheet",
            "technical data", "P&ID", "process diagram", "equipment list",
            # Procurement
            "procurement", "purchase", "vendor", "supplier", "quote",
            "RFQ", "tender", "bidding"
        ]

        # Check for explicit out-of-domain keywords
        has_invalid = any(kw in lowered for kw in invalid_keywords)
        has_industrial = any(kw in lowered for kw in industrial_keywords)

        # If clear invalid keywords and no industrial context
        if has_invalid and not has_industrial:
            return {
                "workflow": "invalid",
                "confidence": 0.90,
                "reasoning": "Out-of-domain query detected (non-industrial topic)",
                "rule_name": "invalid_domain_check"
            }

        # Additional check: very short non-industrial queries
        words = lowered.split()
        if len(words) <= 3 and has_invalid:
            return {
                "workflow": "invalid",
                "confidence": 0.85,
                "reasoning": "Short out-of-domain query",
                "rule_name": "invalid_short_query"
            }

        return None

    def _detect_chat(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Detect generic conversational intents (greetings, acknowledgments, farewells).
        Returns chat intent for simple conversational inputs.

        Returns:
            Dict with workflow='chat' if conversational, None otherwise
        """
        lowered = user_input.strip().lower()

        # Remove punctuation for matching
        cleaned = lowered.replace("!", "").replace("?", "").replace(".", "").strip()

        # Greetings
        greetings = [
            "hello", "hi", "hey", "good morning", "good afternoon",
            "good evening", "greetings", "howdy", "hi there",
            "hello there", "hey there"
        ]

        # Acknowledgments
        acknowledgments = [
            "thanks", "thank you", "appreciate it", "appreciate that",
            "got it", "okay", "ok", "sure", "alright", "sounds good",
            "that's helpful", "that helps", "perfect", "great",
            "excellent", "awesome", "wonderful", "nice", "cool",
            "understood", "makes sense", "i see", "i understand"
        ]

        # Farewells
        farewells = [
            "bye", "goodbye", "good bye", "see you", "take care",
            "later", "catch you later", "talk to you later",
            "have a good day", "have a nice day"
        ]

        # Affirmations
        affirmations = [
            "yes", "yeah", "yep", "yup", "sure", "of course",
            "absolutely", "definitely", "indeed"
        ]

        # Negations (simple)
        negations = [
            "no", "nope", "nah", "not really", "no thanks"
        ]

        # Check for exact matches or starts-with patterns

        # Greetings (high confidence)
        for greeting in greetings:
            if cleaned == greeting or cleaned.startswith(greeting + " "):
                return {
                    "workflow": "chat",
                    "confidence": 0.95,
                    "reasoning": "Greeting detected",
                    "rule_name": "chat_greeting"
                }

        # Acknowledgments (high confidence)
        # Use word boundary matching for ALL acknowledgments to avoid false positives
        # (e.g., "sure" should not match inside "pressure")
        import re
        for ack in acknowledgments:
            # Use word boundary regex to prevent substring false positives
            pattern = r'\b' + re.escape(ack) + r'\b'
            if re.search(pattern, lowered):
                return {
                    "workflow": "chat",
                    "confidence": 0.90,
                    "reasoning": "Acknowledgment detected",
                    "rule_name": "chat_acknowledgment"
                }

        # Farewells (high confidence)
        # Use word boundaries to avoid false positives
        for farewell in farewells:
            pattern = r'\b' + re.escape(farewell) + r'\b'
            if re.search(pattern, lowered):
                return {
                    "workflow": "chat",
                    "confidence": 0.95,
                    "reasoning": "Farewell detected",
                    "rule_name": "chat_farewell"
                }

        # Affirmations (medium confidence - could be part of larger query)
        if len(cleaned.split()) <= 2:
            for affirmation in affirmations:
                if cleaned == affirmation:
                    return {
                        "workflow": "chat",
                        "confidence": 0.80,
                        "reasoning": "Simple affirmation",
                        "rule_name": "chat_affirmation"
                    }

        # Negations (medium confidence)
        if len(cleaned.split()) <= 2:
            for negation in negations:
                if cleaned == negation or cleaned.startswith(negation):
                    return {
                        "workflow": "chat",
                        "confidence": 0.80,
                        "reasoning": "Simple negation",
                        "rule_name": "chat_negation"
                    }

        # Generic short phrases that are conversational
        # Use word boundaries to avoid false positives
        short_chat_phrases = [
            "how are you", "how's it going", "what's up", "whats up",
            "help me", "can you help", "need help"
        ]

        for phrase in short_chat_phrases:
            # Use word boundary to ensure complete phrase match
            pattern = r'\b' + re.escape(phrase) + r'\b'
            if re.search(pattern, lowered):
                return {
                    "workflow": "chat",
                    "confidence": 0.75,
                    "reasoning": "Conversational phrase detected",
                    "rule_name": "chat_phrase"
                }

        return None

    def route(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> RoutingDecision:
        """
        Main routing method - determines which workflow to use

        Flow:
        1. Check domain validity (invalid intent) - highest priority
        2. Check for chat intents (greetings, acknowledgments)
        3. Rule-based classification for main workflows
        4. LLM classification for ambiguous cases
        5. Hybrid merge of rule + LLM results

        Args:
            user_input: User's message
            context: Optional context (session_id, history, etc.)

        Returns:
            RoutingDecision with workflow, confidence, reasoning
        """
        logger.info(f"[ROUTER] Routing input: {user_input[:100]}...")

        # Phase 0a: Check domain validity FIRST (reject out-of-domain queries)
        invalid_check = self._check_domain_validity(user_input)
        if invalid_check:
            logger.info(f"[ROUTER] Invalid domain detected: {invalid_check['reasoning']}")
            return RoutingDecision(
                workflow=WorkflowType.INVALID,
                confidence=invalid_check['confidence'],
                reasoning=invalid_check['reasoning'],
                ambiguity_level=AmbiguityLevel.LOW,
                rule_match=invalid_check.get('rule_name'),
                llm_used=False,
                timestamp=datetime.now().isoformat()
            )

        # Phase 0b: Check for chat intents (greetings, acknowledgments, farewells)
        chat_check = self._detect_chat(user_input)
        if chat_check:
            logger.info(f"[ROUTER] Chat intent detected: {chat_check['reasoning']}")
            return RoutingDecision(
                workflow=WorkflowType.CHAT,
                confidence=chat_check['confidence'],
                reasoning=chat_check['reasoning'],
                ambiguity_level=AmbiguityLevel.LOW,
                rule_match=chat_check.get('rule_name'),
                llm_used=False,
                timestamp=datetime.now().isoformat()
            )

        # Phase 1: Try rule-based classification for main workflows
        rule_result = self._apply_rules(user_input)

        if rule_result and rule_result['confidence'] >= 0.9:
            logger.info(f"[ROUTER] High-confidence rule match: {rule_result['workflow']}")
            return RoutingDecision(
                workflow=WorkflowType(rule_result['workflow']),
                confidence=rule_result['confidence'],
                reasoning=rule_result['reasoning'],
                ambiguity_level=AmbiguityLevel.LOW,
                rule_match=rule_result.get('rule_name'),
                llm_used=False,
                timestamp=datetime.now().isoformat()
            )

        # Phase 2: Use LLM for ambiguous cases
        logger.info("[ROUTER] Using LLM for classification...")
        llm_result = self._classify_with_llm(user_input, context)

        # Phase 3: Merge rule hints with LLM result if needed
        if rule_result and rule_result['confidence'] > 0.5:
            # Boost LLM confidence if rules agree
            if rule_result['workflow'] == llm_result.workflow.value:
                llm_result.confidence = min(llm_result.confidence + 0.1, 1.0)
                llm_result.reasoning += f" (Confirmed by rule: {rule_result['rule_name']})"

        logger.info(f"[ROUTER] Final decision: {llm_result.workflow} (confidence: {llm_result.confidence:.2f})")
        return llm_result

    def _apply_rules(self, user_input: str) -> Optional[Dict[str, Any]]:
        """
        Fast rule-based classification for 3 main workflows
        Returns None or {"workflow": str, "confidence": float, "reasoning": str}
        """
        lowered = user_input.lower()
        matches = []

        # Check for instrument detail patterns (highest priority for project-based)
        project_keywords = ["project", "BOM", "bill of materials", "identify instruments",
                           "instrumentation", "instrument list", "equipment list"]
        context_indicators = ["for", "crude unit", "refinery", "water treatment", "plant",
                            "facility", "distillation", "reactor"]

        has_project_kw = any(kw in lowered for kw in project_keywords)
        has_context = any(ctx in lowered for ctx in context_indicators)

        if has_project_kw and has_context:
            return {
                "workflow": "instrument_detail",
                "confidence": 0.95,
                "reasoning": "Project context and instrument identification keywords",
                "rule_name": "instrument_detail_project_context"
            }
        elif has_project_kw:
            matches.append(("instrument_detail", 0.75, "instrument_detail_keywords"))

        # Check for grounded chat patterns (questions)
        question_keywords = ["what is", "explain", "tell me about", "how does", "why",
                            "difference between", "can you explain", "what are",
                            "define", "meaning of", "help me understand"]

        has_question = any(qkw in lowered for qkw in question_keywords)
        has_question_mark = "?" in user_input

        if has_question or has_question_mark:
            # Strong signal for grounded chat
            if has_question and has_question_mark:
                return {
                    "workflow": "grounded_chat",
                    "confidence": 0.90,
                    "reasoning": "Question keywords and question mark detected",
                    "rule_name": "grounded_chat_question"
                }
            elif has_question:
                matches.append(("grounded_chat", 0.80, "grounded_chat_keywords"))
            elif has_question_mark:
                matches.append(("grounded_chat", 0.70, "grounded_chat_question_mark"))

        # Check for solution patterns (product requirements, procurement, design)
        solution_kws = ["need", "looking for", "find me", "search", "recommend", "suggest",
                       "design", "solution", "best approach", "configure"]
        product_types = ["transmitter", "sensor", "valve", "pump", "meter", "gauge",
                        "switch", "controller", "actuator", "pressure", "flow",
                        "temperature", "level"]

        # NEW: Detect specific product specifications (from sample_input)
        # These indicate the query came from instrument identifier workflow
        spec_indicators = [
            "psi", "bar", "gpm", "lpm", "°f", "°c", "ma", "4-20ma", "hart",
            "range", "accuracy", "±", "output", "input", "connection",
            "npt", "flange", "class", "rating", "material", "stainless",
            "0-", "specifications:"
        ]

        has_solution_kw = any(kw in lowered for kw in solution_kws)
        has_product = any(pt in lowered for pt in product_types)
        has_specs = any(spec in lowered for spec in spec_indicators)

        # ENHANCED: If product + specs (but no project context), it's likely from sample_input → SOLUTION
        if has_product and has_specs and not has_project_kw:
            return {
                "workflow": "solution",
                "confidence": 0.95,
                "reasoning": "Specific product with specifications (likely from instrument selection)",
                "rule_name": "solution_with_specs"
            }

        if has_solution_kw and has_product:
            matches.append(("solution", 0.85, "solution_product_keywords"))
        elif has_product:
            matches.append(("solution", 0.70, "solution_product_only"))
        elif has_solution_kw:
            matches.append(("solution", 0.65, "solution_keywords_only"))

        # Return best match if exists
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            best = matches[0]
            return {
                "workflow": best[0],
                "confidence": best[1],
                "reasoning": f"Rule-based match: {best[2]}",
                "rule_name": best[2]
            }

        return None

    def _classify_with_llm(self, user_input: str, context: Optional[Dict] = None) -> RoutingDecision:
        """
        LLM-based classification for ambiguous cases.

        NOTE: CHAT and INVALID intents are handled by rule-based checks before this method is called.
        This LLM classifier focuses on the 3 main industrial workflows.
        """

        prompt = ChatPromptTemplate.from_template("""
You are Engenie's Workflow Router. Classify this industrial procurement/knowledge request into ONE of 5 workflows.

AVAILABLE WORKFLOWS:

1. SOLUTION - User needs product recommendations, procurement, or design solutions
   Examples:
   - "I need pressure transmitters with 4-20mA output"
   - "Find me HART temperature sensors for steam service"
   - "Design a level measurement system for storage tanks"
   - "Recommend flow meters for crude oil"
   - "Compare Honeywell vs Emerson transmitters"

2. INSTRUMENT_DETAIL - User needs instruments/accessories identified for a project or BOM
   Examples:
   - "Identify instruments for crude oil refinery project"
   - "Generate BOM for water treatment plant"
   - "What instruments are needed for distillation unit"
   - "List equipment for reactor cooling system"

3. GROUNDED_CHAT - User has questions or needs knowledge/explanations about industrial topics
   Examples:
   - "What is the difference between SIL2 and SIL3?"
   - "Explain how HART protocol works"
   - "Tell me about differential pressure transmitters"
   - "Why use RTD over thermocouple?"

4. CHAT - Generic conversational intents (greetings, acknowledgments, simple questions)
   Examples:
   - "Hello" / "Hi" / "Good morning"
   - "Thank you" / "Thanks" / "Got it"
   - "Goodbye" / "Bye" / "See you"
   - "How are you?" / "What's up?"

5. INVALID - Out-of-domain queries (non-industrial topics)
   Examples:
   - "What's the weather today?"
   - "Tell me a joke"
   - "How do I cook pasta?"
   - "Book a flight to Paris"
   - "What's the latest movie?"

User Input: "{user_input}"

Context: {context}

CLASSIFICATION RULES (Priority Order):
1. If out-of-domain (non-industrial topic) → INVALID
2. If greeting/acknowledgment/farewell → CHAT
3. If project/BOM/facility context → INSTRUMENT_DETAIL
4. If question keywords (what, why, how, explain) about industrial topics → GROUNDED_CHAT
5. If product requirements/needs/procurement → SOLUTION
6. Default fallback → SOLUTION

IMPORTANT:
- CHAT is for simple conversational inputs without industrial context
- GROUNDED_CHAT is for technical questions about industrial topics
- INVALID is for queries clearly outside industrial/procurement domain

Return ONLY valid JSON:
{{
    "workflow": "solution" | "instrument_detail" | "grounded_chat" | "chat" | "invalid",
    "confidence": 0.0-1.0,
    "reasoning": "<brief explanation>",
    "ambiguity_level": "low" | "medium" | "high",
    "alternatives": ["<alternative_workflow>"]
}}
""")

        parser = JsonOutputParser()
        chain = prompt | self.llm | parser

        try:
            result = chain.invoke({
                "user_input": user_input,
                "context": context or {}
            })

            # Convert workflow to lowercase to handle LLM returning uppercase
            workflow_value = result["workflow"].lower() if isinstance(result["workflow"], str) else result["workflow"]
            ambiguity_value = result.get("ambiguity_level", "medium").lower() if isinstance(result.get("ambiguity_level"), str) else result.get("ambiguity_level", "medium")

            return RoutingDecision(
                workflow=WorkflowType(workflow_value),
                confidence=result.get("confidence", 0.5),
                reasoning=result.get("reasoning", "LLM classification"),
                ambiguity_level=AmbiguityLevel(ambiguity_value),
                llm_used=True,
                alternatives=[WorkflowType(w.lower() if isinstance(w, str) else w) for w in result.get("alternatives", [])],
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"[ROUTER] LLM classification failed: {e}")
            # Fallback to solution workflow
            return RoutingDecision(
                workflow=WorkflowType.SOLUTION,
                confidence=0.5,
                reasoning=f"LLM failed, defaulting to solution workflow. Error: {str(e)}",
                ambiguity_level=AmbiguityLevel.HIGH,
                llm_used=True,
                timestamp=datetime.now().isoformat()
            )

    def _resolve_conflicts(self, candidates: List[str]) -> str:
        """Resolve conflicts when multiple workflows match"""
        priority = [
            WorkflowType.INSTRUMENT_DETAIL,
            WorkflowType.GROUNDED_CHAT,
            WorkflowType.SOLUTION
        ]

        for workflow in priority:
            if workflow.value in candidates:
                logger.info(f"[ROUTER] Conflict resolved to: {workflow.value}")
                return workflow.value

        return WorkflowType.SOLUTION.value  # Default fallback


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_router_instance = None

def get_workflow_router() -> WorkflowRouter:
    """Get singleton router instance"""
    global _router_instance
    if _router_instance is None:
        _router_instance = WorkflowRouter()
    return _router_instance
