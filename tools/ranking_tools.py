# tools/ranking_tools.py
# Product Ranking and Judging Tools

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class RankProductsInput(BaseModel):
    """Input for product ranking"""
    vendor_matches: List[Dict[str, Any]] = Field(description="List of vendor match results")
    requirements: Dict[str, Any] = Field(description="Original user requirements")


class JudgeAnalysisInput(BaseModel):
    """Input for analysis judging"""
    original_requirements: Dict[str, Any] = Field(description="Original user requirements")
    vendor_analysis: Dict[str, Any] = Field(description="Vendor analysis results")
    strategy_rules: Optional[Dict[str, Any]] = Field(default=None, description="Procurement strategy rules")


# ============================================================================
# PROMPTS
# ============================================================================

RANKING_PROMPT = """
You are Engenie - a product ranking specialist for industrial requisitioners and buyers.

**IMPORTANT: Think step-by-step through your ranking process.**

Before creating the final ranking:
1. First, review all vendor analysis results and identify common patterns
2. Then, extract ALL mandatory and optional parameter matches for each product
3. Identify any limitations or concerns mentioned in the vendor analysis
4. Calculate comparative scores based on requirement fulfillment
5. Finally, rank products from best to worst match

**CRITICAL: You must extract and preserve ALL information from the vendor analysis, especially:**
1. **Mandatory Parameters Analysis** - Convert these to Key Strengths
2. **Optional Parameters Analysis** - Convert these to Key Strengths or Concerns based on match
3. **Comprehensive Analysis & Assessment** - Extract both Reasoning and Key Limitations
4. **Any unmatched requirements** - These become Concerns

**Original Requirements:**
{requirements}

**Vendor Analysis Results:**
{vendor_matches}

---

**Ranking Criteria:**
1. Match Score (40%): How well specifications match requirements
2. Critical Requirements (30%): All mandatory requirements met
3. Optional Features (15%): Additional beneficial features
4. Vendor Reliability (15%): Based on available data

For each product, provide:

**Key Strengths:**
For each parameter that matches requirements:
- **[Friendly Parameter Name] (User Requirement)** - Product provides "[Product Specification]" - [Holistic explanation paragraph: why it matches, justification from datasheet/JSON, impact on overall suitability, interactions with other parameters].

**Concerns:**
For each parameter that does not match:
- Holistic explanation paragraph: why it does not meet requirement, limitation, potential impact, interactions with other parameters.

**Guidelines:**
- **MANDATORY**: Extract and include ALL limitations mentioned in the vendor analysis "Key Limitations" section
- Include EVERY parameter from the user requirements in either strengths or concerns
- For each parameter, show: Parameter name, User requirement, Product specification, Detailed holistic explanation
- Explain the technical and business impact of each match or mismatch
- Each explanation should be 1-2 sentences that clearly show why it's a strength or concern
- Base explanations on actual specifications from the vendor analysis
- If a parameter wasn't analyzed, note it as "Not specified in available documentation"
- **Always preserve limitations from vendor analysis** - these are critical for buyer decision-making

**CRITICAL - Limitation Extraction Verification:**
Before finalizing your response, verify:
1. ✓ Have I extracted EVERY limitation from the vendor analysis?
2. ✓ Are all limitations included in the concerns section?
3. ✓ Did I check the "Key Limitations" or "Comprehensive Analysis & Assessment" sections?
4. ✓ Are there any unmatched requirements that should be concerns?
5. ✓ Have I explained WHY each limitation matters?

If you answer NO to any question, review and add the missing limitations.

Return ONLY valid JSON:
{{
    "ranked_products": [
        {{
            "rank": 1,
            "vendor": "<vendor name>",
            "product_name": "<product model>",
            "model_family": "<model series>",
            "overall_score": <0-100>,
            "key_strengths": ["<strength 1 with parameter details>", "<strength 2>"],
            "concerns": ["<concern 1 with limitation details>", "<concern 2>"],
            "recommendation": "<recommendation text>"
        }}
    ],
    "ranking_summary": "<overall ranking summary>",
    "top_pick": {{
        "vendor": "<vendor>",
        "product_name": "<product>",
        "reason": "<why this is the top pick>"
    }}
}}
"""

JUDGE_PROMPT = """
You are Engenie - the Final Review Judge for industrial procurement.
Validate the vendor analysis results against original requirements and strategy.

**Original Requirements:**
{original_requirements}

**Vendor Analysis:**
{vendor_analysis}

**Strategy Rules:**
{strategy_rules}

**Validation Checks:**
1. Consistency: Do analysis results match the requirements?
2. Accuracy: Are specifications correctly matched?
3. Completeness: Are all requirements addressed?
4. Strategy Compliance: Do results follow procurement strategy?
5. Error Detection: Any contradictions or issues?

Return ONLY valid JSON:
{{
    "is_valid": <true|false>,
    "validation_score": <0-100>,
    "issues": [
        {{
            "type": "error|warning|info",
            "field": "<affected field>",
            "description": "<issue description>",
            "suggested_fix": "<how to fix>"
        }}
    ],
    "approved_vendors": ["<list of approved vendors>"],
    "rejected_vendors": [
        {{
            "vendor": "<vendor name>",
            "reason": "<rejection reason>"
        }}
    ],
    "validation_summary": "<summary of validation>"
}}
"""


# ============================================================================
# TOOLS
# ============================================================================

@tool("rank_products", args_schema=RankProductsInput)
def rank_products_tool(
    vendor_matches: List[Dict[str, Any]],
    requirements: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Rank products based on vendor analysis results.
    Returns ordered list with scores and recommendations.
    """
    try:
        if not vendor_matches:
            return {
                "success": False,
                "ranked_products": [],
                "error": "No vendor matches to rank"
            }

        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(RANKING_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "requirements": json.dumps(requirements, indent=2),
            "vendor_matches": json.dumps(vendor_matches, indent=2)
        })

        return {
            "success": True,
            "ranked_products": result.get("ranked_products", []),
            "ranking_summary": result.get("ranking_summary"),
            "top_pick": result.get("top_pick"),
            "total_ranked": len(result.get("ranked_products", []))
        }

    except Exception as e:
        logger.error(f"Product ranking failed: {e}")
        return {
            "success": False,
            "ranked_products": [],
            "error": str(e)
        }


@tool("judge_analysis", args_schema=JudgeAnalysisInput)
def judge_analysis_tool(
    original_requirements: Dict[str, Any],
    vendor_analysis: Dict[str, Any],
    strategy_rules: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Validate and judge the vendor analysis results.
    Checks for consistency, accuracy, and strategy compliance.
    """
    try:
        llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )

        prompt = ChatPromptTemplate.from_template(JUDGE_PROMPT)
        parser = JsonOutputParser()

        chain = prompt | llm | parser

        result = chain.invoke({
            "original_requirements": json.dumps(original_requirements, indent=2),
            "vendor_analysis": json.dumps(vendor_analysis, indent=2),
            "strategy_rules": json.dumps(strategy_rules, indent=2) if strategy_rules else "No specific strategy rules"
        })

        return {
            "success": True,
            "is_valid": result.get("is_valid", False),
            "validation_score": result.get("validation_score", 0),
            "issues": result.get("issues", []),
            "approved_vendors": result.get("approved_vendors", []),
            "rejected_vendors": result.get("rejected_vendors", []),
            "validation_summary": result.get("validation_summary")
        }

    except Exception as e:
        logger.error(f"Analysis judging failed: {e}")
        return {
            "success": False,
            "is_valid": False,
            "issues": [{"type": "error", "description": str(e)}],
            "error": str(e)
        }
