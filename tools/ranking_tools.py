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
from prompts_library import load_prompt

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
# PROMPTS - Loaded from prompts_library
# ============================================================================

RANKING_PROMPT = load_prompt("ranking_tool_prompt")

JUDGE_PROMPT = load_prompt("judge_prompt")


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
