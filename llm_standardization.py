# llm_standardization.py
# Simple LLM-based standardization using existing Gemini models with OpenAI fallback

import json
import logging
from typing import Dict, Any, Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
from llm_fallback import create_llm_with_fallback

# Load environment variables
load_dotenv()

class StandardizedResult(BaseModel):
    """Result of LLM standardization"""
    vendor: str = Field(description="Standardized vendor name", default="")
    product_type: str = Field(description="Standardized product type", default="")
    model_family: Optional[str] = Field(description="Standardized model family", default=None)
    specifications: Dict[str, str] = Field(description="Standardized specification names", default_factory=dict)
    confidence: float = Field(description="Confidence score 0-1", default=1.0)
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_fields
    
    @classmethod
    def validate_fields(cls, v):
        if isinstance(v, dict):
            # Ensure required string fields are not None or empty
            v['vendor'] = v.get('vendor') or ""
            v['product_type'] = v.get('product_type') or ""
        return v

class LLMStandardizer:
    """Dynamic LLM-based standardization using Gemini with OpenAI fallback - fully configurable via prompts"""

    def __init__(self):
        self.llm = create_llm_with_fallback(
            model="gemini-2.5-flash",
            temperature=0.1,
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.parser = JsonOutputParser(pydantic_object=StandardizedResult)
        
        # Import the dynamic standardization prompt
        from prompts import standardization_prompt
        self.prompt = standardization_prompt
        
        self.chain = self.prompt | self.llm | self.parser
        
    def standardize_data(self, raw_data: Dict[str, Any], context: str = "") -> StandardizedResult:
        """
        Standardize raw vendor/product data using LLM
        
        Args:
            raw_data: Dictionary with vendor, product_type, model_family, specifications
            context: Additional context for standardization
            
        Returns:
            StandardizedResult with standardized names
        """
        try:
            result = self.chain.invoke({
                "vendor": raw_data.get("vendor", ""),
                "product_type": raw_data.get("product_type", ""),
                "model_family": raw_data.get("model_family", ""),
                "specifications": json.dumps(raw_data.get("specifications", {}), indent=2),
                "context": context,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            # Handle both dict and StandardizedResult objects
            if isinstance(result, dict):
                # Convert dict to StandardizedResult, ensuring required fields are not None
                return StandardizedResult(
                    vendor=result.get("vendor") or "",
                    product_type=result.get("product_type") or "",
                    model_family=result.get("model_family"),
                    specifications=result.get("specifications", {}),
                    confidence=result.get("confidence", 1.0)
                )
            else:
                # Already a StandardizedResult object, but ensure required fields are not None
                if result.vendor is None:
                    result.vendor = ""
                if result.product_type is None:
                    result.product_type = ""
                return result
        except Exception as e:
            logging.error(f"LLM standardization failed: {e}")
            # Fallback to basic standardization
            return self._fallback_standardize(raw_data)
    
    def standardize(self, raw_data: Dict[str, Any], context: str = "") -> StandardizedResult:
        """Backward compatibility method - calls standardize_data"""
        return self.standardize_data(raw_data, context)
    
    def _fallback_standardize(self, raw_data: Dict[str, Any]) -> StandardizedResult:
        """LLM-only fallback standardization - no hardcoded mappings"""
        vendor = raw_data.get("vendor", "").strip()
        product_type = raw_data.get("product_type", "").strip()
        
        # If LLM fails, try a simpler prompt structure
        try:
            simple_prompt = ChatPromptTemplate.from_template("""
            Standardize these industrial product names using proper industry terminology:
            
            Vendor: {vendor}
            Product Type: {product_type}
            
            Return standardized names in JSON format:
            {{
                "vendor": "[standardized vendor name]",
                "product_type": "[standardized product type]"
            }}
            """)
            
            simple_chain = simple_prompt | self.llm | JsonOutputParser()
            
            result = simple_chain.invoke({
                "vendor": vendor,
                "product_type": product_type
            })
            
            return StandardizedResult(
                vendor=result.get("vendor") or vendor or "",
                product_type=result.get("product_type") or product_type or "",
                model_family=raw_data.get("model_family") or "",
                specifications=raw_data.get("specifications", {}),
                confidence=0.5  # Lower confidence for simple fallback
            )
            
        except Exception as e:
            logging.warning(f"Simple LLM fallback also failed: {e}")
            # Last resort - return original data cleaned up
            return StandardizedResult(
                vendor=vendor or "",
                product_type=product_type or "",
                model_family=raw_data.get("model_family") or "",
                specifications=raw_data.get("specifications", {}),
                confidence=0.1  # Very low confidence for no standardization
            )

# Global standardizer instance
_llm_standardizer = None

def get_llm_standardizer() -> LLMStandardizer:
    """Get global LLM standardizer instance"""
    global _llm_standardizer
    if _llm_standardizer is None:
        _llm_standardizer = LLMStandardizer()
    return _llm_standardizer

def standardize_with_llm(data: Dict[str, Any], context: str = "") -> Dict[str, Any]:
    """
    Standardize data using LLM
    
    Args:
        data: Raw data to standardize
        context: Additional context
        
    Returns:
        Standardized data dictionary
    """
    standardizer = get_llm_standardizer()
    result = standardizer.standardize(data, context)
    
    # Handle both Pydantic model and dictionary results
    if hasattr(result, 'vendor'):
        # Pydantic model result
        return {
            "vendor": result.vendor,
            "product_type": result.product_type, 
            "model_family": result.model_family,
            "specifications": result.specifications,
            "confidence": result.confidence,
            "method": "llm"
        }
    else:
        # Dictionary result (from fallback)
        return {
            "vendor": result.get("vendor", ""),
            "product_type": result.get("product_type", ""), 
            "model_family": result.get("model_family", ""),
            "specifications": result.get("specifications", {}),
            "confidence": result.get("confidence", 0.7),
            "method": "fallback"
        }

# Integration functions for existing code
def standardize_vendor_analysis_result(analysis_result: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize vendor analysis result using LLM"""
    if not analysis_result:
        return analysis_result
        
    # Extract data for standardization
    raw_data = {
        "vendor": analysis_result.get("vendor", ""),
        "product_type": analysis_result.get("product_type", ""),
        "model_family": analysis_result.get("model_family", ""),
        "specifications": analysis_result.get("specifications", {})
    }
    
    # Get context from analysis
    context = f"Product analysis with {len(analysis_result.get('matched_models', []))} models"
    
    # Standardize
    standardized = standardize_with_llm(raw_data, context)
    
    # Update analysis result
    result = analysis_result.copy()
    result.update({
        "vendor": standardized["vendor"],
        "product_type": standardized["product_type"],
        "model_family": standardized["model_family"],
        "standardization_confidence": standardized["confidence"],
        "standardization_method": "llm"
    })
    
    return result

def standardize_ranking_result(ranking_result: Dict[str, Any]) -> Dict[str, Any]:
    """Standardize ranking result using LLM"""
    if not ranking_result:
        return ranking_result
        
    result = ranking_result.copy()
    
    # Standardize each ranked vendor
    if "ranked_vendors" in result:
        for vendor_data in result["ranked_vendors"]:
            raw_data = {
                "vendor": vendor_data.get("vendor", ""),
                "product_type": vendor_data.get("product_type", ""),
                "model_family": vendor_data.get("model_family", ""),
                "specifications": vendor_data.get("specifications", {})
            }
            
            standardized = standardize_with_llm(raw_data)
            vendor_data.update({
                "vendor": standardized["vendor"],
                "product_type": standardized["product_type"], 
                "model_family": standardized["model_family"],
                "standardization_confidence": standardized["confidence"]
            })
    
    return result
