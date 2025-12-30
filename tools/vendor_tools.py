# tools/vendor_tools.py
# Vendor Search and Matching Tools

import json
import logging
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


# ============================================================================
# TOOL INPUT SCHEMAS
# ============================================================================

class SearchVendorsInput(BaseModel):
    """Input for vendor search"""
    product_type: str = Field(description="Product type to search vendors for")
    requirements: Optional[Dict[str, Any]] = Field(default=None, description="User requirements")


class GetVendorProductsInput(BaseModel):
    """Input for getting vendor products"""
    vendor: str = Field(description="Vendor name")
    product_type: str = Field(description="Product type")


class FuzzyMatchVendorsInput(BaseModel):
    """Input for fuzzy vendor matching"""
    vendor_names: List[str] = Field(description="List of vendor names to match")
    allowed_vendors: List[str] = Field(description="List of allowed vendors to match against")
    threshold: int = Field(default=70, description="Matching threshold (0-100)")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_vendors_from_mongodb(product_type: str) -> List[str]:
    """Get vendors from Azure Blob Storage for a product type (MongoDB API compatible)"""
    try:
        from azure_blob_utils import get_vendors_for_product_type
        vendors = get_vendors_for_product_type(product_type)
        return vendors if vendors else []
    except Exception as e:
        logger.warning(f"Failed to get vendors from Azure Blob: {e}")
        return []


def get_all_vendors_from_mongodb() -> List[str]:
    """Get all available vendors from Azure Blob Storage (MongoDB API compatible)"""
    try:
        from azure_blob_utils import get_available_vendors_from_mongodb
        vendors = get_available_vendors_from_mongodb()
        return vendors if vendors else []
    except Exception as e:
        logger.warning(f"Failed to get all vendors: {e}")
        return []


def get_vendor_products_from_mongodb(vendor: str, product_type: str) -> List[Dict[str, Any]]:
    """Get products for a vendor from Azure Blob Storage (MongoDB API compatible)"""
    try:
        from azure_blob_utils import azure_blob_file_manager
        # Query products from Azure Blob
        products = azure_blob_file_manager.list_files(
            'vendors',
            {'vendor_name': vendor, 'product_type': product_type}
        )
        return products
    except Exception as e:
        logger.warning(f"Failed to get vendor products: {e}")
        return []


# ============================================================================
# TOOLS
# ============================================================================

@tool("search_vendors", args_schema=SearchVendorsInput)
def search_vendors_tool(
    product_type: str,
    requirements: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search for vendors that offer products matching the specified type.
    Returns list of vendors with their available product families.
    """
    try:
        # Get vendors for product type
        vendors = get_vendors_from_mongodb(product_type)

        if not vendors:
            # Fall back to all vendors
            vendors = get_all_vendors_from_mongodb()
            logger.info(f"No specific vendors for {product_type}, using all vendors")

        # Get product families for each vendor
        vendor_details = []
        for vendor in vendors:
            vendor_info = {
                "vendor": vendor,
                "product_type": product_type,
                "has_products": True  # Would be checked against DB
            }
            vendor_details.append(vendor_info)

        return {
            "success": True,
            "product_type": product_type,
            "vendors": vendors,
            "vendor_count": len(vendors),
            "vendor_details": vendor_details
        }

    except Exception as e:
        logger.error(f"Vendor search failed: {e}")
        return {
            "success": False,
            "product_type": product_type,
            "vendors": [],
            "error": str(e)
        }


@tool("get_vendor_products", args_schema=GetVendorProductsInput)
def get_vendor_products_tool(vendor: str, product_type: str) -> Dict[str, Any]:
    """
    Get products from a specific vendor for the given product type.
    Returns product list with specifications.
    """
    try:
        products = get_vendor_products_from_mongodb(vendor, product_type)

        return {
            "success": True,
            "vendor": vendor,
            "product_type": product_type,
            "products": products,
            "product_count": len(products)
        }

    except Exception as e:
        logger.error(f"Failed to get vendor products: {e}")
        return {
            "success": False,
            "vendor": vendor,
            "products": [],
            "error": str(e)
        }


@tool("fuzzy_match_vendors", args_schema=FuzzyMatchVendorsInput)
def fuzzy_match_vendors_tool(
    vendor_names: List[str],
    allowed_vendors: List[str],
    threshold: int = 70
) -> Dict[str, Any]:
    """
    Perform fuzzy matching between vendor names and allowed vendors list.
    Used to filter vendors based on CSV uploads or user preferences.
    """
    try:
        from fuzzywuzzy import fuzz, process
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.validation_utils import validate_list, validate_numeric, ValidationException

        matched_vendors = []
        unmatched_vendors = []

        for vendor in vendor_names:
            # Find best match
            result = process.extractOne(vendor, allowed_vendors)

            if result and result[1] >= threshold:
                matched_vendors.append({
                    "original": vendor,
                    "matched_to": result[0],
                    "score": result[1]
                })
            else:
                unmatched_vendors.append({
                    "original": vendor,
                    "best_match": result[0] if result else None,
                    "score": result[1] if result else 0
                })

        return {
            "success": True,
            "matched_vendors": matched_vendors,
            "unmatched_vendors": unmatched_vendors,
            "match_count": len(matched_vendors),
            "threshold_used": threshold
        }

    except Exception as e:
        logger.error(f"Fuzzy matching failed: {e}")
        return {
            "success": False,
            "matched_vendors": [],
            "unmatched_vendors": vendor_names,
            "error": str(e)
        }
