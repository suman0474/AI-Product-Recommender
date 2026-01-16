"""
Product Metrics Generator

Extracts comprehensive metrics from workflow results (Solution & Product Search)
and formats them for storage and grounded chat retrieval.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ProductMetricsGenerator:
    """
    Generates comprehensive metrics from workflow results.

    Extracts:
    - Product specifications (pressure range, accuracy, material, certifications)
    - Pricing and cost analysis
    - Ranking scores and reasoning
    - Vendor and availability information
    """

    @staticmethod
    def generate_from_solution_workflow(state: Dict[str, Any], expiration_days: int = 7) -> Dict[str, Any]:
        """
        Generate metrics from Solution Workflow results.

        Args:
            state: SolutionState dictionary
            expiration_days: Number of days before metrics expire

        Returns:
            Comprehensive metrics JSON
        """
        logger.info("[METRICS] Generating metrics from Solution Workflow...")

        try:
            timestamp = datetime.utcnow()
            expires_at = timestamp + timedelta(days=expiration_days)

            # Extract ranked results
            ranked_results = state.get("ranked_results", [])
            parallel_analysis = state.get("parallel_analysis_results", [])
            response_data = state.get("response_data", {})

            # Build product metrics
            products = []
            total_cost = 0.0

            for i, product in enumerate(ranked_results[:10], 1):  # Top 10 products
                # Extract vendor match data
                vendor_match = product.get("vendor_match", {})
                analysis = product.get("analysis", {})

                # Extract specifications
                specifications = ProductMetricsGenerator._extract_specifications(vendor_match)

                # Extract pricing
                pricing_info = ProductMetricsGenerator._extract_pricing(vendor_match)
                if pricing_info.get("unit_price"):
                    total_cost += pricing_info["unit_price"]

                # Extract ranking information
                ranking_info = ProductMetricsGenerator._extract_ranking(product, i)

                # Extract vendor information
                vendor_info = ProductMetricsGenerator._extract_vendor_info(vendor_match)

                product_metric = {
                    "rank": i,
                    "name": vendor_match.get("model", "Unknown"),
                    "vendor": vendor_match.get("vendor_name", "Unknown"),
                    "series": vendor_match.get("series", ""),
                    "product_type": state.get("product_type", ""),
                    "specifications": specifications,
                    "pricing": pricing_info,
                    "ranking": ranking_info,
                    "vendor_info": vendor_info,
                    "analysis": {
                        "match_score": analysis.get("match_score", 0.0),
                        "strengths": analysis.get("strengths", []),
                        "weaknesses": analysis.get("weaknesses", []),
                        "recommendation": analysis.get("recommendation", "")
                    }
                }

                products.append(product_metric)

            # Generate summary metrics
            summary_metrics = {
                "total_products": len(products),
                "total_cost_estimate": round(total_cost, 2),
                "avg_ranking_score": round(sum(p["ranking"]["overall_score"] for p in products) / len(products), 3) if products else 0.0,
                "top_vendor": products[0]["vendor"] if products else None,
                "product_type": state.get("product_type", ""),
                "user_requirements": state.get("provided_requirements", {}),
                "workflow_intent": state.get("intent", "requirements")
            }

            # Build searchable text for RAG
            searchable_text = ProductMetricsGenerator._build_searchable_text_solution(
                products, summary_metrics, state
            )

            metrics = {
                "session_id": state.get("session_id", ""),
                "workflow_type": "solution",
                "timestamp": timestamp.isoformat(),
                "expires_at": expires_at.isoformat(),
                "products": products,
                "summary_metrics": summary_metrics,
                "searchable_text": searchable_text,
                "metadata": {
                    "user_input": state.get("user_input", ""),
                    "intent": state.get("intent", ""),
                    "comparison_mode": state.get("comparison_mode", False)
                }
            }

            logger.info(f"[METRICS] Generated metrics for {len(products)} products (Solution Workflow)")
            return metrics

        except Exception as e:
            logger.error(f"[METRICS] Failed to generate solution workflow metrics: {e}")
            return {}



    # ========================================================================
    # HELPER METHODS - Extraction
    # ========================================================================

    @staticmethod
    def _extract_specifications(vendor_match: Dict[str, Any]) -> Dict[str, Any]:
        """Extract product specifications from vendor match."""
        specs = {}

        # Common specification fields
        spec_fields = [
            "pressure_range", "accuracy", "material", "certifications",
            "temperature_range", "output_signal", "process_connection",
            "electrical_connection", "power_supply", "enclosure_rating",
            "hazardous_area_approval", "communication_protocol"
        ]

        for field in spec_fields:
            if field in vendor_match:
                specs[field] = vendor_match[field]

        # Also check in nested 'specifications' dict
        if "specifications" in vendor_match:
            vendor_specs = vendor_match["specifications"]
            if isinstance(vendor_specs, dict):
                specs.update(vendor_specs)

        return specs

    @staticmethod
    def _extract_pricing(vendor_match: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pricing information from vendor match."""
        pricing = {
            "unit_price": None,
            "currency": "USD",
            "availability": "Check with vendor",
            "lead_time": None
        }

        # Extract price
        if "price" in vendor_match:
            try:
                pricing["unit_price"] = float(vendor_match["price"])
            except (ValueError, TypeError):
                pass

        # Extract availability
        if "availability" in vendor_match:
            pricing["availability"] = vendor_match["availability"]

        if "lead_time" in vendor_match:
            pricing["lead_time"] = vendor_match["lead_time"]

        if "currency" in vendor_match:
            pricing["currency"] = vendor_match["currency"]

        return pricing

    @staticmethod
    def _extract_ranking(product: Dict[str, Any], rank: int) -> Dict[str, Any]:
        """Extract ranking information from product."""
        ranking = {
            "overall_score": 0.0,
            "match_percentage": 0.0,
            "reasoning": "",
            "rank": rank
        }

        # Extract score
        if "overall_score" in product:
            ranking["overall_score"] = product["overall_score"]
        elif "score" in product:
            ranking["overall_score"] = product["score"]

        # Extract match percentage
        if "match_percentage" in product:
            ranking["match_percentage"] = product["match_percentage"]

        # Extract reasoning
        analysis = product.get("analysis", {})
        if "recommendation" in analysis:
            ranking["reasoning"] = analysis["recommendation"]
        elif "reasoning" in product:
            ranking["reasoning"] = product["reasoning"]

        return ranking

    @staticmethod
    def _extract_vendor_info(vendor_match: Dict[str, Any]) -> Dict[str, Any]:
        """Extract vendor information from vendor match."""
        vendor_info = {
            "vendor_name": vendor_match.get("vendor_name", "Unknown"),
            "vendor_type": vendor_match.get("vendor_type", ""),
            "contact_info": vendor_match.get("contact_info", {}),
            "website": vendor_match.get("website", ""),
            "support_level": vendor_match.get("support_level", "")
        }

        return vendor_info



    # ========================================================================
    # HELPER METHODS - Searchable Text
    # ========================================================================

    @staticmethod
    def _build_searchable_text_solution(products: List[Dict], summary: Dict, state: Dict) -> str:
        """Build searchable text for solution workflow results."""
        lines = []

        # Header
        lines.append("=== PRODUCT RECOMMENDATION RESULTS ===")
        lines.append(f"Product Type: {summary.get('product_type', 'N/A')}")
        lines.append(f"User Query: {state.get('user_input', '')}")
        lines.append(f"Total Products Found: {summary['total_products']}")
        lines.append(f"Estimated Total Cost: ${summary['total_cost_estimate']}")
        lines.append("")

        # Product details
        for product in products:
            lines.append(f"--- Product #{product['rank']}: {product['name']} ({product['vendor']}) ---")
            lines.append(f"Series: {product['series']}")
            lines.append(f"Overall Score: {product['ranking']['overall_score']:.2f}")

            # Specifications
            specs = product.get("specifications", {})
            if specs:
                lines.append("Specifications:")
                for key, value in specs.items():
                    lines.append(f"  - {key.replace('_', ' ').title()}: {value}")

            # Pricing
            pricing = product.get("pricing", {})
            if pricing.get("unit_price"):
                lines.append(f"Price: ${pricing['unit_price']} {pricing.get('currency', 'USD')}")

            # Ranking reasoning
            if product['ranking'].get('reasoning'):
                lines.append(f"Recommendation: {product['ranking']['reasoning']}")

            # Analysis
            analysis = product.get("analysis", {})
            if analysis.get("strengths"):
                lines.append(f"Strengths: {', '.join(analysis['strengths'][:3])}")
            if analysis.get("weaknesses"):
                lines.append(f"Weaknesses: {', '.join(analysis['weaknesses'][:3])}")

            lines.append("")

        # Summary
        lines.append("=== SUMMARY ===")
        lines.append(f"Top Recommended Vendor: {summary.get('top_vendor', 'N/A')}")
        lines.append(f"Average Ranking Score: {summary['avg_ranking_score']:.2f}")

        return "\n".join(lines)




# Convenience functions
def generate_solution_metrics(state: Dict[str, Any], expiration_days: int = 7) -> Dict[str, Any]:
    """Generate metrics from Solution Workflow state."""
    return ProductMetricsGenerator.generate_from_solution_workflow(state, expiration_days)





__all__ = [
    'ProductMetricsGenerator',
    'generate_solution_metrics',

]
