"""
Internal API Client for Workflow Orchestration

This module provides a clean interface for internal calls to workflow API endpoints.
Instead of directly invoking workflow functions (e.g., run_solution_workflow()),
orchestration code must use this API client to ensure complete decoupling.

Architecture Principle:
- All workflow execution MUST go through API endpoints
- Router and workflow chainers use this client, not direct function calls
- Ensures consistent middleware (auth, logging, error handling)
- Enables easy microservice extraction

Usage:
    from .internal_api import api_client

    result = api_client.call_solution_workflow(
        message="User requirements",
        session_id="session-123"
    )
"""

import logging
from typing import Dict, Any, Optional
from flask import current_app, session as flask_session

logger = logging.getLogger(__name__)


class InternalAPIClient:
    """
    Client for making internal API calls to workflow endpoints.

    Uses Flask test client to make in-process HTTP-like calls without
    network overhead, while preserving session context and authentication.
    """

    @staticmethod
    def _call(endpoint: str, data: Dict[str, Any], method: str = 'POST') -> Dict[str, Any]:
        """
        Make an internal API call with session preservation.

        Args:
            endpoint: API endpoint path (e.g., '/api/agentic/solution')
            data: JSON payload to send
            method: HTTP method (default: POST)

        Returns:
            Dictionary with workflow result data

        Raises:
            Exception: If API call fails or returns error
        """
        try:
            # Get Flask test client for in-process calls
            client = current_app.test_client()

            # Preserve session context for authentication
            with client.session_transaction() as sess:
                for key, value in flask_session.items():
                    sess[key] = value

            # Make the internal API call
            logger.debug(f"Internal API call: {method} {endpoint}")
            response = client.post(endpoint, json=data) if method == 'POST' else client.get(endpoint, json=data)

            # Parse response
            response_data = response.get_json()

            if not response_data:
                error_msg = f"Empty response from {endpoint}"
                logger.error(error_msg)
                raise Exception(error_msg)

            # Check for errors in response
            if not response_data.get('success', True):
                error = response_data.get('error', 'Unknown error')
                logger.error(f"API call to {endpoint} failed: {error}")
                raise Exception(f"Workflow API error: {error}")

            # Return the data portion
            return response_data.get('data', response_data)

        except Exception as e:
            logger.exception(f"Internal API call to {endpoint} failed: {str(e)}")
            raise

    # ========================================================================
    # WORKFLOW-SPECIFIC METHODS
    # ========================================================================

    @staticmethod
    def call_solution_workflow(message: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        Call the Solution Workflow API.

        Handles product requirements, procurement requests, and recommendations.

        Args:
            message: User input message
            session_id: Session identifier
            **kwargs: Additional workflow parameters

        Returns:
            Solution workflow result with ranked products
        """
        data = {
            "message": message,
            "session_id": session_id,
            **kwargs
        }
        return InternalAPIClient._call('/api/agentic/solution', data)

# Comparative analysis is now handled via call_product_search with auto_compare=True

    @staticmethod
    def call_comparison_from_spec(spec_object: Dict[str, Any], session_id: str,
                                   comparison_type: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Call the Comparison Workflow API with SpecObject input.

        Used when chaining from instrument detail workflow.

        Args:
            spec_object: Structured specification object
            session_id: Session identifier
            comparison_type: Type of comparison (vendor/series/model)
            **kwargs: Additional workflow parameters

        Returns:
            Comparison workflow result
        """
        data = {
            "spec_object": spec_object,
            "session_id": session_id,
            **kwargs
        }
        if comparison_type:
            data["comparison_type"] = comparison_type

        return InternalAPIClient._call('/api/agentic/compare-from-spec', data)

    @staticmethod
    def call_instrument_detail(message: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        Call the Instrument Detail Workflow API.

        Captures detailed specifications for project BOM.

        Args:
            message: User input with project/instrument details
            session_id: Session identifier
            **kwargs: Additional workflow parameters

        Returns:
            Instrument detail result with spec_object
        """
        data = {
            "message": message,
            "session_id": session_id,
            **kwargs
        }
        return InternalAPIClient._call('/api/agentic/instrument-detail', data)

    @staticmethod
    def call_product_info(query: str, session_id: str, rag_type: str = None,
                          validate: bool = True, entities: list = None, **kwargs) -> Dict[str, Any]:
        """
        Call the Product Info Query API.

        Unified endpoint for knowledge queries with intelligent routing to:
        Index RAG, Standards RAG, Strategy RAG, Deep Agent, or Web Search.

        Args:
            query: User question
            session_id: Session identifier
            rag_type: Optional routing hint (index_rag, standards_rag, strategy_rag,
                     deep_agent, web_search, hybrid)
            validate: Whether to validate response (default: True)
            entities: Optional entity list for verification
            **kwargs: Additional workflow parameters

        Returns:
            Product Info result with answer, sources, validation metadata
        """
        data = {
            "query": query,
            "session_id": session_id,
            "rag_type": rag_type,
            "validate": validate,
            "entities": entities,
            **kwargs
        }
        return InternalAPIClient._call('/api/product-info/query', data)

    @staticmethod
    def call_instrument_identifier(message: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        Call the Instrument Identifier Workflow API.

        Identifies instruments/accessories from process descriptions.

        Args:
            message: User input with process description
            session_id: Session identifier
            **kwargs: Additional workflow parameters

        Returns:
            Instrument identifier result with identified items
        """
        data = {
            "message": message,
            "session_id": session_id,
            **kwargs
        }
        return InternalAPIClient._call('/api/agentic/instrument-identifier', data)

    @staticmethod
    def call_potential_product_index(product_type: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        Call the Potential Product Index (PPI) Workflow API.

        Discovers new products and generates schemas.

        Args:
            product_type: Product type to index
            session_id: Session identifier
            **kwargs: Additional workflow parameters

        Returns:
            PPI workflow result with schema
        """
        data = {
            "product_type": product_type,
            "session_id": session_id,
            **kwargs
        }
        return InternalAPIClient._call('/api/agentic/potential-product-index', data)

    def call_legacy_workflow(self, message: str, session_id: str, workflow_type: str = "procurement", **kwargs) -> Dict[str, Any]:
        """
        Call the Legacy Workflow API.

        Supports original procurement and instrument identification workflows.

        Args:
            message: User input
            session_id: Session identifier
            workflow_type: "procurement" or "instrument_identification"
            **kwargs: Additional workflow parameters

        Returns:
            Legacy workflow result
        """
        payload = {
            "message": message,
            "session_id": session_id,
            "workflow_type": workflow_type
        }
        payload.update(kwargs)

        return self._call('/api/agentic/legacy-workflow', payload)

    def call_product_search(self, message: str, session_id: str, **kwargs) -> Dict[str, Any]:
        """
        Call the Product Search Workflow API.

        Executed after instrument identification to find specific products.

        Args:
            message: User input (sample_input)
            session_id: Session identifier
            **kwargs: Additional workflow parameters

        Returns:
            Product search result with ranked products
        """
        payload = {
            "message": message,
            "session_id": session_id
        }
        payload.update(kwargs)

        return self._call('/api/agentic/product-search', payload)


# Create singleton instance
api_client = InternalAPIClient()

__all__ = ['api_client', 'InternalAPIClient']
