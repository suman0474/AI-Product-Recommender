# tools/rag_tools.py
# LangChain Tools for RAG Document Management

import logging
from typing import Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

# Import the vector store (Pinecone primary, ChromaDB HTTP secondary)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agentic.vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ============================================================================
# INPUT SCHEMAS
# ============================================================================

class AddDocumentInput(BaseModel):
    """Input schema for adding a document to the vector store."""
    collection_type: str = Field(
        description="Type of collection: 'strategy', 'standards', 'inventory', or 'general'"
    )
    content: str = Field(
        description="The document text content to store"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata dictionary (e.g., source, author, type)"
    )
    doc_id: Optional[str] = Field(
        default=None,
        description="Optional document ID (auto-generated if not provided)"
    )


class SearchDocumentsInput(BaseModel):
    """Input schema for searching documents."""
    collection_type: str = Field(
        description="Type of collection to search: 'strategy', 'standards', 'inventory', 'general', or 'all'"
    )
    query: str = Field(
        description="Search query text for semantic search"
    )
    top_k: int = Field(
        default=5,
        description="Number of results to return (default: 5)"
    )


class ListDocumentsInput(BaseModel):
    """Input schema for listing documents."""
    collection_type: str = Field(
        description="Type of collection: 'strategy', 'standards', 'inventory', or 'general'"
    )
    limit: int = Field(
        default=50,
        description="Maximum number of documents to return"
    )


class DeleteDocumentInput(BaseModel):
    """Input schema for deleting a document."""
    collection_type: str = Field(
        description="Type of collection: 'strategy', 'standards', 'inventory', or 'general'"
    )
    doc_id: str = Field(
        description="Document ID to delete"
    )


# ============================================================================
# LANGCHAIN TOOLS
# ============================================================================

class AddDocumentTool(BaseTool):
    """Tool for adding documents to the ChromaDB vector store."""
    
    name: str = "add_document"
    description: str = """Add a document to the RAG vector store for later retrieval.
    
    Use this tool to store:
    - Strategy documents (vendor policies, procurement rules)
    - Standards documents (SIL ratings, ATEX requirements, certifications)
    - Inventory documents (installed base info, spare parts)
    - General knowledge documents
    
    The document will be automatically chunked and embedded for semantic search.
    """
    args_schema: Type[BaseModel] = AddDocumentInput
    
    def _run(
        self, 
        collection_type: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a document to the vector store."""
        try:
            store = get_vector_store()
            result = store.add_document(
                collection_type=collection_type,
                content=content,
                metadata=metadata,
                doc_id=doc_id
            )
            logger.info(f"AddDocumentTool: Added document to {collection_type}")
            return result
        except Exception as e:
            logger.error(f"AddDocumentTool failed: {e}")
            return {"success": False, "error": str(e)}


class SearchDocumentsTool(BaseTool):
    """Tool for searching documents in the ChromaDB vector store."""
    
    name: str = "search_documents"
    description: str = """Search for documents semantically in the RAG vector store.
    
    Use this tool to find:
    - Relevant vendor policies and procurement strategies
    - Certification and compliance requirements
    - Installed base and inventory constraints
    - Any stored knowledge documents
    
    Returns ranked results by relevance to the query.
    """
    args_schema: Type[BaseModel] = SearchDocumentsInput
    
    def _run(
        self, 
        collection_type: str, 
        query: str, 
        top_k: int = 5
    ) -> Dict[str, Any]:
        """Search for documents matching the query."""
        try:
            store = get_vector_store()
            
            if collection_type == "all":
                result = store.search_all_collections(query, top_k_per_collection=top_k)
            else:
                result = store.search(collection_type, query, top_k)
            
            logger.info(f"SearchDocumentsTool: Searched {collection_type} for '{query[:50]}...'")
            return result
        except Exception as e:
            logger.error(f"SearchDocumentsTool failed: {e}")
            return {"success": False, "error": str(e), "results": []}


class ListDocumentsTool(BaseTool):
    """Tool for listing documents in the ChromaDB vector store."""
    
    name: str = "list_documents"
    description: str = """List all documents stored in a specific collection.
    
    Use this tool to see what documents are available in:
    - strategy: Vendor policies, procurement strategies
    - standards: Certification requirements, compliance docs
    - inventory: Installed base, spare parts info
    - general: General knowledge documents
    """
    args_schema: Type[BaseModel] = ListDocumentsInput
    
    def _run(
        self, 
        collection_type: str, 
        limit: int = 50
    ) -> Dict[str, Any]:
        """List documents in the collection."""
        try:
            store = get_vector_store()
            result = store.list_documents(collection_type, limit)
            logger.info(f"ListDocumentsTool: Listed {result.get('document_count', 0)} docs from {collection_type}")
            return result
        except Exception as e:
            logger.error(f"ListDocumentsTool failed: {e}")
            return {"success": False, "error": str(e), "documents": []}


class DeleteDocumentTool(BaseTool):
    """Tool for deleting documents from the ChromaDB vector store."""
    
    name: str = "delete_document"
    description: str = """Delete a document from the RAG vector store.
    
    Use this tool to remove outdated or incorrect documents.
    Requires the document ID and collection type.
    """
    args_schema: Type[BaseModel] = DeleteDocumentInput
    
    def _run(
        self, 
        collection_type: str, 
        doc_id: str
    ) -> Dict[str, Any]:
        """Delete a document from the collection."""
        try:
            store = get_vector_store()
            result = store.delete_document(collection_type, doc_id)
            logger.info(f"DeleteDocumentTool: Deleted {doc_id} from {collection_type}")
            return result
        except Exception as e:
            logger.error(f"DeleteDocumentTool failed: {e}")
            return {"success": False, "error": str(e)}


class GetCollectionStatsTool(BaseTool):
    """Tool for getting statistics about the vector store collections."""
    
    name: str = "get_collection_stats"
    description: str = """Get statistics about all document collections in the vector store.
    
    Returns information about:
    - Number of document chunks per collection
    - Collection status (active/empty)
    - Persist directory location
    """
    
    def _run(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            store = get_vector_store()
            result = store.get_collection_stats()
            logger.info("GetCollectionStatsTool: Retrieved collection stats")
            return result
        except Exception as e:
            logger.error(f"GetCollectionStatsTool failed: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# TOOL FACTORY
# ============================================================================

def get_rag_tools():
    """Get all RAG document management tools."""
    return [
        AddDocumentTool(),
        SearchDocumentsTool(),
        ListDocumentsTool(),
        DeleteDocumentTool(),
        GetCollectionStatsTool()
    ]


def get_rag_search_tools():
    """Get only the search tool (for read-only access)."""
    return [
        SearchDocumentsTool(),
        ListDocumentsTool(),
        GetCollectionStatsTool()
    ]
