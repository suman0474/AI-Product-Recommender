# agentic/vector_store.py
# Vector Store Implementation - Pinecone Primary, ChromaDB HTTP Secondary
# FAISS has been completely removed
#
# Available backends:
# 1. Pinecone - Primary (managed cloud service, works with Python 3.14)
# 2. ChromaDB HTTP - Secondary (requires Docker for ChromaDB server)
#
# The chromadb Python package has compatibility issues with Python 3.14,
# so we use either Pinecone or the HTTP-based ChromaDB client.

import os
import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Environment configuration
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "pinecone").lower()

# ChromaDB settings (for HTTP mode with Docker)
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))

# Pinecone Settings
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", None)
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")

# Collection names for different RAG types
COLLECTIONS = {
    "strategy": "strategy_documents",
    "standards": "standards_documents",
    "inventory": "inventory_documents",
    "general": "general_documents"
}

# Text splitting configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ============================================================================
# BASE DOCUMENT STORE (Abstract)
# ============================================================================

class BaseDocumentStore:
    """Base class for all document stores."""
    
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def add_document(self, collection_type: str, content: str, 
                     metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Dict:
        raise NotImplementedError
    
    def search(self, collection_type: str, query: str, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> Dict:
        raise NotImplementedError
    
    def list_documents(self, collection_type: str, limit: int = 100) -> Dict:
        raise NotImplementedError
    
    def delete_document(self, collection_type: str, doc_id: str) -> Dict:
        raise NotImplementedError
    
    def get_collection_stats(self) -> Dict:
        raise NotImplementedError
    
    def clear_collection(self, collection_type: str) -> Dict:
        raise NotImplementedError


# ============================================================================
# PINECONE DOCUMENT STORE (Primary)
# ============================================================================

class PineconeDocumentStore(BaseDocumentStore):
    """
    Pinecone Vector Database Document Store.
    
    This is the PRIMARY vector store implementation.
    Works with Python 3.14 and provides managed cloud infrastructure.
    """
    
    def __init__(self, api_key: str = None, index_name: str = None):
        super().__init__()
        
        self.api_key = api_key or PINECONE_API_KEY
        self.index_name = index_name or PINECONE_INDEX_NAME
        
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY is required.")
        
        try:
            from pinecone import Pinecone
            
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            
            logger.info(f"PineconeDocumentStore initialized with index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def _get_namespace(self, collection_type: str) -> str:
        """Map collection type to Pinecone namespace."""
        return COLLECTIONS.get(collection_type, COLLECTIONS["general"])
    
    def add_document(self, collection_type: str, content: str,
                     metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Dict:
        """Add a document to Pinecone."""
        try:
            doc_id = doc_id or str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "doc_id": doc_id,
                "collection_type": collection_type,
                "ingested_at": datetime.now().isoformat(),
                "content_length": len(content)
            })
            
            chunks = self.text_splitter.split_text(content)
            namespace = self._get_namespace(collection_type)
            
            vectors = []
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                chunk_metadata["text"] = chunk
                
                embedding = self.embeddings.embed_query(chunk)
                
                vectors.append({
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": chunk_metadata
                })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i+batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
            
            logger.info(f"Added document {doc_id} with {len(chunks)} chunks to {collection_type}")
            
            return {
                "success": True,
                "doc_id": doc_id,
                "collection": collection_type,
                "chunk_count": len(chunks),
                "doc_ids": [v["id"] for v in vectors],
                "message": f"Document added with {len(chunks)} chunks"
            }
        except Exception as e:
            logger.error(f"Failed to add document to Pinecone: {e}")
            return {"success": False, "error": str(e)}
    
    def search(self, collection_type: str, query: str, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> Dict:
        """Search for documents in Pinecone."""
        try:
            namespace = self._get_namespace(collection_type)
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.index.query(
                namespace=namespace,
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_metadata
            )
            
            formatted_results = []
            for match in results.get("matches", []):
                formatted_results.append({
                    "id": match["id"],
                    "content": match.get("metadata", {}).get("text", ""),
                    "metadata": match.get("metadata", {}),
                    "relevance_score": match.get("score", 0)
                })
            
            return {
                "success": True,
                "results": formatted_results,
                "result_count": len(formatted_results),
                "collection": collection_type
            }
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return {"success": False, "results": [], "result_count": 0, "error": str(e)}
    
    def search_all_collections(self, query: str, top_k_per_collection: int = 3) -> Dict:
        """Search across all collections."""
        all_results = []
        for collection_type in COLLECTIONS.keys():
            try:
                result = self.search(collection_type, query, top_k=top_k_per_collection)
                if result.get("results"):
                    for r in result["results"]:
                        r["collection_type"] = collection_type
                    all_results.extend(result["results"])
            except Exception as e:
                logger.warning(f"Search in {collection_type} failed: {e}")
        
        all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return {"success": True, "results": all_results, "result_count": len(all_results)}
    
    def list_documents(self, collection_type: str, limit: int = 100) -> Dict:
        """List documents - limited support in Pinecone."""
        return {
            "success": True, 
            "documents": [], 
            "count": 0, 
            "message": "Pinecone doesn't support listing all documents. Use search instead."
        }
    
    def delete_document(self, collection_type: str, doc_id: str) -> Dict:
        """Delete a document from Pinecone."""
        try:
            namespace = self._get_namespace(collection_type)
            # Delete all chunks with this doc_id
            self.index.delete(filter={"doc_id": doc_id}, namespace=namespace)
            return {"success": True, "message": f"Deleted document {doc_id}"}
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_collection_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        try:
            stats = self.index.describe_index_stats()
            
            collections = {}
            for ns, data in (stats.namespaces or {}).items():
                # Reverse lookup collection type from namespace
                coll_type = next((k for k, v in COLLECTIONS.items() if v == ns), ns)
                collections[coll_type] = {
                    "document_count": data.vector_count,
                    "status": "active" if data.vector_count > 0 else "empty"
                }
            
            return {
                "success": True,
                "backend": "pinecone",
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "collections": collections
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"success": False, "error": str(e)}
    
    def clear_collection(self, collection_type: str) -> Dict:
        """Clear all vectors in a namespace."""
        try:
            namespace = self._get_namespace(collection_type)
            self.index.delete(delete_all=True, namespace=namespace)
            return {"success": True, "message": f"Cleared collection {collection_type}"}
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return {"success": False, "error": str(e)}


# ============================================================================
# SINGLETON INSTANCE AND FACTORY
# ============================================================================

_store_instance: Optional[BaseDocumentStore] = None


def get_vector_store() -> BaseDocumentStore:
    """
    Get the singleton vector store instance.
    
    Uses environment variable VECTOR_STORE_TYPE to determine backend:
    - "pinecone" (default): Uses Pinecone managed service
    - "chromadb" or "chromadb_http": Uses ChromaDB HTTP client (requires Docker)
    """
    global _store_instance
    
    if _store_instance is None:
        store_type = VECTOR_STORE_TYPE
        
        if store_type in ("chromadb", "chromadb_http"):
            logger.info("Attempting to initialize ChromaDB HTTP document store...")
            try:
                from agentic.chromadb_http_client import ChromaDBHttpDocumentStore
                _store_instance = ChromaDBHttpDocumentStore()
                logger.info("ChromaDB HTTP document store initialized")
            except Exception as e:
                logger.warning(f"ChromaDB HTTP failed (is Docker running?): {e}")
                logger.info("Falling back to Pinecone...")
                _store_instance = PineconeDocumentStore()
        else:
            # Default to Pinecone
            logger.info("Initializing Pinecone document store (primary)")
            try:
                _store_instance = PineconeDocumentStore()
            except Exception as e:
                logger.error(f"Failed to initialize Pinecone: {e}")
                raise RuntimeError(f"Vector store initialization failed: {e}")
    
    return _store_instance


def create_vector_store(store_type: str = "pinecone", **kwargs) -> BaseDocumentStore:
    """Create a new vector store instance (not singleton)."""
    store_type = store_type.lower()
    
    if store_type in ("chromadb", "chromadb_http"):
        from agentic.chromadb_http_client import ChromaDBHttpDocumentStore
        return ChromaDBHttpDocumentStore(**kwargs)
    else:
        return PineconeDocumentStore(**kwargs)


# Backward compatibility aliases
ChromaDocumentStore = PineconeDocumentStore  # Fallback to Pinecone
get_chroma_store = get_vector_store
create_chroma_store = create_vector_store
