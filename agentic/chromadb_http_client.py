# agentic/chromadb_http_client.py
"""
HTTP-based ChromaDB Client

This module provides a direct HTTP client for ChromaDB that bypasses
the chromadb Python package, which has compatibility issues with Python 3.14.

This client uses the ChromaDB REST API v2 directly.
"""

import os
import json
import uuid
import logging
import requests
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import AgenticConfig
from .embedding_batch_processor import get_batch_processor
from .embedding_cache_manager import get_embedding_cache

load_dotenv()
logger = logging.getLogger(__name__)


class ChromaDBHttpClient:
    """
    HTTP-based ChromaDB client that works with Python 3.14+
    by directly calling the ChromaDB REST API.
    """
    
    def __init__(self, host: str = None, port: int = None, auth_token: str = None):
        # Use config defaults if not specified
        host = host or AgenticConfig.CHROMADB_HOST
        port = port or AgenticConfig.CHROMADB_PORT

        self.base_url = f"http://{host}:{port}"
        self.auth_token = auth_token
        self.timeout = AgenticConfig.CHROMADB_TIMEOUT

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"

        # Cache for collection name -> ID mapping
        self._collection_cache: Dict[str, str] = {}

        # ============================================================================
        # HTTP SESSION POOLING - IMPROVEMENT #1 (290+ seconds per workflow saved!)
        # ============================================================================
        # Create a persistent session with connection pooling instead of creating
        # new TCP connections for each request. This reuses existing connections,
        # reducing overhead from ~290 seconds to ~10 seconds per workflow.
        # See: IMPLEMENTATION_ROADMAP.md Phase 1, Item 1
        self._session = requests.Session()

        # Configure connection pooling
        adapter = HTTPAdapter(
            pool_connections=10,      # Keep 10 connections alive in pool
            pool_maxsize=10,          # Max size of the pool per host
            max_retries=Retry(
                total=3,              # Retry up to 3 times
                backoff_factor=0.5,   # Exponential backoff: 0.5, 1, 2 seconds
                status_forcelist=[429, 500, 502, 503, 504]  # Retry on these status codes
            )
        )

        # Mount adapter for both HTTP and HTTPS
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)

        logger.info(f"ChromaDBHttpClient initialized with connection pooling for {self.base_url}")

        # Validate API key
        if not AgenticConfig.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY not found. Please set it in your .env file or environment variables."
            )

        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=AgenticConfig.GOOGLE_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _request(self, method: str, endpoint: str, data: dict = None, params: dict = None) -> requests.Response:
        """
        Make an HTTP request to the ChromaDB server.

        Uses a persistent session with connection pooling to reuse TCP connections
        instead of creating new ones for each request.

        Performance: ~290 seconds saved per workflow (1934 requests without pooling)
        """
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = self._session.get(url, headers=self.headers, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                response = self._session.post(url, headers=self.headers, json=data, timeout=self.timeout)
            elif method.upper() == "PUT":
                response = self._session.put(url, headers=self.headers, json=data, timeout=self.timeout)
            elif method.upper() == "DELETE":
                response = self._session.delete(url, headers=self.headers, json=data, timeout=self.timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            raise

    def close(self):
        """
        Cleanup: Close the session and release all pooled connections.

        This should be called when the client is no longer needed.
        """
        if hasattr(self, '_session') and self._session:
            self._session.close()
            logger.info("ChromaDBHttpClient session closed and connections released")

    def __del__(self):
        """Cleanup on object destruction."""
        self.close()
    
    def heartbeat(self) -> Dict:
        """Check if the ChromaDB server is alive."""
        try:
            response = self._request("GET", "/api/v2/heartbeat")
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            return {"success": False, "error": f"Status {response.status_code}: {response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def list_collections(self) -> List[Dict]:
        """List all collections."""
        try:
            response = self._request("GET", "/api/v2/tenants/default_tenant/databases/default_database/collections")
            if response.status_code == 200:
                collections = response.json()
                # Update cache
                for coll in collections:
                    self._collection_cache[coll["name"]] = coll["id"]
                return collections
            logger.error(f"Failed to list collections: {response.status_code} - {response.text}")
            return []
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []
    
    def _get_collection_id(self, name: str) -> Optional[str]:
        """Get collection ID from name, using cache or API."""
        # Check cache first
        if name in self._collection_cache:
            return self._collection_cache[name]
        
        # Refresh cache from API
        self.list_collections()
        return self._collection_cache.get(name)
    
    def get_or_create_collection(self, name: str, metadata: Dict = None) -> Dict:
        """Get or create a collection. Returns collection info with ID."""
        try:
            # Check if collection exists
            collection_id = self._get_collection_id(name)
            if collection_id:
                return {"id": collection_id, "name": name}
            
            # Create new collection
            data = {
                "name": name,
                "metadata": metadata or {"hnsw:space": "cosine"}
            }
            response = self._request("POST", "/api/v2/tenants/default_tenant/databases/default_database/collections", data=data)
            if response.status_code in [200, 201]:
                result = response.json()
                # Cache the new collection
                self._collection_cache[name] = result["id"]
                return result
            
            logger.error(f"Failed to create collection: {response.status_code} - {response.text}")
            return {"error": response.text}
        except Exception as e:
            logger.error(f"Failed to get/create collection: {e}")
            return {"error": str(e)}
    
    def add_documents(self, collection_name: str, ids: List[str], documents: List[str], 
                      metadatas: List[Dict] = None, embeddings: List[List[float]] = None) -> Dict:
        """Add documents to a collection."""
        try:
            # Ensure collection exists and get its ID
            coll_info = self.get_or_create_collection(collection_name)
            if "error" in coll_info:
                return {"success": False, "error": coll_info["error"]}
            collection_id = coll_info["id"]
            
            # Generate embeddings if not provided
            if embeddings is None:
                processor = get_batch_processor(self.embeddings)
                embeddings = processor.embed_documents_batch(documents)
            
            data = {
                "ids": ids,
                "documents": documents,
                "embeddings": embeddings
            }
            if metadatas:
                data["metadatas"] = metadatas
            
            response = self._request(
                "POST", 
                f"/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/add",
                data=data
            )
            
            if response.status_code in [200, 201]:
                return {"success": True, "count": len(ids)}
            
            logger.error(f"Failed to add documents: {response.status_code} - {response.text}")
            return {"success": False, "error": response.text}
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return {"success": False, "error": str(e)}
    
    def query(self, collection_name: str, query_texts: List[str] = None, 
              query_embeddings: List[List[float]] = None, n_results: int = 5,
              where: Dict = None, include: List[str] = None) -> Dict:
        """Query a collection."""
        try:
            # Get collection ID
            collection_id = self._get_collection_id(collection_name)
            if not collection_id:
                logger.warning(f"Collection {collection_name} not found")
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            
            # Generate embeddings if text provided (with caching)
            if query_embeddings is None and query_texts:
                cache = get_embedding_cache()
                query_embeddings = []
                uncached_texts = []
                uncached_indices = []

                # Check cache for each query
                for i, text in enumerate(query_texts):
                    cached = cache.get(text)
                    if cached:
                        query_embeddings.append(cached)
                    else:
                        uncached_texts.append(text)
                        uncached_indices.append(i)

                # Batch embed uncached queries
                if uncached_texts:
                    processor = get_batch_processor(self.embeddings)
                    new_embeddings = processor.embed_documents_batch(uncached_texts)

                    # Cache results
                    for text, embedding in zip(uncached_texts, new_embeddings):
                        cache.put(text, embedding)

                    # Merge results in original order
                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        query_embeddings.insert(idx, embedding)
            
            data = {
                "query_embeddings": query_embeddings,
                "n_results": n_results,
                "include": include or ["documents", "metadatas", "distances"]
            }
            if where:
                data["where"] = where
            
            response = self._request(
                "POST",
                f"/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/query",
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            
            logger.error(f"Failed to query: {response.status_code} - {response.text}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        except Exception as e:
            logger.error(f"Failed to query: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get(self, collection_name: str, ids: List[str] = None, 
            where: Dict = None, limit: int = 100, include: List[str] = None) -> Dict:
        """Get documents from a collection."""
        try:
            # Get collection ID
            collection_id = self._get_collection_id(collection_name)
            if not collection_id:
                return {"ids": [], "documents": [], "metadatas": []}
            
            data = {
                "limit": limit,
                "include": include or ["documents", "metadatas"]
            }
            if ids:
                data["ids"] = ids
            if where:
                data["where"] = where
            
            response = self._request(
                "POST",
                f"/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/get",
                data=data
            )
            
            if response.status_code == 200:
                return response.json()
            
            logger.error(f"Failed to get documents: {response.status_code} - {response.text}")
            return {"ids": [], "documents": [], "metadatas": []}
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
    
    def delete(self, collection_name: str, ids: List[str] = None, where: Dict = None) -> Dict:
        """Delete documents from a collection."""
        try:
            # Get collection ID
            collection_id = self._get_collection_id(collection_name)
            if not collection_id:
                return {"success": False, "error": "Collection not found"}
            
            data = {}
            if ids:
                data["ids"] = ids
            if where:
                data["where"] = where
            
            response = self._request(
                "POST",
                f"/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/delete",
                data=data
            )
            
            if response.status_code == 200:
                return {"success": True}
            
            logger.error(f"Failed to delete: {response.status_code} - {response.text}")
            return {"success": False, "error": response.text}
        except Exception as e:
            logger.error(f"Failed to delete: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_collection(self, collection_name: str) -> Dict:
        """Delete a collection."""
        try:
            # Get collection ID
            collection_id = self._get_collection_id(collection_name)
            if not collection_id:
                return {"success": False, "error": "Collection not found"}
            
            response = self._request(
                "DELETE",
                f"/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}"
            )
            
            if response.status_code in [200, 204]:
                # Remove from cache
                if collection_name in self._collection_cache:
                    del self._collection_cache[collection_name]
                return {"success": True}
            
            logger.error(f"Failed to delete collection: {response.status_code} - {response.text}")
            return {"success": False, "error": response.text}
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return {"success": False, "error": str(e)}
    
    def count(self, collection_name: str) -> int:
        """Get the count of documents in a collection."""
        try:
            # Get collection ID
            collection_id = self._get_collection_id(collection_name)
            if not collection_id:
                return 0
            
            response = self._request(
                "GET",
                f"/api/v2/tenants/default_tenant/databases/default_database/collections/{collection_id}/count"
            )
            
            if response.status_code == 200:
                return response.json()
            return 0
        except Exception as e:
            logger.error(f"Failed to get count: {e}")
            return 0


# ============================================================================
# CHROMADB HTTP DOCUMENT STORE (for vector_store.py integration)
# ============================================================================

class ChromaDBHttpDocumentStore:
    """
    HTTP-based ChromaDB Document Store that integrates with the vector_store.py
    architecture but uses direct HTTP calls instead of the chromadb Python package.
    """
    
    # Collection names
    COLLECTIONS = {
        "strategy": "strategy_documents",
        "standards": "standards_documents",
        "inventory": "inventory_documents",
        "general": "general_documents"
    }
    
    def __init__(self, host: str = None, port: int = None, auth_token: str = None):
        self.host = host or os.getenv("CHROMADB_HOST", "localhost")
        self.port = port or int(os.getenv("CHROMADB_PORT", "8000"))
        self.auth_token = auth_token or os.getenv("CHROMADB_AUTH_TOKEN")
        
        self.client = ChromaDBHttpClient(
            host=self.host,
            port=self.port,
            auth_token=self.auth_token
        )
        
        # Verify connection
        heartbeat = self.client.heartbeat()
        if not heartbeat.get("success"):
            raise ConnectionError(f"Cannot connect to ChromaDB at {self.host}:{self.port}: {heartbeat.get('error')}")
        
        logger.info(f"ChromaDBHttpDocumentStore connected to {self.host}:{self.port}")
    
    def _get_collection_name(self, collection_type: str) -> str:
        return self.COLLECTIONS.get(collection_type, self.COLLECTIONS["general"])
    
    def add_document(self, collection_type: str, content: str,
                     metadata: Optional[Dict] = None, doc_id: Optional[str] = None) -> Dict:
        """Add a document to a collection."""
        try:
            doc_id = doc_id or str(uuid.uuid4())
            metadata = metadata or {}
            metadata.update({
                "doc_id": doc_id,
                "collection_type": collection_type,
                "ingested_at": datetime.now().isoformat(),
                "content_length": len(content)
            })
            
            # Split content into chunks
            chunks = self.client.text_splitter.split_text(content)
            
            ids = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                chunk_metadata = metadata.copy()
                chunk_metadata["chunk_index"] = i
                chunk_metadata["total_chunks"] = len(chunks)
                
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append(chunk_metadata)
            
            collection_name = self._get_collection_name(collection_type)
            result = self.client.add_documents(collection_name, ids, documents, metadatas)
            
            if result.get("success"):
                return {
                    "success": True,
                    "doc_id": doc_id,
                    "collection": collection_type,
                    "chunks_created": len(chunks),
                    "message": f"Document added with {len(chunks)} chunks"
                }
            
            return {"success": False, "error": result.get("error")}
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return {"success": False, "error": str(e)}
    
    def search(self, collection_type: str, query: str, top_k: int = 5,
               filter_metadata: Optional[Dict] = None) -> Dict:
        """Search for documents in a collection."""
        try:
            collection_name = self._get_collection_name(collection_type)
            results = self.client.query(
                collection_name,
                query_texts=[query],
                n_results=top_k,
                where=filter_metadata
            )
            
            formatted = []
            if results.get("documents") and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    formatted.append({
                        "content": doc,
                        "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                        "relevance_score": float(1 / (1 + distance)),
                        "doc_id": results["metadatas"][0][i].get("doc_id") if results.get("metadatas") else None
                    })
            
            return {
                "success": True,
                "collection": collection_type,
                "query": query,
                "results": formatted,
                "result_count": len(formatted)
            }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    def search_all_collections(self, query: str, top_k_per_collection: int = 3) -> Dict:
        """Search across all collections."""
        all_results = {}
        for collection_type in self.COLLECTIONS.keys():
            all_results[collection_type] = self.search(collection_type, query, top_k_per_collection)
        
        merged = []
        for coll_type, result in all_results.items():
            if result.get("success"):
                for r in result.get("results", []):
                    r["source_collection"] = coll_type
                    merged.append(r)
        
        merged.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return {
            "success": True,
            "query": query,
            "merged_results": merged[:top_k_per_collection * 3],
            "by_collection": all_results
        }
    
    def list_documents(self, collection_type: str, limit: int = 100) -> Dict:
        """List documents in a collection."""
        try:
            collection_name = self._get_collection_name(collection_type)
            results = self.client.get(collection_name, limit=limit)
            
            seen_doc_ids = set()
            unique_docs = []
            for metadata in results.get("metadatas", []):
                doc_id = metadata.get("doc_id")
                if doc_id and doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    unique_docs.append({
                        "doc_id": doc_id,
                        "collection_type": metadata.get("collection_type"),
                        "ingested_at": metadata.get("ingested_at"),
                        "content_length": metadata.get("content_length"),
                        "total_chunks": metadata.get("total_chunks")
                    })
            
            return {
                "success": True,
                "collection": collection_type,
                "documents": unique_docs,
                "document_count": len(unique_docs)
            }
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return {"success": False, "error": str(e), "documents": []}
    
    def delete_document(self, collection_type: str, doc_id: str) -> Dict:
        """Delete a document from a collection."""
        try:
            collection_name = self._get_collection_name(collection_type)
            result = self.client.delete(collection_name, where={"doc_id": doc_id})
            
            if result.get("success"):
                return {"success": True, "doc_id": doc_id, "message": "Document deleted"}
            
            return {"success": False, "error": result.get("error")}
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {"success": False, "error": str(e)}
    
    def get_collection_stats(self) -> Dict:
        """Get statistics for all collections."""
        stats = {}
        for coll_type, coll_name in self.COLLECTIONS.items():
            try:
                count = self.client.count(coll_name)
                stats[coll_type] = {
                    "document_count": count,
                    "status": "active" if count > 0 else "empty"
                }
            except Exception:
                stats[coll_type] = {"document_count": 0, "status": "not_created"}
        
        return {
            "success": True,
            "backend": "chromadb_http",
            "host": f"{self.host}:{self.port}",
            "collections": stats
        }
    
    def clear_collection(self, collection_type: str) -> Dict:
        """Clear/delete a collection."""
        try:
            collection_name = self._get_collection_name(collection_type)
            result = self.client.delete_collection(collection_name)
            
            if result.get("success"):
                return {"success": True, "message": f"Collection {collection_type} cleared"}
            
            return {"success": False, "error": result.get("error")}
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return {"success": False, "error": str(e)}
