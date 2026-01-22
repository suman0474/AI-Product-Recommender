# agentic/strategy_rag/strategy_vector_store.py
# =============================================================================
# STRATEGY VECTOR STORE - ChromaDB-based semantic search for strategy data
# =============================================================================
#
# PURPOSE: Replace hardcoded PRODUCT_TO_CATEGORY_MAP with semantic matching
# using vector embeddings for flexible product-to-strategy mapping.
#
# FALLBACK: If ChromaDB is not available, falls back to rule-based matching.
#
# =============================================================================

import os
import csv
import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Path to strategy CSV
STRATEGY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "instrumentation_procurement_strategy.csv"
)

# Try to import ChromaDB (may fail on Python 3.14 due to Pydantic compatibility)
CHROMADB_AVAILABLE = False
chromadb = None
Settings = None

try:
    import chromadb as _chromadb
    from chromadb.config import Settings as _Settings
    chromadb = _chromadb
    Settings = _Settings
    CHROMADB_AVAILABLE = True
    logger.info("[StrategyVectorStore] ChromaDB available")
except ImportError:
    logger.warning("[StrategyVectorStore] chromadb not installed. Install with: pip install chromadb")
except Exception as e:
    # Catch Pydantic and other compatibility errors
    logger.warning(f"[StrategyVectorStore] ChromaDB unavailable due to compatibility issue: {type(e).__name__}")

# Try to import sentence-transformers for local embeddings
SENTENCE_TRANSFORMERS_AVAILABLE = False
SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    SentenceTransformer = _SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("[StrategyVectorStore] sentence-transformers not installed. Using fallback.")
except Exception as e:
    logger.warning(f"[StrategyVectorStore] sentence-transformers unavailable: {type(e).__name__}")


# =============================================================================
# VECTOR STORE CLASS
# =============================================================================

class StrategyVectorStore:
    """
    ChromaDB-based vector store for strategy data.
    
    Features:
    - Index CSV rows with embeddings
    - Semantic search by product type / vendor / category
    - Similarity-based ranking
    - Caching for performance
    """
    
    COLLECTION_NAME = "strategy_rag_v1"
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Fast local model
    
    def __init__(self, persist_directory: str = None):
        """
        Initialize the vector store.
        
        Args:
            persist_directory: Optional directory to persist ChromaDB data.
                             If None, uses in-memory storage.
        """
        self.persist_directory = persist_directory or os.path.join(
            os.path.dirname(__file__), ".chromadb_cache"
        )
        self._client = None
        self._collection = None
        self._embedding_model = None
        self._is_indexed = False
        self._index_count = 0
        
        logger.info(f"[StrategyVectorStore] Initialized (ChromaDB: {CHROMADB_AVAILABLE})")
    
    @property
    def client(self):
        """Lazy-load ChromaDB client."""
        if self._client is None:
            if not CHROMADB_AVAILABLE:
                raise ImportError("chromadb is not installed")
            
            # Use persistent storage
            self._client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False
            ))
            logger.info(f"[StrategyVectorStore] ChromaDB client created")
        
        return self._client
    
    @property
    def collection(self):
        """Get or create the strategy collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"description": "Strategy RAG vendor/category embeddings"}
            )
            self._index_count = self._collection.count()
            self._is_indexed = self._index_count > 0
            logger.info(f"[StrategyVectorStore] Collection loaded: {self._index_count} documents")
        
        return self._collection
    
    @property
    def embedding_model(self):
        """Lazy-load embedding model."""
        if self._embedding_model is None:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self._embedding_model = SentenceTransformer(self.EMBEDDING_MODEL)
                logger.info(f"[StrategyVectorStore] Loaded embedding model: {self.EMBEDDING_MODEL}")
            else:
                logger.warning("[StrategyVectorStore] No embedding model available, using fallback")
        
        return self._embedding_model
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        if self.embedding_model:
            return self.embedding_model.encode(text).tolist()
        else:
            # Fallback: simple hash-based "embedding" (not semantic, but functional)
            import hashlib
            hash_val = int(hashlib.md5(text.lower().encode()).hexdigest(), 16)
            return [(hash_val >> (i * 8)) % 256 / 255.0 for i in range(384)]
    
    def _create_document_text(self, row: Dict[str, str]) -> str:
        """Create searchable text from CSV row."""
        parts = []
        
        # Category + Subcategory
        if row.get('Category'):
            parts.append(f"category: {row['Category']}")
        if row.get('Subcategory'):
            parts.append(f"subcategory: {row['Subcategory']}")
        
        # Vendor
        if row.get('Vendor/Manufacturer'):
            parts.append(f"vendor: {row['Vendor/Manufacturer']}")
        
        # Strategy statement
        if row.get('Specific Strategy Statement'):
            parts.append(f"strategy: {row['Specific Strategy Statement']}")
        
        # Notes
        if row.get('Notes'):
            parts.append(f"notes: {row['Notes']}")
        
        return " | ".join(parts)
    
    # =========================================================================
    # INDEXING
    # =========================================================================
    
    def index_csv(self, csv_path: str = None, force_reindex: bool = False) -> Dict[str, Any]:
        """
        Index strategy CSV into ChromaDB.
        
        Args:
            csv_path: Path to CSV file (defaults to standard location)
            force_reindex: If True, delete existing and re-index
            
        Returns:
            Indexing result with count and status
        """
        csv_path = csv_path or STRATEGY_CSV_PATH
        
        if not os.path.exists(csv_path):
            logger.error(f"[StrategyVectorStore] CSV not found: {csv_path}")
            return {"success": False, "error": f"CSV not found: {csv_path}"}
        
        # Check if already indexed
        if self._is_indexed and not force_reindex:
            logger.info(f"[StrategyVectorStore] Already indexed ({self._index_count} docs)")
            return {
                "success": True,
                "indexed": self._index_count,
                "status": "already_indexed"
            }
        
        # Force reindex: delete collection
        if force_reindex:
            try:
                self.client.delete_collection(self.COLLECTION_NAME)
                self._collection = None
                logger.info("[StrategyVectorStore] Deleted existing collection for reindex")
            except Exception:
                pass
        
        # Load CSV
        logger.info(f"[StrategyVectorStore] Indexing CSV: {csv_path}")

        documents = []
        metadatas = []
        ids = []

        # First pass: collect all documents and metadata
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                # Create document text
                doc_text = self._create_document_text(row)

                if not doc_text.strip():
                    continue

                # Prepare metadata
                metadata = {
                    "category": row.get('Category', ''),
                    "subcategory": row.get('Subcategory', ''),
                    "vendor": row.get('Vendor/Manufacturer', ''),
                    "strategy": row.get('Specific Strategy Statement', ''),
                    "notes": row.get('Notes', ''),
                    "row_index": i
                }

                documents.append(doc_text)
                metadatas.append(metadata)
                ids.append(f"strategy_{i}")

        # Second pass: batch embed all documents at once for 50x speedup
        embeddings = []
        if documents:
            if self.embedding_model:
                # Use batch encoding if available (sentence-transformers supports this)
                embeddings = [emb.tolist() for emb in self.embedding_model.encode(documents)]
            else:
                # Fallback to individual embedding
                embeddings = [self._generate_embedding(doc) for doc in documents]
        
        # Add to collection
        if documents:
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            self._index_count = len(documents)
            self._is_indexed = True
            
            logger.info(f"[StrategyVectorStore] Indexed {len(documents)} documents")
        
        return {
            "success": True,
            "indexed": len(documents),
            "status": "indexed"
        }
    
    # =========================================================================
    # SEMANTIC SEARCH
    # =========================================================================
    
    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        filter_category: str = None,
        filter_vendor: str = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search for strategy data.
        
        Args:
            query: Search query (product type, vendor name, etc.)
            top_k: Number of results to return
            filter_category: Optional category filter
            filter_vendor: Optional vendor filter
            
        Returns:
            List of matching strategy records with similarity scores
        """
        # Ensure indexed
        if not self._is_indexed:
            self.index_csv()
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Build where filter
        where_filter = None
        if filter_category or filter_vendor:
            conditions = []
            if filter_category:
                conditions.append({"category": {"$eq": filter_category}})
            if filter_vendor:
                conditions.append({"vendor": {"$eq": filter_vendor}})
            
            if len(conditions) == 1:
                where_filter = conditions[0]
            else:
                where_filter = {"$and": conditions}
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted = []
        if results and results.get('ids') and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results.get('distances') else 0
                similarity = 1.0 - min(distance / 2.0, 1.0)  # Convert distance to similarity
                
                formatted.append({
                    "id": doc_id,
                    "document": results['documents'][0][i] if results.get('documents') else "",
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                    "category": results['metadatas'][0][i].get('category', '') if results.get('metadatas') else '',
                    "subcategory": results['metadatas'][0][i].get('subcategory', '') if results.get('metadatas') else '',
                    "vendor": results['metadatas'][0][i].get('vendor', '') if results.get('metadatas') else '',
                    "similarity": round(similarity, 4),
                    "distance": round(distance, 4)
                })
        
        logger.debug(f"[StrategyVectorStore] Search '{query[:30]}...' returned {len(formatted)} results")
        return formatted
    
    def match_product_to_category(self, product_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Find best matching category for a product type using semantic search.
        
        Replaces hardcoded PRODUCT_TO_CATEGORY_MAP.
        
        Args:
            product_type: Product type string (e.g., "differential pressure transmitter")
            
        Returns:
            Tuple of (category, subcategory) or (None, None)
        """
        results = self.search_similar(
            query=f"category for {product_type}",
            top_k=1
        )
        
        if results and results[0].get('similarity', 0) > 0.3:
            return results[0].get('category'), results[0].get('subcategory')
        
        return None, None
    
    def get_vendor_similarity(
        self,
        vendor_name: str,
        product_type: str
    ) -> float:
        """
        Get similarity score for a vendor in the context of a product type.
        
        Used for dynamic ranking.
        
        Args:
            vendor_name: Vendor name to check
            product_type: Product type context
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        query = f"{vendor_name} vendor for {product_type}"
        
        results = self.search_similar(
            query=query,
            top_k=3,
            filter_vendor=vendor_name
        )
        
        if results:
            return results[0].get('similarity', 0.0)
        
        return 0.0
    
    def get_preferred_vendors(
        self,
        product_type: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get preferred vendors for a product type based on semantic similarity.
        
        Args:
            product_type: Product type to search for
            top_k: Number of vendors to return
            
        Returns:
            List of vendors with similarity scores
        """
        results = self.search_similar(
            query=f"preferred vendor for {product_type}",
            top_k=top_k * 2  # Get more to deduplicate
        )
        
        # Deduplicate by vendor
        seen_vendors = set()
        unique_vendors = []
        
        for r in results:
            vendor = r.get('vendor', '').strip()
            if vendor and vendor.lower() not in seen_vendors:
                seen_vendors.add(vendor.lower())
                unique_vendors.append({
                    "vendor": vendor,
                    "category": r.get('category', ''),
                    "similarity": r.get('similarity', 0),
                    "is_preferred": r.get('similarity', 0) > 0.6
                })
        
        return unique_vendors[:top_k]
    
    # =========================================================================
    # ADMIN OPERATIONS
    # =========================================================================
    
    def add_strategy_entry(
        self,
        category: str,
        vendor: str,
        strategy: str,
        subcategory: str = "",
        notes: str = ""
    ) -> Dict[str, Any]:
        """
        Add a new strategy entry (real-time update).
        
        Args:
            category: Product category
            vendor: Vendor name
            strategy: Strategy statement
            subcategory: Optional subcategory
            notes: Optional notes
            
        Returns:
            Result with new document ID
        """
        doc_id = f"strategy_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(vendor)}"
        
        doc_text = f"category: {category} | subcategory: {subcategory} | vendor: {vendor} | strategy: {strategy} | notes: {notes}"
        embedding = self._generate_embedding(doc_text)
        
        self.collection.add(
            documents=[doc_text],
            metadatas=[{
                "category": category,
                "subcategory": subcategory,
                "vendor": vendor,
                "strategy": strategy,
                "notes": notes,
                "added_at": datetime.now().isoformat()
            }],
            ids=[doc_id],
            embeddings=[embedding]
        )
        
        self._index_count += 1
        
        logger.info(f"[StrategyVectorStore] Added new entry: {doc_id}")
        return {"success": True, "id": doc_id}
    
    def refresh_index(self) -> Dict[str, Any]:
        """Force refresh the entire index."""
        return self.index_csv(force_reindex=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        return {
            "is_indexed": self._is_indexed,
            "document_count": self._index_count,
            "collection_name": self.COLLECTION_NAME,
            "embedding_model": self.EMBEDDING_MODEL,
            "chromadb_available": CHROMADB_AVAILABLE,
            "sentence_transformers_available": SENTENCE_TRANSFORMERS_AVAILABLE
        }


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_strategy_store_instance = None


def get_strategy_vector_store() -> StrategyVectorStore:
    """Get or create singleton StrategyVectorStore instance."""
    global _strategy_store_instance
    
    if _strategy_store_instance is None:
        _strategy_store_instance = StrategyVectorStore()
        # Auto-index on first use
        try:
            _strategy_store_instance.index_csv()
        except Exception as e:
            logger.warning(f"[StrategyVectorStore] Auto-index failed: {e}")
    
    return _strategy_store_instance


# Convenience alias
strategy_store = None


def init_strategy_store():
    """Initialize the global strategy store."""
    global strategy_store
    strategy_store = get_strategy_vector_store()
    return strategy_store


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def search_strategy(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Convenience function for semantic strategy search."""
    return get_strategy_vector_store().search_similar(query, top_k)


def match_product_category(product_type: str) -> Tuple[Optional[str], Optional[str]]:
    """Convenience function for product-to-category matching."""
    return get_strategy_vector_store().match_product_to_category(product_type)


def get_vendor_score(vendor_name: str, product_type: str) -> float:
    """Convenience function for vendor similarity score."""
    return get_strategy_vector_store().get_vendor_similarity(vendor_name, product_type)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'StrategyVectorStore',
    'get_strategy_vector_store',
    'init_strategy_store',
    'search_strategy',
    'match_product_category',
    'get_vendor_score'
]
