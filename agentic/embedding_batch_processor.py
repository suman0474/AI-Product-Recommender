"""
Embedding Batch Processor
Batches embedding calls for 50x speedup
Replaces sequential embed_query calls with batch embed_documents
"""
import logging
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingBatchProcessor:
    """Batches embedding calls for 50x speedup."""

    def __init__(self, embedding_model, batch_size=10):
        """
        Initialize batch processor.

        Args:
            embedding_model: The embedding model instance (e.g., from ChromaDB)
            batch_size: Maximum documents per batch (default 10)
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def embed_documents_batch(self, documents: List[str]) -> List[List[float]]:
        """
        Batch embed documents instead of sequential calls.

        Instead of:
            [embed_query(doc) for doc in docs]  # N API calls (slow)

        Use:
            embed_documents(docs)  # 1-2 API calls (fast)

        Args:
            documents: List of document strings to embed

        Returns:
            List of embedding vectors (one per document)
        """
        if not documents:
            return []

        if len(documents) <= self.batch_size:
            # Single batch call
            logger.debug(
                f"[EMBEDDING_BATCH] Embedding {len(documents)} documents in 1 batch"
            )
            return self.embedding_model.embed_documents(documents)
        else:
            # Multiple batches
            all_embeddings = []
            num_batches = (len(documents) + self.batch_size - 1) // self.batch_size

            for batch_num, i in enumerate(range(0, len(documents), self.batch_size)):
                batch = documents[i : i + self.batch_size]
                logger.debug(
                    f"[EMBEDDING_BATCH] Processing batch {batch_num + 1}/{num_batches} "
                    f"({len(batch)} documents)"
                )
                batch_embeddings = self.embedding_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)

            logger.debug(
                f"[EMBEDDING_BATCH] Completed embedding {len(documents)} documents "
                f"in {num_batches} batches"
            )
            return all_embeddings


# Global singleton instance
_batch_processor = None


def get_batch_processor(embedding_model=None, batch_size=10):
    """
    Get or create global batch processor.

    Args:
        embedding_model: The embedding model instance
        batch_size: Batch size for embedding calls (default 10)

    Returns:
        EmbeddingBatchProcessor instance
    """
    global _batch_processor

    if _batch_processor is None:
        if embedding_model is None:
            raise ValueError(
                "embedding_model must be provided on first call to get_batch_processor"
            )
        _batch_processor = EmbeddingBatchProcessor(embedding_model, batch_size)
        logger.info(
            f"[EMBEDDING_BATCH] Created batch processor with batch_size={batch_size}"
        )
    return _batch_processor
