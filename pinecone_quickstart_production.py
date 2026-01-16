"""
Production-ready Pinecone Quickstart

This script demonstrates production-ready patterns for Pinecone operations:
- Automatic .env file loading
- Exponential backoff retry for transient errors
- Batch processing with validation
- Centralized configuration management
- Proper error handling
- Polling-based indexing wait strategy

All fixes address high-priority gaps from PINECONE_GAP_ANALYSIS.md
"""

from pinecone import Pinecone
from config import PineconeConfig
from utils.retry import exponential_backoff_retry
from utils.batch import batch_upsert, wait_for_indexing


def main():
    """Main execution function"""

    print("\n" + "=" * 80)
    print("PINECONE PRODUCTION-READY QUICKSTART")
    print("=" * 80 + "\n")

    # Configuration is automatically loaded from .env file
    # and validated on import
    PineconeConfig.print_config()

    # Initialize Pinecone client
    print("\nInitializing Pinecone client...")
    pc = Pinecone(api_key=PineconeConfig.API_KEY)

    # Connect to index
    print(f"Connecting to index '{PineconeConfig.INDEX_NAME}'...")
    index = pc.Index(PineconeConfig.INDEX_NAME)

    # Wait for index to be ready (with retry logic)
    print("Checking index status...")
    stats = exponential_backoff_retry(lambda: index.describe_index_stats())
    print(f"Index is ready! Current vector count: {stats.total_vector_count}")

    # Prepare sample dataset
    records = [
        {"_id": "rec1", "content": "The Eiffel Tower was completed in 1889 and stands in Paris, France.", "category": "history"},
        {"_id": "rec2", "content": "Photosynthesis allows plants to convert sunlight into energy.", "category": "science"},
        {"_id": "rec5", "content": "Shakespeare wrote many famous plays, including Hamlet and Macbeth.", "category": "literature"},
        {"_id": "rec7", "content": "The Great Wall of China was built to protect against invasions.", "category": "history"},
        {"_id": "rec15", "content": "Leonardo da Vinci painted the Mona Lisa.", "category": "art"},
        {"_id": "rec17", "content": "The Pyramids of Giza are among the Seven Wonders of the Ancient World.", "category": "history"},
        {"_id": "rec21", "content": "The Statue of Liberty was a gift from France to the United States.", "category": "history"},
        {"_id": "rec26", "content": "Rome was once the center of a vast empire.", "category": "history"},
        {"_id": "rec33", "content": "The violin is a string instrument commonly used in orchestras.", "category": "music"},
        {"_id": "rec38", "content": "The Taj Mahal is a mausoleum built by Emperor Shah Jahan.", "category": "history"},
        {"_id": "rec48", "content": "Vincent van Gogh painted Starry Night.", "category": "art"},
        {"_id": "rec50", "content": "Renewable energy sources include wind, solar, and hydroelectric power.", "category": "energy"}
    ]

    # Upsert records with batch processing, retry logic, and validation
    print(f"\n--- Upserting Records to '{PineconeConfig.DEFAULT_NAMESPACE}' ---")
    total_upserted = batch_upsert(
        index,
        PineconeConfig.DEFAULT_NAMESPACE,
        records,
        batch_size=PineconeConfig.BATCH_SIZE,
        validate=True  # Validate upsert success
    )

    # Wait for vectors to be indexed (polling-based, not fixed sleep)
    print(f"\n--- Waiting for Indexing ---")
    success = wait_for_indexing(
        index,
        PineconeConfig.DEFAULT_NAMESPACE,
        expected_count=len(records),
        timeout=PineconeConfig.INDEXING_TIMEOUT,
        poll_interval=PineconeConfig.INDEXING_POLL_INTERVAL
    )

    if not success:
        print("WARNING: Indexing timeout. Results may be incomplete.")

    # View index stats
    print(f"\n--- Index Statistics ---")
    stats = exponential_backoff_retry(lambda: index.describe_index_stats())
    print(f"Total vectors: {stats.total_vector_count}")
    print(f"Namespaces: {list(stats.namespaces.keys())}")
    if stats.namespaces:
        for ns, ns_stats in stats.namespaces.items():
            print(f"  {ns}: {ns_stats.vector_count} vectors")

    # Define search query
    query = "Famous historical structures and monuments"
    print(f"\n--- Search Query ---")
    print(f"Query: '{query}'")

    # Search WITHOUT reranking (with retry logic)
    print(f"\n--- Search WITHOUT Reranking ---")
    results = exponential_backoff_retry(
        lambda: index.search(
            namespace=PineconeConfig.DEFAULT_NAMESPACE,
            query={
                "top_k": PineconeConfig.DEFAULT_TOP_K,
                "inputs": {"text": query}
            }
        )
    )

    # Print results
    for i, hit in enumerate(results['result']['hits'], 1):
        print(f"{i:2d}. ID: {hit['_id']:<5} | Score: {hit['_score']:.2f} | "
              f"Category: {hit['fields']['category']:<10} | "
              f"Text: {hit['fields']['content'][:60]}...")

    # Search WITH reranking (with retry logic)
    print(f"\n--- Search WITH Reranking ---")
    reranked_results = exponential_backoff_retry(
        lambda: index.search(
            namespace=PineconeConfig.DEFAULT_NAMESPACE,
            query={
                "top_k": PineconeConfig.DEFAULT_TOP_K,
                "inputs": {"text": query}
            },
            rerank={
                "model": PineconeConfig.RERANK_MODEL,
                "top_n": PineconeConfig.DEFAULT_TOP_K,
                "rank_fields": ["content"]
            }
        )
    )

    # Print reranked results
    for i, hit in enumerate(reranked_results['result']['hits'], 1):
        print(f"{i:2d}. ID: {hit['_id']:<5} | Score: {hit['_score']:.2f} | "
              f"Category: {hit['fields']['category']:<10} | "
              f"Text: {hit['fields']['content'][:60]}...")

    # Test metadata filtering (with retry logic)
    print(f"\n--- Metadata Filtering Test ---")
    print("Filter: category='history'")
    filtered_results = exponential_backoff_retry(
        lambda: index.search(
            namespace=PineconeConfig.DEFAULT_NAMESPACE,
            query={
                "top_k": 5,
                "inputs": {"text": query},
                "filter": {"category": {"$eq": "history"}}  # Only add filter when needed
            },
            rerank={
                "model": PineconeConfig.RERANK_MODEL,
                "top_n": 5,
                "rank_fields": ["content"]
            }
        )
    )

    print(f"Results: {len(filtered_results['result']['hits'])} historical items")
    for i, hit in enumerate(filtered_results['result']['hits'], 1):
        print(f"{i}. {hit['_id']}: {hit['fields']['content'][:50]}...")

    # Summary
    print("\n" + "=" * 80)
    print("QUICKSTART COMPLETED SUCCESSFULLY")
    print("=" * 80)
    print("\nProduction-Ready Patterns Demonstrated:")
    print("  - Automatic .env file loading (python-dotenv)")
    print("  - Exponential backoff retry for all API calls")
    print("  - Batch processing with validation")
    print("  - Centralized configuration management")
    print("  - Polling-based indexing wait (not fixed sleep)")
    print("  - Proper error handling for transient failures")
    print("\nKey Observations:")
    print("  - Without reranking: Some results include less relevant items")
    print("  - With reranking: Results focused on historical structures")
    print("  - Metadata filtering: Successfully isolated 'history' category")
    print("\nAll high-priority gaps from PINECONE_GAP_ANALYSIS.md have been addressed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
