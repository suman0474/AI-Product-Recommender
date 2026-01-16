"""
Batch processing utilities for Pinecone operations.

According to CLAUDE.md best practices:
- Text records: Maximum 96 records per batch
- Vector records: Maximum 1000 records per batch
- Maximum 2MB per batch for both types
"""

import time
from typing import List, Dict, Any, Callable
from .retry import exponential_backoff_retry


def batch_upsert(
    index: Any,
    namespace: str,
    records: List[Dict[str, Any]],
    batch_size: int = 96,
    delay_between_batches: float = 0.1,
    validate: bool = True
) -> int:
    """
    Upsert records in batches with retry logic and validation.

    Args:
        index: Pinecone index object
        namespace: Namespace to upsert records into
        records: List of records to upsert
        batch_size: Number of records per batch (default: 96 for text records)
        delay_between_batches: Delay in seconds between batches (default: 0.1)
        validate: Whether to validate upsert success (default: True)

    Returns:
        Total number of records successfully upserted

    Raises:
        Exception: If upsert fails after retries or validation fails

    Example:
        >>> records = [{"_id": "1", "content": "text", "metadata": {...}}, ...]
        >>> total = batch_upsert(index, "my-namespace", records)
        >>> print(f"Upserted {total} records")
    """
    total_upserted = 0
    total_records = len(records)

    print(f"Starting batch upsert: {total_records} records in batches of {batch_size}")

    for i in range(0, total_records, batch_size):
        batch = records[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_records + batch_size - 1) // batch_size

        print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} records)...")

        try:
            # Upsert with retry logic
            response = exponential_backoff_retry(
                lambda: index.upsert_records(namespace, batch)
            )

            # Validate upsert if requested
            if validate:
                upserted_count = len(batch)  # Assume success if no exception

                # Try to get actual count from response if available
                if hasattr(response, 'upserted_count'):
                    upserted_count = response.upserted_count
                elif isinstance(response, dict) and 'upserted_count' in response:
                    upserted_count = response['upserted_count']

                if upserted_count != len(batch):
                    print(f"WARNING: Batch {batch_num} - Expected {len(batch)} but upserted {upserted_count}")
                else:
                    print(f"  Success: {upserted_count} records upserted")

                total_upserted += upserted_count
            else:
                total_upserted += len(batch)
                print(f"  Success: {len(batch)} records upserted (validation skipped)")

        except Exception as e:
            print(f"ERROR: Batch {batch_num} failed: {e}")
            raise

        # Add delay between batches to avoid rate limits
        if i + batch_size < total_records:
            time.sleep(delay_between_batches)

    print(f"Batch upsert complete: {total_upserted}/{total_records} records upserted")
    return total_upserted


def batch_process(
    items: List[Any],
    process_func: Callable[[List[Any]], Any],
    batch_size: int = 96,
    delay_between_batches: float = 0.1,
    show_progress: bool = True
) -> List[Any]:
    """
    Generic batch processing with retry logic.

    Args:
        items: List of items to process
        process_func: Function to process each batch
        batch_size: Number of items per batch
        delay_between_batches: Delay in seconds between batches
        show_progress: Whether to print progress messages

    Returns:
        List of results from each batch

    Example:
        >>> def process_batch(batch):
        ...     return index.delete(namespace="ns", ids=batch)
        >>> results = batch_process(record_ids, process_batch)
    """
    results = []
    total_items = len(items)

    if show_progress:
        print(f"Processing {total_items} items in batches of {batch_size}")

    for i in range(0, total_items, batch_size):
        batch = items[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_items + batch_size - 1) // batch_size

        if show_progress:
            print(f"Processing batch {batch_num}/{total_batches} ({len(batch)} items)...")

        try:
            result = exponential_backoff_retry(
                lambda: process_func(batch)
            )
            results.append(result)

            if show_progress:
                print(f"  Batch {batch_num} complete")

        except Exception as e:
            print(f"ERROR: Batch {batch_num} failed: {e}")
            raise

        # Add delay between batches
        if i + batch_size < total_items:
            time.sleep(delay_between_batches)

    if show_progress:
        print(f"All {total_batches} batches processed successfully")

    return results


def wait_for_indexing(
    index: Any,
    namespace: str,
    expected_count: int,
    timeout: int = 60,
    poll_interval: int = 2
) -> bool:
    """
    Wait for vectors to be indexed with polling strategy.

    Args:
        index: Pinecone index object
        namespace: Namespace to check
        expected_count: Expected number of vectors
        timeout: Maximum time to wait in seconds (default: 60)
        poll_interval: Time between polls in seconds (default: 2)

    Returns:
        True if indexing complete, False if timeout

    Example:
        >>> success = wait_for_indexing(index, "my-namespace", 100)
        >>> if success:
        ...     print("Indexing complete!")
    """
    import time

    start_time = time.time()
    print(f"Waiting for {expected_count} vectors to be indexed in namespace '{namespace}'...")

    while time.time() - start_time < timeout:
        try:
            stats = exponential_backoff_retry(
                lambda: index.describe_index_stats()
            )

            if namespace in stats.namespaces:
                current_count = stats.namespaces[namespace].vector_count

                if current_count >= expected_count:
                    elapsed = time.time() - start_time
                    print(f"Indexing complete: {current_count} vectors indexed in {elapsed:.1f}s")
                    return True

                print(f"  Indexing in progress... {current_count}/{expected_count} vectors")
            else:
                print(f"  Namespace '{namespace}' not yet visible...")

        except Exception as e:
            print(f"  Error checking index stats: {e}")

        time.sleep(poll_interval)

    elapsed = time.time() - start_time
    print(f"WARNING: Timeout after {elapsed:.1f}s. Indexing may still be in progress.")
    return False
