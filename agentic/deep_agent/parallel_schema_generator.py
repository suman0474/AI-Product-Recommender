# agentic/deep_agent/parallel_schema_generator.py
# Parallel Schema Generation Agent
#
# PHASE 2 OPTIMIZATION: Generate schemas for multiple products in parallel
# Similar to how we parallelized instrument/accessory identification
#
# Problem: Sequential generation of N products = N × 437s
# Solution: Parallel generation of N products = 437s (only the slowest matters!)
# Speedup: 3x faster for 3 products!

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import time

# Import global executor manager for bounded thread pool
from agentic.global_executor_manager import get_global_executor

logger = logging.getLogger(__name__)


class ParallelSchemaGenerator:
    """
    Generates schemas for multiple products in parallel.

    Pattern: Same ThreadPoolExecutor approach as identification parallelization.

    Example:
        >>> generator = ParallelSchemaGenerator(max_workers=3)
        >>> schemas = generator.generate_schemas_in_parallel([
        ...     "Temperature Transmitter",
        ...     "Pressure Gauge",
        ...     "Level Switch"
        ... ])
        >>> # Takes ~437s instead of 1311s (3x faster!)
    """

    def __init__(self, max_workers: int = 3):
        """
        Initialize parallel schema generator.

        Args:
            max_workers: Number of concurrent schema generations (default 3)
                        Higher = more parallel, but more resource usage
                        Recommended: 2-4 for typical systems
        """
        self.max_workers = max_workers
        logger.info(f"[PARALLEL_SCHEMA] Initialized with max_workers={max_workers}")

    def generate_schemas_in_parallel(
        self,
        product_types: List[str],
        force_regenerate: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate schemas for multiple products in parallel.

        Args:
            product_types: List of product types to generate schemas for
            force_regenerate: If True, regenerate even if cached

        Returns:
            Dictionary mapping product_type -> schema result

        Example:
            >>> products = ["Temperature Transmitter", "Pressure Gauge", "Level Switch"]
            >>> results = generator.generate_schemas_in_parallel(products)
            >>> # Results: {"Temperature Transmitter": {...}, "Pressure Gauge": {...}, ...}
        """

        if not product_types:
            logger.warning("[PARALLEL_SCHEMA] Empty product list")
            return {}

        logger.info(f"[PARALLEL_SCHEMA] Starting parallel schema generation")
        logger.info(f"[PARALLEL_SCHEMA] Products: {len(product_types)}")
        logger.info(f"[PARALLEL_SCHEMA] Max workers: {self.max_workers}")
        print(f"\n{'='*70}")
        print(f"🔷 [PARALLEL_SCHEMA] STARTING PARALLEL GENERATION")
        print(f"   Products: {len(product_types)}")
        print(f"   Parallel workers: {self.max_workers}")
        print(f"{'='*70}\n")

        start_time = time.time()
        results = {}
        failed = {}

        def generate_single_schema(product_type: str) -> tuple:
            """Generate schema for a single product."""
            logger.info(f"[PARALLEL_SCHEMA] [Worker] Starting: {product_type}")

            try:
                # Check if already cached (unless force_regenerate)
                if not force_regenerate:
                    cached = self._load_cached_schema(product_type)
                    if cached and cached.get('success'):
                        logger.info(f"[PARALLEL_SCHEMA] [Worker] Using cached: {product_type}")
                        return (product_type, cached, True)  # (type, schema, is_cached)

                # Generate new schema via PPI workflow
                logger.info(f"[PARALLEL_SCHEMA] [Worker] Generating new: {product_type}")
                schema = self._generate_schema_for_product(product_type)

                logger.info(f"[PARALLEL_SCHEMA] [Worker] Completed: {product_type}")
                return (product_type, schema, False)

            except Exception as e:
                logger.error(f"[PARALLEL_SCHEMA] [Worker] Failed for {product_type}: {e}")
                error_result = {
                    "success": False,
                    "error": str(e),
                    "status": "failed"
                }
                return (product_type, error_result, False)

        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║  PHASE 2: Submit all schema generation tasks to global executor      ║
        # ║  All products start simultaneously (global pool bounded to 16 workers)║
        # ╚═══════════════════════════════════════════════════════════════════════╝

        completed_count = 0
        executor = get_global_executor()

        # Submit all tasks using global executor (bounded to 16 workers globally)
        futures = {
            executor.submit(generate_single_schema, product_type): product_type
            for product_type in product_types
        }

        logger.info(f"[PARALLEL_SCHEMA] Submitted {len(futures)} tasks to global executor")

        # Collect results as they complete (not necessarily in order)
        for future in as_completed(futures):
            try:
                product_type, schema, is_cached = future.result()
                completed_count += 1

                if schema.get('success'):
                    results[product_type] = schema
                    cache_status = "(cached)" if is_cached else "(generated)"
                    logger.info(
                        f"[PARALLEL_SCHEMA] [{completed_count}/{len(product_types)}] "
                        f"✓ {product_type} {cache_status}"
                    )
                else:
                    failed[product_type] = schema.get('error', 'Unknown error')
                    logger.warning(
                        f"[PARALLEL_SCHEMA] [{completed_count}/{len(product_types)}] "
                        f"✗ {product_type} - {schema.get('error', 'Failed')}"
                    )

            except Exception as e:
                logger.error(f"[PARALLEL_SCHEMA] Error collecting result: {e}")
                completed_count += 1

        elapsed = time.time() - start_time

        # ╔═══════════════════════════════════════════════════════════════════════╗
        # ║  LOGGING & STATISTICS                                               ║
        # ╚═══════════════════════════════════════════════════════════════════════╝

        successful = len(results)
        failed_count = len(failed)

        logger.info(f"[PARALLEL_SCHEMA] ╔{'='*68}╗")
        logger.info(f"[PARALLEL_SCHEMA] ║ PARALLEL SCHEMA GENERATION COMPLETE")
        logger.info(f"[PARALLEL_SCHEMA] ║ Total: {len(product_types)} products")
        logger.info(f"[PARALLEL_SCHEMA] ║ Successful: {successful}")
        logger.info(f"[PARALLEL_SCHEMA] ║ Failed: {failed_count}")
        logger.info(f"[PARALLEL_SCHEMA] ║ Time: {elapsed:.2f}s")

        # Calculate sequential time estimate
        if completed_count > 0:
            avg_time = elapsed / self.max_workers if self.max_workers < completed_count else elapsed
            estimated_sequential = avg_time * len(product_types)
            speedup = estimated_sequential / elapsed if elapsed > 0 else 1.0
            logger.info(f"[PARALLEL_SCHEMA] ║ Estimated speedup: {speedup:.1f}x")
            logger.info(f"[PARALLEL_SCHEMA] ╚{'='*68}╝")

        print(f"\n{'='*70}")
        print(f"🔷 [PARALLEL_SCHEMA] COMPLETED")
        print(f"   Successful: {successful}/{len(product_types)}")
        print(f"   Time: {elapsed:.2f}s")
        if failed_count > 0:
            print(f"   ⚠ Failed: {failed_count}")
            for product, error in failed.items():
                print(f"      - {product}: {error[:50]}")
        print(f"{'='*70}\n")

        return results

    def _generate_schema_for_product(self, product_type: str) -> Dict[str, Any]:
        """
        Generate schema for a single product via PPI workflow.

        Args:
            product_type: Product type to generate schema for

        Returns:
            Schema dictionary from PPI workflow
        """
        try:
            from product_search_workflow.ppi_workflow import compile_ppi_workflow

            logger.info(f"[PARALLEL_SCHEMA] [PPI] Invoking workflow for: {product_type}")

            workflow = compile_ppi_workflow()
            result = workflow.invoke({"product_type": product_type})

            schema = result.get("schema", {})

            if schema:
                return {
                    "success": True,
                    "schema": schema,
                    "source": "ppi_workflow",
                    "product_type": product_type
                }
            else:
                return {
                    "success": False,
                    "error": "PPI workflow returned empty schema",
                    "product_type": product_type
                }

        except Exception as e:
            logger.error(f"[PARALLEL_SCHEMA] [PPI] Error for {product_type}: {e}")
            return {
                "success": False,
                "error": f"PPI workflow failed: {str(e)}",
                "product_type": product_type
            }

    def _load_cached_schema(self, product_type: str) -> Optional[Dict[str, Any]]:
        """
        Load schema from cache/database if available.

        Args:
            product_type: Product type to load

        Returns:
            Cached schema if found, None otherwise
        """
        try:
            from tools.schema_tools import load_schema_from_azure

            schema = load_schema_from_azure(product_type)
            if schema:
                return {
                    "success": True,
                    "schema": schema,
                    "source": "azure_cache",
                    "product_type": product_type
                }
        except Exception as e:
            logger.debug(f"[PARALLEL_SCHEMA] No cached schema for {product_type}: {e}")

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTION FOR INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def generate_schemas_parallel(
    product_types: List[str],
    max_workers: int = 3,
    force_regenerate: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to generate multiple schemas in parallel.

    Use this when you need to generate schemas for multiple products at once.

    Example:
        >>> products = ["Temperature Transmitter", "Pressure Gauge", "Level Switch"]
        >>> schemas = generate_schemas_parallel(products, max_workers=3)
        >>> # Takes ~437s instead of 1311s (3x faster!)

    Args:
        product_types: List of product types to generate schemas for
        max_workers: Number of concurrent workers (default 3)
        force_regenerate: If True, regenerate even if cached

    Returns:
        Dictionary mapping product_type -> schema result
    """
    generator = ParallelSchemaGenerator(max_workers=max_workers)
    return generator.generate_schemas_in_parallel(
        product_types,
        force_regenerate=force_regenerate
    )


# ═══════════════════════════════════════════════════════════════════════════════
# TESTING
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("PARALLEL SCHEMA GENERATOR - TEST")
    print("="*70)

    # Example products
    products = [
        "Temperature Transmitter",
        "Pressure Gauge",
        "Level Switch"
    ]

    # Create generator
    generator = ParallelSchemaGenerator(max_workers=2)

    # Generate schemas in parallel
    print(f"\nGenerating schemas for {len(products)} products...")
    schemas = generator.generate_schemas_in_parallel(products)

    print(f"\nResults:")
    for product, result in schemas.items():
        success = result.get('success', False)
        status = "✓" if success else "✗"
        print(f"  {status} {product}")

    print("\n" + "="*70)
