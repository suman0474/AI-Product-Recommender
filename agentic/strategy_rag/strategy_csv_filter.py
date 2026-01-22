# agentic/strategy_csv_filter.py
# =============================================================================
# STRATEGY CSV FILTER - Simple CSV-Based Vendor Filtering
# =============================================================================
#
# PURPOSE: Replace complex Strategy RAG with simple CSV-based vendor filtering.
# Reads vendor strategy data from instrumentation_procurement_strategy.csv
# and filters vendors based on category/subcategory matching.
#
# CSV COLUMNS:
#   vendor ID, vendor name, category, subcategory, strategy, refinery,
#   additional comments, owner name
#
# USAGE:
#   filter = StrategyCSVFilter()
#   result = filter.filter_vendors_for_product(
#       product_type="pressure transmitter",
#       available_vendors=["Emerson", "ABB", "Siemens"]
#   )
#
# =============================================================================

import csv
import logging
import os
from typing import Dict, Any, List, Optional, Tuple
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# Path to the strategy CSV file
STRATEGY_CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "instrumentation_procurement_strategy.csv"
)

# Category mapping: Map product types to CSV categories
PRODUCT_TO_CATEGORY_MAP = {
    # Pressure instruments
    "pressure transmitter": ("Pressure Instruments", None),
    "pressure gauge": ("Pressure Instruments", "Pressure Gauges"),
    "differential pressure": ("Pressure Instruments", "Differential Pressure Transmitters"),
    "pressure sensor": ("Pressure Instruments", None),

    # Temperature instruments
    "temperature sensor": ("Temperature Instruments", None),
    "temperature transmitter": ("Temperature Instruments", None),
    "thermocouple": ("Temperature Instruments", "Thermocouples"),
    "rtd": ("Temperature Instruments", "RTDs"),
    "infrared sensor": ("Temperature Instruments", "Infrared Sensors"),
    "thermometer": ("Temperature Instruments", None),

    # Flow instruments
    "flow meter": ("Flow Instruments", None),
    "flowmeter": ("Flow Instruments", None),
    "ultrasonic flow": ("Flow Instruments", "Ultrasonic Flow Meters"),
    "coriolis": ("Flow Instruments", None),
    "mass flow": ("Flow Instruments", None),
    "vortex": ("Flow Instruments", None),

    # Level instruments
    "level sensor": ("Level Instruments", None),
    "level transmitter": ("Level Instruments", None),
    "radar level": ("Level Instruments", "Radar Level Sensors"),
    "ultrasonic level": ("Level Instruments", "Ultrasonic Level Sensors"),
    "capacitance level": ("Level Instruments", "Capacitance Level Sensors"),

    # Control valves
    "control valve": ("Control Valves", None),
    "ball valve": ("Control Valves", "Ball Valves"),
    "globe valve": ("Control Valves", "Globe Valves"),
    "butterfly valve": ("Control Valves", None),

    # Analytical instruments
    "analyzer": ("Analytical Instruments", None),
    "ph meter": ("Analytical Instruments", None),
    "conductivity meter": ("Analytical Instruments", "Conductivity Meters"),
    "dissolved oxygen": ("Analytical Instruments", "Dissolved Oxygen Meters"),
    "gas chromatograph": ("Analytical Instruments", None),

    # Safety instruments
    "gas detector": ("Safety Instruments", "Gas Detectors"),
    "safety valve": ("Safety Instruments", "Safety Valves"),
    "flame detector": ("Safety Instruments", None),

    # Vibration instruments
    "vibration sensor": ("Vibration Measurement Instruments", "Vibration Sensors"),
    "vibrometer": ("Vibration Measurement Instruments", "Portable Vibrometers"),
    "accelerometer": ("Vibration Measurement Instruments", None),

    # Signal conditioning
    "transmitter": ("Signal Conditioning", "Transmitters"),
    "signal converter": ("Signal Conditioning", "Signal Converters"),
    "isolator": ("Signal Conditioning", None),
    "filter": ("Signal Conditioning", "Filters"),
}


class StrategyCSVFilter:
    """
    Simple CSV-based vendor strategy filter with lazy loading.

    Loads vendor strategy data from CSV and provides vendor filtering
    based on product category matching. Uses lazy loading to minimize memory usage.
    """

    def __init__(self, csv_path: str = None):
        """
        Initialize the strategy filter.

        Args:
            csv_path: Path to the strategy CSV file. Defaults to
                      instrumentation_procurement_strategy.csv in backend directory.
        """
        self.csv_path = csv_path or STRATEGY_CSV_PATH
        self._category_cache = {}  # Lazy cache: category -> rows
        self._vendor_cache = {}    # Lazy cache: vendor_lower -> rows
        self._max_cache_size = 10  # LRU: Keep at most 10 categories in memory
        self._all_data = None      # Full data (loaded on demand)

    def _load_all_data(self) -> List[Dict]:
        """Load all CSV data (used when full scan is needed)."""
        if self._all_data is not None:
            return self._all_data

        logger.info(f"[StrategyCSVFilter] Loading full strategy data from {self.csv_path}")

        self._all_data = []

        if not os.path.exists(self.csv_path):
            logger.warning(f"[StrategyCSVFilter] CSV file not found: {self.csv_path}")
            return self._all_data

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for idx, row in enumerate(reader):
                    # Normalize the row data
                    normalized_row = {
                        'vendor_id': row.get('vendor ID', '').strip(),
                        'vendor_name': row.get('vendor name', '').strip(),
                        'category': row.get('category', '').strip(),
                        'subcategory': row.get('subcategory', '').strip(),
                        'strategy': row.get('strategy', '').strip(),
                        'refinery': row.get('refinery', '').strip(),
                        'comments': row.get('additional comments', '').strip(),
                        'owner': row.get('owner name', '').strip()
                    }
                    self._all_data.append(normalized_row)

            logger.info(f"[StrategyCSVFilter] Loaded {len(self._all_data)} strategy records")

        except Exception as e:
            logger.error(f"[StrategyCSVFilter] Error loading CSV: {e}")
            self._all_data = []

        return self._all_data

    def _load_category_lazy(self, category: str, subcategory: str = None) -> List[Dict]:
        """
        Lazy load only rows matching a specific category (30x memory reduction).

        Args:
            category: Category name to load
            subcategory: Optional subcategory filter

        Returns:
            List of matching rows
        """
        cache_key = (category, subcategory)

        # Check if already cached
        if cache_key in self._category_cache:
            logger.debug(f"[StrategyCSVFilter] Using cached category: {category}/{subcategory}")
            return self._category_cache[cache_key]

        # Load only matching rows from CSV
        logger.debug(f"[StrategyCSVFilter] Lazy loading category: {category}/{subcategory}")
        matching_rows = []

        if not os.path.exists(self.csv_path):
            logger.warning(f"[StrategyCSVFilter] CSV file not found: {self.csv_path}")
            return matching_rows

        try:
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row in reader:
                    row_category = row.get('category', '').strip()
                    row_subcategory = row.get('subcategory', '').strip()

                    # Check if this row matches the requested category
                    if row_category == category:
                        if subcategory is None or row_subcategory == subcategory:
                            normalized_row = {
                                'vendor_id': row.get('vendor ID', '').strip(),
                                'vendor_name': row.get('vendor name', '').strip(),
                                'category': row_category,
                                'subcategory': row_subcategory,
                                'strategy': row.get('strategy', '').strip(),
                                'refinery': row.get('refinery', '').strip(),
                                'comments': row.get('additional comments', '').strip(),
                                'owner': row.get('owner name', '').strip()
                            }
                            matching_rows.append(normalized_row)

        except Exception as e:
            logger.error(f"[StrategyCSVFilter] Error loading category: {e}")
            return []

        # Cache with LRU eviction (keep max 10 categories)
        if len(self._category_cache) >= self._max_cache_size:
            # Evict oldest (first) entry
            oldest_key = next(iter(self._category_cache))
            del self._category_cache[oldest_key]
            logger.debug(f"[StrategyCSVFilter] Evicted category cache: {oldest_key}")

        self._category_cache[cache_key] = matching_rows
        logger.debug(f"[StrategyCSVFilter] Cached category: {category}/{subcategory} ({len(matching_rows)} rows)")

        return matching_rows

    def _map_product_to_category(self, product_type: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Map a product type to CSV category/subcategory.
        
        UPDATED: Uses vector store semantic matching as fallback.

        Args:
            product_type: Product type string (e.g., "pressure transmitter")

        Returns:
            Tuple of (category, subcategory) or (None, None) if no mapping
        """
        product_lower = product_type.lower().strip()

        # 1. Direct mapping (fast path)
        if product_lower in PRODUCT_TO_CATEGORY_MAP:
            return PRODUCT_TO_CATEGORY_MAP[product_lower]

        # 2. Fuzzy matching - check if any key is contained in product_type
        for key, (category, subcategory) in PRODUCT_TO_CATEGORY_MAP.items():
            if key in product_lower or product_lower in key:
                return (category, subcategory)

        # 3. Check for category keywords
        category_keywords = {
            "pressure": "Pressure Instruments",
            "temperature": "Temperature Instruments",
            "temp": "Temperature Instruments",
            "flow": "Flow Instruments",
            "level": "Level Instruments",
            "valve": "Control Valves",
            "analyz": "Analytical Instruments",
            "gas": "Safety Instruments",
            "safety": "Safety Instruments",
            "vibration": "Vibration Measurement Instruments",
            "signal": "Signal Conditioning"
        }

        for keyword, category in category_keywords.items():
            if keyword in product_lower:
                return (category, None)

        # 4. SEMANTIC FALLBACK: Use vector store for matching
        try:
            from .strategy_vector_store import get_strategy_vector_store
            
            vector_store = get_strategy_vector_store()
            category, subcategory = vector_store.match_product_to_category(product_type)
            
            if category:
                logger.info(f"[StrategyCSVFilter] Semantic match: '{product_type}' -> {category}/{subcategory}")
                return (category, subcategory)
                
        except ImportError:
            logger.debug("[StrategyCSVFilter] Vector store not available for semantic fallback")
        except Exception as e:
            logger.warning(f"[StrategyCSVFilter] Semantic fallback failed: {e}")

        return (None, None)

    def get_vendors_for_category(
        self,
        category: str,
        subcategory: str = None
    ) -> List[Dict[str, Any]]:
        """
        Get all vendors for a specific category/subcategory.

        Uses lazy loading to minimize memory usage - only loads requested categories.

        Args:
            category: CSV category (e.g., "Pressure Instruments")
            subcategory: Optional subcategory (e.g., "Pressure Gauges")

        Returns:
            List of vendor records with strategy information
        """
        # Use lazy loading to load only this category
        if subcategory:
            results = self._load_category_lazy(category, subcategory)
        else:
            # Load category with any subcategory
            results = self._load_category_lazy(category, None)

        return results

    def get_vendor_strategy(self, vendor_name: str, category: str = None) -> Dict[str, Any]:
        """
        Get strategy information for a specific vendor.

        Args:
            vendor_name: Vendor name to look up
            category: Optional category to filter by

        Returns:
            Dict with vendor strategy info, or empty dict if not found
        """
        # Load all data since we need to search by vendor across all categories
        all_data = self._load_all_data()

        vendor_lower = vendor_name.lower().strip()

        # Check for partial matches (e.g., "Emerson" matches "Emerson Electric Co.")
        matching_records = [
            row for row in all_data
            if vendor_lower in row['vendor_name'].lower() or row['vendor_name'].lower() in vendor_lower
        ]

        if not matching_records:
            return {}

        # If category specified, filter further
        if category:
            for record in matching_records:
                if record['category'] == category:
                    return record

        # Return first match
        return matching_records[0]

    def filter_vendors_for_product(
        self,
        product_type: str,
        available_vendors: List[str] = None,
        refinery: str = None
    ) -> Dict[str, Any]:
        """
        Filter and prioritize vendors for a product type based on CSV strategy.

        This is the main entry point for vendor filtering.

        Args:
            product_type: Product type (e.g., "pressure transmitter")
            available_vendors: List of available vendor names to filter
            refinery: Optional refinery to filter by

        Returns:
            {
                "filtered_vendors": [
                    {
                        "vendor": "Vendor Name",
                        "strategy": "Strategy statement",
                        "priority_score": 10,
                        "category_match": True,
                        "refinery_match": True
                    }
                ],
                "excluded_vendors": [],
                "category": "Pressure Instruments",
                "total_strategy_entries": 123
            }
        """
        # Map product to category
        category, subcategory = self._map_product_to_category(product_type)

        logger.info(f"[StrategyCSVFilter] Filtering for product: {product_type} -> "
                   f"Category: {category}, Subcategory: {subcategory}")

        # Get vendors in this category (lazy loaded)
        category_vendors = self.get_vendors_for_category(category, subcategory) if category else []

        # Build vendor strategy lookup
        vendor_strategies = {}
        for record in category_vendors:
            vname_lower = record['vendor_name'].lower()
            if vname_lower not in vendor_strategies:
                vendor_strategies[vname_lower] = []
            vendor_strategies[vname_lower].append(record)

        # Filter and score available vendors
        filtered = []
        excluded = []

        # If no available vendors specified, use all vendors from the category
        if not available_vendors and category_vendors:
            available_vendors = list(set(v['vendor_name'] for v in category_vendors))

        vendors_to_check = available_vendors if available_vendors else []

        for vendor in vendors_to_check:
            vendor_lower = vendor.lower().strip()

            # Find matching strategy records
            matching_records = []
            for vname, records in vendor_strategies.items():
                if vendor_lower in vname or vname in vendor_lower:
                    matching_records.extend(records)

            if not matching_records:
                # No strategy entry for this vendor in this category
                # Still include them but with lower priority
                filtered.append({
                    "vendor": vendor,
                    "strategy": "No specific strategy defined",
                    "priority_score": 5,
                    "category_match": False,
                    "refinery_match": False,
                    "comments": ""
                })
                continue

            # Calculate priority score based on strategy
            best_record = matching_records[0]
            priority_score = self._calculate_priority_score(best_record, refinery)

            # Check for refinery match
            refinery_match = False
            if refinery:
                for record in matching_records:
                    if refinery.lower() in record['refinery'].lower():
                        best_record = record
                        refinery_match = True
                        priority_score += 5  # Bonus for refinery match
                        break

            filtered.append({
                "vendor": vendor,
                "strategy": best_record['strategy'],
                "priority_score": priority_score,
                "category_match": True,
                "refinery_match": refinery_match,
                "comments": best_record['comments'],
                "vendor_id": best_record['vendor_id']
            })

        # Sort by priority score (descending)
        filtered.sort(key=lambda x: x['priority_score'], reverse=True)

        logger.info(f"[StrategyCSVFilter] Filtered {len(filtered)} vendors for {product_type}")

        # Get total strategy entries count
        all_data = self._load_all_data()
        total_entries = len(all_data) if all_data else 0

        return {
            "filtered_vendors": filtered,
            "excluded_vendors": excluded,
            "category": category,
            "subcategory": subcategory,
            "total_strategy_entries": total_entries
        }

    def _calculate_priority_score(self, record: Dict[str, Any], refinery: str = None) -> int:
        """
        Calculate priority score based on strategy statement.

        Higher score = higher priority.

        Args:
            record: Strategy record from CSV
            refinery: Optional refinery context

        Returns:
            Priority score (0-100)
        """
        strategy = record.get('strategy', '').lower()
        score = 10  # Base score

        # High priority strategies
        high_priority = [
            "long-term partnership",
            "framework agreement",
            "preferred supplier",
            "strategic partner"
        ]

        # Medium priority strategies
        medium_priority = [
            "dual sourcing",
            "multi-source",
            "standardization",
            "bundling",
            "volume discount",
            "bulk purchasing"
        ]

        # Lower priority strategies
        lower_priority = [
            "spend analysis",
            "cost optimization",
            "evaluation",
            "consignment"
        ]

        for phrase in high_priority:
            if phrase in strategy:
                score += 15
                break

        for phrase in medium_priority:
            if phrase in strategy:
                score += 10
                break

        for phrase in lower_priority:
            if phrase in strategy:
                score += 5
                break

        # Bonus for specific positive indicators
        if "sustainability" in strategy or "green" in strategy:
            score += 3
        if "lifecycle" in strategy or "life-cycle" in strategy:
            score += 3
        if "technology upgrade" in strategy or "digitalization" in strategy:
            score += 3

        return min(score, 100)  # Cap at 100

    def get_all_vendors(self) -> List[str]:
        """Get list of all unique vendor names from CSV."""
        all_data = self._load_all_data()
        return list(set(row['vendor_name'] for row in all_data)) if all_data else []

    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories from CSV."""
        all_data = self._load_all_data()
        return list(set(row['category'] for row in all_data)) if all_data else []


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Singleton instance
_strategy_filter_instance = None


def get_strategy_filter() -> StrategyCSVFilter:
    """Get or create singleton StrategyCSVFilter instance."""
    global _strategy_filter_instance
    if _strategy_filter_instance is None:
        _strategy_filter_instance = StrategyCSVFilter()
    return _strategy_filter_instance


def filter_vendors_by_strategy(
    product_type: str,
    available_vendors: List[str] = None,
    refinery: str = None
) -> Dict[str, Any]:
    """
    Convenience function to filter vendors using CSV strategy.

    Args:
        product_type: Product type (e.g., "pressure transmitter")
        available_vendors: Optional list of vendors to filter
        refinery: Optional refinery context

    Returns:
        Filter result with filtered_vendors list
    """
    filter_instance = get_strategy_filter()
    return filter_instance.filter_vendors_for_product(
        product_type=product_type,
        available_vendors=available_vendors,
        refinery=refinery
    )


def get_vendor_strategy_info(vendor_name: str, product_type: str = None) -> Dict[str, Any]:
    """
    Get strategy information for a specific vendor.

    Args:
        vendor_name: Vendor name
        product_type: Optional product type for category context

    Returns:
        Strategy info dict
    """
    filter_instance = get_strategy_filter()

    category = None
    if product_type:
        category, _ = filter_instance._map_product_to_category(product_type)

    return filter_instance.get_vendor_strategy(vendor_name, category)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 80)
    print("TESTING STRATEGY CSV FILTER")
    print("=" * 80)

    strategy_filter = StrategyCSVFilter()

    # Test 1: Get all categories
    print("\n[Test 1] All Categories:")
    categories = strategy_filter.get_all_categories()
    for cat in categories:
        print(f"  - {cat}")

    # Test 2: Get all vendors
    print(f"\n[Test 2] Total Vendors: {len(strategy_filter.get_all_vendors())}")

    # Test 3: Filter for specific product types
    test_products = [
        "pressure transmitter",
        "flow meter",
        "temperature sensor",
        "control valve",
        "gas detector"
    ]

    test_vendors = ["Emerson", "ABB", "Siemens", "Yokogawa", "Honeywell", "WIKA"]

    for product in test_products:
        print(f"\n[Test 3] Filtering for: {product}")
        print("-" * 40)

        result = strategy_filter.filter_vendors_for_product(
            product_type=product,
            available_vendors=test_vendors
        )

        print(f"  Category: {result['category']}")
        print(f"  Filtered vendors:")
        for v in result['filtered_vendors'][:5]:
            print(f"    - {v['vendor']}: Score={v['priority_score']}, "
                  f"Strategy='{v['strategy'][:50]}...'")

    # Test 4: Get specific vendor strategy
    print("\n[Test 4] Emerson Strategy for Pressure Instruments:")
    info = get_vendor_strategy_info("Emerson", "pressure transmitter")
    if info:
        print(f"  Strategy: {info.get('strategy', 'N/A')}")
        print(f"  Comments: {info.get('comments', 'N/A')}")
    else:
        print("  No strategy found")

    print("\n" + "=" * 80)
