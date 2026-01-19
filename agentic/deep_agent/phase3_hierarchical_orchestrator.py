# agentic/deep_agent/phase3_hierarchical_orchestrator.py
# =============================================================================
# PHASE 3: MULTI-LEVEL PARALLEL HIERARCHICAL ORCHESTRATOR
# =============================================================================
#
# Purpose: Orchestrate multi-level parallel specification enrichment with:
# - Level 0: Stratification (instruments vs accessories)
# - Level 1-2: Product type grouping & sub-agent creation
# - Level 3A: User specs extraction with mandatory locking
# - Level 3B: Domain-specific LLM specs generation
# - Level 3C: Per-product-type RAG standards lookup
# - Level 4: Merge & store in memory
# - Level 5: Consolidation & final merge
#
# Key Features:
# - TRUE multi-level parallelization (not just product-level)
# - Mandatory spec preservation throughout pipeline
# - Domain-specific LLM prompts (specialized accuracy)
# - Per-product-type RAG queries (reduced search space)
# - Deep Agent memory integration
# - Comprehensive performance metrics
#
# =============================================================================

import logging
import time
import os
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from llm_fallback import create_llm_with_fallback
from .phase3_memory_manager import Phase3MemoryManager
from .optimized_parallel_agent import (
    get_shared_llm,
    extract_user_specs_with_shared_llm,
    generate_llm_specs_with_shared_llm
)

load_dotenv()
logger = logging.getLogger(__name__)


# =============================================================================
# LEVEL 0: STRATIFICATION
# =============================================================================

class StratificationPhase:
    """Split items into instruments and accessories"""

    @staticmethod
    def stratify(items: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Stratify items into instruments and accessories.

        Args:
            items: List of all items (mixed)

        Returns:
            Tuple of (instruments, accessories)
        """
        instruments = []
        accessories = []

        for item in items:
            item_type = item.get("type", "instrument").lower()

            if item_type == "instrument":
                instruments.append(item)
            elif item_type == "accessory":
                accessories.append(item)
            else:
                # Default to instrument
                instruments.append(item)

        logger.info(
            f"[STRATIFY] Stratified {len(items)} items: "
            f"{len(instruments)} instruments, {len(accessories)} accessories"
        )

        return instruments, accessories


# =============================================================================
# LEVEL 1-2: PRODUCT TYPE GROUPING
# =============================================================================

class ProductTypeGrouping:
    """Group items by product type for specialized handling"""

    INSTRUMENT_TYPES = {
        "temperature_sensor": {
            "keywords": ["temperature", "thermocouple", "rtd", "thermometer", "sensor"],
            "standards_domains": ["temperature", "safety", "calibration", "installation"]
        },
        "pressure_transmitter": {
            "keywords": ["pressure", "transmitter", "gauge", "transducer"],
            "standards_domains": ["pressure", "safety", "calibration", "installation"]
        },
        "flow_meter": {
            "keywords": ["flow", "flowmeter", "coriolis", "ultrasonic", "vortex", "turbine"],
            "standards_domains": ["flow", "safety", "calibration", "installation"]
        },
        "level_transmitter": {
            "keywords": ["level", "tank", "radar", "ultrasonic", "displacer"],
            "standards_domains": ["level", "safety", "calibration", "installation"]
        },
        "analyzer": {
            "keywords": ["analyzer", "ph", "conductivity", "oxygen", "gas", "chromatograph"],
            "standards_domains": ["analytical", "safety", "calibration", "installation"]
        },
        "control_valve": {
            "keywords": ["valve", "control valve", "actuator", "positioner", "solenoid"],
            "standards_domains": ["valves", "control", "safety", "installation"]
        }
    }

    ACCESSORY_TYPES = {
        "thermowell": {
            "keywords": ["thermowell", "insertion", "pocket"],
            "standards_domains": ["temperature", "accessories", "installation", "calibration"]
        },
        "cable_gland": {
            "keywords": ["cable gland", "gland", "connector", "entry"],
            "standards_domains": ["accessories", "installation", "safety"]
        },
        "enclosure": {
            "keywords": ["enclosure", "junction box", "cabinet", "housing"],
            "standards_domains": ["accessories", "safety", "installation"]
        },
        "mounting_kit": {
            "keywords": ["mounting", "bracket", "rail"],
            "standards_domains": ["accessories", "installation"]
        },
        "adapter": {
            "keywords": ["adapter", "converter", "fitting"],
            "standards_domains": ["accessories", "installation"]
        }
    }

    @staticmethod
    def classify_product_type(
        item: Dict,
        type_map: Dict,
        default: str = "generic"
    ) -> str:
        """
        Classify item into a product type.

        Args:
            item: Item to classify
            type_map: Type definitions with keywords
            default: Default type if no match

        Returns:
            Product type string
        """
        name = (item.get("name") or item.get("product_name", "")).lower()
        category = item.get("category", "").lower()
        search_text = f"{name} {category}".lower()

        for product_type, config in type_map.items():
            keywords = config.get("keywords", [])
            if any(keyword in search_text for keyword in keywords):
                return product_type

        return default

    @classmethod
    def group_by_product_type(
        cls,
        instruments: List[Dict],
        accessories: List[Dict]
    ) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
        """
        Group instruments and accessories by product type.

        Returns:
            Tuple of (instruments_grouped, accessories_grouped)
        """
        instruments_grouped: Dict[str, List[Dict]] = {}
        accessories_grouped: Dict[str, List[Dict]] = {}

        # Group instruments
        for item in instruments:
            product_type = cls.classify_product_type(item, cls.INSTRUMENT_TYPES, "generic_instrument")

            if product_type not in instruments_grouped:
                instruments_grouped[product_type] = []

            instruments_grouped[product_type].append(item)

        # Group accessories
        for item in accessories:
            product_type = cls.classify_product_type(item, cls.ACCESSORY_TYPES, "generic_accessory")

            if product_type not in accessories_grouped:
                accessories_grouped[product_type] = []

            accessories_grouped[product_type].append(item)

        logger.info(
            f"[GROUP] Grouped {len(instruments)} instruments into {len(instruments_grouped)} types, "
            f"{len(accessories)} accessories into {len(accessories_grouped)} types"
        )

        return instruments_grouped, accessories_grouped

    @classmethod
    def get_standards_domains(
        cls,
        product_type: str,
        is_accessory: bool = False
    ) -> List[str]:
        """Get relevant standards domains for a product type"""
        type_map = cls.ACCESSORY_TYPES if is_accessory else cls.INSTRUMENT_TYPES

        if product_type in type_map:
            return type_map[product_type].get("standards_domains", [])

        return ["safety", "calibration", "installation"]  # defaults


# =============================================================================
# LEVEL 3B: DOMAIN-SPECIFIC LLM PROMPTS
# =============================================================================

class DomainSpecificLLMPhase:
    """Generate domain-specific LLM specs for each product type"""

    DOMAIN_PROMPTS = {
        "temperature_sensor": """
You are an Expert in Temperature Measurement Instrumentation.
Generate technical specifications for TEMPERATURE SENSORS/TRANSMITTERS based on the context.

CRITICAL: Focus on these key specifications:
- Temperature Range: min and max operating temperature (-50°C to +500°C typical range)
- Sensor Type: Thermocouple (K/J/T/E), RTD (Pt100/Pt1000), Resistance thermometer
- Accuracy: ±0.5°C, ±0.1%, ±1% of Full Scale typical values
- Response Time: Time constant (tau) in seconds (0.5s to 10s typical)
- Output Signal: mA (4-20mA standard), digital (Modbus, HART), analog (0-10V)
- Connection: Head-mounted, Transmitter module, DIN connector, M16 connector
- Material: Stainless Steel 316L, Hastelloy, Invar sheath
- Process Connection: 1/2" NPT, G1/2", 1/4" NPT, flange sizes
- Wetted Materials: Stainless, Hastelloy
- Hazardous Area: ATEX Cat 2G, IECEx Zone 1/2, SIL ratings
- Communication Protocol: HART, Modbus RTU, Profibus PA, analog

PRODUCT TYPE: {product_type}
CONTEXT: {context}

Return ONLY clean technical values. NO descriptions, NO narratives.
Format: JSON with flat key-value pairs.
""",

        "pressure_transmitter": """
You are an Expert in Pressure Instrumentation.
Generate technical specifications for PRESSURE TRANSMITTERS based on context.

CRITICAL: Focus on these key specifications:
- Pressure Range: 0-10 bar, 0-250 bar, -1 to +100 bar (gauge/absolute/differential)
- Measurement Type: Gauge (relative to atmosphere), Absolute, Differential
- Accuracy: ±0.5% FS, ±1% FS, ±2% FS (common industrial values)
- Output Signal: 4-20mA standard, 0-10V analog, HART, Modbus, digital
- Flush Diaphragm: Yes/No (for clogging-prone applications)
- Vented Gauge: Yes/No (atmospheric reference)
- Material: Stainless Steel 316L, Hastelloy (aggressive media)
- Process Connection: 1/4" NPT, M14x2, SAE flange, customized
- Temperature Compensation: Yes/No, range
- Ambient Temperature: -20 to +70°C typical
- Protection Rating: IP67, IP69K
- Hazardous Area: ATEX, IECEx, Zone classification, SIL rating
- Response Time: Milliseconds (typically 50-200ms)
- Burst Pressure: 1.5x-3x of max rated pressure

PRODUCT TYPE: {product_type}
CONTEXT: {context}

Return ONLY clean technical values. NO descriptions, NO narratives.
Format: JSON with flat key-value pairs.
""",

        "flow_meter": """
You are an Expert in Flow Measurement Instrumentation.
Generate technical specifications for FLOW METERS based on context.

CRITICAL: Focus on these key specifications:
- Flow Range: Min to max flow rates (L/min, m³/h, kg/h)
- Measurement Type: Mass flow, volumetric flow, standard flow
- Technology: Coriolis mass flow, ultrasonic, vortex shedding, turbine, differential pressure
- Accuracy: ±0.5% of reading, ±1% FS, ±2% FS
- Output Signal: 4-20mA, pulse output (Hz/counts), Modbus, HART
- Temperature Range: -40 to +150°C (liquid), -40 to +200°C (gas)
- Pressure Range: Max operating pressure (bar)
- Material: Stainless Steel 316L, Hastelloy, PFA for corrosive media
- Connection Size: 1/2", 1", 2", 3", 4" (DN values)
- Connection Type: NPT, SAE flange, Wafer, Tri-clamp
- Density: Medium density (liquid, gas)
- Viscosity: Kinematic viscosity range (cSt)
- Response Time: Milliseconds
- Hazardous Area: ATEX, IECEx, Zone classification

PRODUCT TYPE: {product_type}
CONTEXT: {context}

Return ONLY clean technical values. NO descriptions, NO narratives.
Format: JSON with flat key-value pairs.
""",

        "level_transmitter": """
You are an Expert in Level Measurement Instrumentation.
Generate technical specifications for LEVEL TRANSMITTERS based on context.

CRITICAL: Focus on these key specifications:
- Level Range: Min to max measurement height (m, ft)
- Technology: Radar, ultrasonic, float, displacer, capacitive, guided wave
- Measurement Type: Continuous, point level (switch)
- Accuracy: ±5mm, ±10mm, ±2% of range
- Output Signal: 4-20mA, digital output, Modbus, HART
- Temperature Range: -40 to +120°C (typical liquid), -40 to +150°C (gas)
- Pressure Range: -1 to +10 bar typical
- Vessel Type: Open tank, closed tank, pressurized vessel
- Medium: Liquid (water, oil, chemical), slurry, powder
- Density/Dielectric: For capacitive/radar technologies
- Deadband: Dead zone at top/bottom
- Connection Size: 1/2", 1", 2" NPT/BSP
- Material: Stainless Steel 316L, Hastelloy
- Hazardous Area: ATEX, IECEx certification
- Response Time: Milliseconds

PRODUCT TYPE: {product_type}
CONTEXT: {context}

Return ONLY clean technical values. NO descriptions, NO narratives.
Format: JSON with flat key-value pairs.
""",

        "generic_instrument": """
You are an Industrial Instrumentation Expert.
Generate technical specifications for the given instrument type.

Key areas to cover:
- Measurement range, accuracy, and repeatability
- Output signals and communication protocols
- Environmental operating conditions
- Material composition and construction
- Connection types and sizes
- Safety certifications and approvals
- Response time and stability
- Calibration requirements

PRODUCT TYPE: {product_type}
CONTEXT: {context}

Return ONLY clean technical values. NO descriptions, NO narratives.
Format: JSON with flat key-value pairs.
""",

        # Accessories
        "thermowell": """
You are an Expert in Temperature Accessories.
Generate technical specifications for THERMOWELLS based on context.

CRITICAL: Focus on:
- Length: Insertion length (mm, typically 50-1000mm)
- Diameter: Bore size (typically 3-8mm)
- Material: Stainless Steel, Hastelloy
- Connection: Process thread (1/2" NPT, G1/2", SAE flange)
- Design: Solid bore, separate sheath, pocket design
- Temperature Rating: Max temperature the thermowell can handle
- Pressure Rating: Max process pressure (bar)
- Fill Material: Air, silicone oil (if filled)
- Surface Finish: As-welded, machined, pickled

PRODUCT TYPE: {product_type}
CONTEXT: {context}

Return ONLY clean technical values.
""",

        "cable_gland": """
You are an Expert in Cable & Connectivity Accessories.
Generate technical specifications for CABLE GLANDS based on context.

CRITICAL: Focus on:
- Cable Diameter Range: min-max cable size (mm, typical 5-14mm)
- Thread Size: M16, M20, M25, M32 (metric) or NPT equivalents
- Material: Brass, stainless steel 316L
- Protection Rating: IP67, IP68, IP69K
- Strain Relief: Integrated or separate
- Temperature Range: -40 to +100°C typical
- EMC/Shielding: Yes/No for EMI protection
- Hazardous Area: ATEX, IECEx if required

PRODUCT TYPE: {product_type}
CONTEXT: {context}

Return ONLY clean technical values.
"""
    }

    @classmethod
    def generate_llm_specs_for_product_type(
        cls,
        product_type: str,
        category: str,
        context: str,
        llm: Any = None
    ) -> Dict[str, Any]:
        """
        Generate LLM specs using domain-specific prompt.

        Args:
            product_type: Type of product (e.g., "temperature_sensor")
            category: Product category
            context: Additional context
            llm: Optional LLM instance (uses shared if None)

        Returns:
            Dictionary of generated specifications
        """
        llm = llm or get_shared_llm(temperature=0.3)

        # Get domain-specific prompt (fallback to generic)
        prompt_template = cls.DOMAIN_PROMPTS.get(
            product_type,
            cls.DOMAIN_PROMPTS.get("generic_instrument")
        )

        try:
            prompt = ChatPromptTemplate.from_template(prompt_template)
            parser = JsonOutputParser()
            chain = prompt | llm | parser

            result = chain.invoke({
                "product_type": product_type,
                "category": category,
                "context": context
            })

            specs = result.get("specifications", {})

            # Clean and validate
            clean_specs = {}
            for key, value in specs.items():
                if isinstance(value, dict):
                    clean_specs[key] = value
                else:
                    clean_specs[key] = {"value": str(value), "confidence": 0.7}

            logger.debug(f"[LLM_DOMAIN] Generated {len(clean_specs)} specs for {product_type}")

            return clean_specs

        except Exception as e:
            logger.error(f"[LLM_DOMAIN] Failed for {product_type}: {e}")
            return {}


# =============================================================================
# LEVEL 3: PER-PRODUCT-TYPE PROCESSOR
# =============================================================================

class Level3ProductTypeProcessor:
    """Process a single product type through all enrichment sources (3A, 3B, 3C)"""

    def __init__(
        self,
        memory: Phase3MemoryManager,
        llm: Any = None,
        session_id: str = "default"
    ):
        self.memory = memory
        self.llm = llm or get_shared_llm()
        self.session_id = session_id

    def process_product_type_group(
        self,
        product_type: str,
        items: List[Dict],
        user_input: str,
        is_accessory: bool = False,
        standards_domains: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Process all items of a specific product type through all levels.

        Args:
            product_type: Product type identifier
            items: All items of this product type
            user_input: Original user input
            is_accessory: Whether this is an accessory type
            standards_domains: Relevant standards domains for RAG

        Returns:
            Dict mapping item_id to enriched specs
        """
        logger.info(
            f"[LEVEL3] Processing {len(items)} items of type {product_type} "
            f"(accessory={is_accessory})"
        )

        start_time = time.time()
        results = {}

        # Process each item in this product type group
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(
                    self._process_single_item,
                    item,
                    user_input,
                    product_type,
                    is_accessory,
                    standards_domains
                ): item
                for item in items
            }

            for future in as_completed(futures):
                item = futures[future]
                item_id = f"{self.session_id}_{item.get('number', 0)}"

                try:
                    result = future.result()
                    results[item_id] = result
                except Exception as e:
                    logger.error(
                        f"[LEVEL3] Failed to process {item.get('name', 'Unknown')}: {e}"
                    )
                    results[item_id] = {}

        elapsed = time.time() - start_time
        logger.info(
            f"[LEVEL3] Completed {product_type} ({len(items)} items) in {elapsed:.2f}s"
        )

        return results

    def _process_single_item(
        self,
        item: Dict,
        user_input: str,
        product_type: str,
        is_accessory: bool,
        standards_domains: Optional[List[str]]
    ) -> Dict:
        """
        Process a single item through all enrichment levels.

        Levels:
        - 3A: User specs (extract & lock)
        - 3B: LLM specs (domain-specific)
        - 3C: Standards specs (RAG)
        - 4: Merge & store in memory
        """
        item_id = f"{self.session_id}_{item.get('number', 0)}"
        item_name = item.get("name") or item.get("product_name", "Unknown")
        item_type = "accessory" if is_accessory else "instrument"

        logger.debug(f"[LEVEL3_ITEM] Processing {item_name} ({product_type})")

        # Initialize memory for this item
        self.memory.initialize_item(
            item_id=item_id,
            item_name=item_name,
            item_type=item_type,
            product_type=product_type
        )

        # ===== LEVEL 3A: User specs extraction & locking =====
        user_specs = extract_user_specs_with_shared_llm(
            user_input=user_input,
            product_type=product_type,
            sample_input=item.get("sample_input", ""),
            llm=self.llm
        )

        user_specs_dict = user_specs.get("specifications", {})

        # CRITICAL: Lock mandatory specs immediately
        self.memory.lock_mandatory_specs(
            item_id=item_id,
            mandatory_specs=user_specs_dict
        )

        # ===== LEVEL 3B: Domain-specific LLM specs =====
        llm_specs = DomainSpecificLLMPhase.generate_llm_specs_for_product_type(
            product_type=product_type,
            category=item.get("category", ""),
            context=item.get("sample_input", ""),
            llm=self.llm
        )

        # Store with locking
        filtered_llm_specs = self.memory.store_enrichment_specs(
            item_id=item_id,
            source="llm_generated",
            specs=llm_specs
        )

        # ===== LEVEL 3C: Per-product-type RAG standards =====
        # TODO: Integrate with vector store for RAG
        # For now, return empty dict (will be enhanced)
        standards_specs = {}

        # Store standards specs
        filtered_standards_specs = self.memory.store_enrichment_specs(
            item_id=item_id,
            source="standards",
            specs=standards_specs
        )

        # ===== LEVEL 4: Merge & store =====
        merged_specs = self.memory.merge_all_specs(item_id)

        return {
            "item_id": item_id,
            "item_name": item_name,
            "product_type": product_type,
            "user_specs": user_specs_dict,
            "llm_specs": filtered_llm_specs,
            "standards_specs": filtered_standards_specs,
            "merged_specs": merged_specs
        }


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

def run_phase3_hierarchical_enrichment(
    items: List[Dict[str, Any]],
    user_input: str,
    session_id: Optional[str] = None,
    domain_context: Optional[str] = None,
    safety_requirements: Optional[Dict[str, Any]] = None,
    max_parallel_products: int = 5
) -> Dict[str, Any]:
    """
    Run Phase 3 multi-level parallel hierarchical enrichment.

    Architecture:
    - Level 0: Stratification (instruments vs accessories)
    - Level 1-2: Product type grouping
    - Level 3A: User specs extraction & locking
    - Level 3B: Domain-specific LLM enrichment
    - Level 3C: Per-product-type RAG standards
    - Level 4: Merge & store in memory
    - Level 5: Consolidation

    Args:
        items: List of product items
        user_input: Original user input
        session_id: Session identifier
        domain_context: Domain context (e.g., "Oil & Gas")
        safety_requirements: Safety requirements dict
        max_parallel_products: Max concurrent threads

    Returns:
        Dict with enriched items and metadata
    """
    start_time = time.time()
    session_id = session_id or f"phase3-{uuid4().hex[:8]}"

    logger.info("=" * 70)
    logger.info("[PHASE3] MULTI-LEVEL HIERARCHICAL ENRICHMENT")
    logger.info(f"[PHASE3] Session: {session_id}")
    logger.info(f"[PHASE3] Items: {len(items)}")
    logger.info("=" * 70)

    try:
        # ===================================================================
        # LEVEL 0: STRATIFICATION
        # ===================================================================
        level0_start = time.time()
        logger.info("[PHASE3] LEVEL 0: Stratification...")

        instruments, accessories = StratificationPhase.stratify(items)

        level0_time = time.time() - level0_start
        logger.info(f"[PHASE3] Level 0 done in {level0_time:.2f}s")

        # ===================================================================
        # LEVEL 1-2: PRODUCT TYPE GROUPING
        # ===================================================================
        level1_start = time.time()
        logger.info("[PHASE3] LEVEL 1-2: Product type grouping...")

        inst_grouped, acc_grouped = ProductTypeGrouping.group_by_product_type(
            instruments,
            accessories
        )

        level1_time = time.time() - level1_start
        logger.info(
            f"[PHASE3] Level 1-2 done in {level1_time:.2f}s "
            f"({len(inst_grouped)} instrument types, {len(acc_grouped)} accessory types)"
        )

        # ===================================================================
        # CREATE MEMORY MANAGER
        # ===================================================================
        memory = Phase3MemoryManager(session_id=session_id, max_items=1000)

        # ===================================================================
        # LEVEL 3: PARALLEL PRODUCT TYPE PROCESSING
        # ===================================================================
        level3_start = time.time()
        logger.info("[PHASE3] LEVEL 3: Parallel product type enrichment...")

        processor = Level3ProductTypeProcessor(memory=memory, session_id=session_id)

        all_results = {}

        # Process instrument types in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            inst_futures = {
                executor.submit(
                    processor.process_product_type_group,
                    product_type,
                    items_list,
                    user_input,
                    False,
                    ProductTypeGrouping.get_standards_domains(product_type, False)
                ): product_type
                for product_type, items_list in inst_grouped.items()
            }

            acc_futures = {
                executor.submit(
                    processor.process_product_type_group,
                    product_type,
                    items_list,
                    user_input,
                    True,
                    ProductTypeGrouping.get_standards_domains(product_type, True)
                ): product_type
                for product_type, items_list in acc_grouped.items()
            }

            # Collect results
            for future in as_completed(list(inst_futures) + list(acc_futures)):
                result = future.result()
                all_results.update(result)

        level3_time = time.time() - level3_start
        logger.info(f"[PHASE3] Level 3 done in {level3_time:.2f}s ({len(all_results)} items processed)")

        # ===================================================================
        # LEVEL 5: CONSOLIDATION
        # ===================================================================
        level5_start = time.time()
        logger.info("[PHASE3] LEVEL 5: Consolidation...")

        # Build final enriched items
        final_items = []
        for item in items:
            item_id = f"{session_id}_{item.get('number', 0)}"

            if item_id in all_results:
                enriched_item = item.copy()

                # Get merged specs from memory
                item_memory = memory.get_item_memory(item_id)

                if item_memory:
                    merged_specs_dict = memory.get_merged_specs(item_id)
                    merged_specs_with_meta = memory.get_merged_specs_with_metadata(item_id)

                    enriched_item["user_specified_specs"] = {
                        k: v.value for k, v in item_memory.user_specified_specs.items()
                    }
                    enriched_item["llm_generated_specs"] = {
                        k: v.value for k, v in item_memory.llm_generated_specs.items()
                    }
                    enriched_item["standards_specifications"] = {
                        k: v.value for k, v in item_memory.standards_specs.items()
                    }
                    enriched_item["combined_specifications"] = merged_specs_with_meta
                    enriched_item["specifications"] = merged_specs_dict

                    enriched_item["standards_info"] = {
                        "enrichment_status": "success",
                        "locked_specs": list(item_memory.locked_keys),
                        "mandatory_specs_count": len(item_memory.mandatory_specs),
                        "user_specs_count": len(item_memory.user_specified_specs),
                        "llm_specs_count": len(item_memory.llm_generated_specs),
                        "standards_specs_count": len(item_memory.standards_specs),
                        "total_merged": len(merged_specs_dict)
                    }

                final_items.append(enriched_item)
            else:
                # Fallback: return original item
                final_items.append(item)

        level5_time = time.time() - level5_start
        total_time = time.time() - start_time

        logger.info(f"[PHASE3] Level 5 done in {level5_time:.2f}s")

        # ===================================================================
        # FINAL REPORTING
        # ===================================================================
        logger.info("=" * 70)
        logger.info("[PHASE3] COMPLETE")
        logger.info(f"[PHASE3] Level 0 (Stratification):           {level0_time:.2f}s")
        logger.info(f"[PHASE3] Level 1-2 (Grouping):               {level1_time:.2f}s")
        logger.info(f"[PHASE3] Level 3 (Enrichment):               {level3_time:.2f}s")
        logger.info(f"[PHASE3] Level 5 (Consolidation):            {level5_time:.2f}s")
        logger.info(f"[PHASE3] TOTAL:                              {total_time:.2f}s")
        logger.info(f"[PHASE3] Items processed: {len(final_items)}")
        logger.info("=" * 70)

        memory_stats = memory.get_memory_stats()
        logger.info(f"[PHASE3] Memory stats: {memory_stats}")

        return {
            "success": True,
            "items": final_items,
            "metadata": {
                "total_items": len(final_items),
                "processing_time_ms": int(total_time * 1000),
                "session_id": session_id,
                "phase_times": {
                    "stratification_s": round(level0_time, 2),
                    "grouping_s": round(level1_time, 2),
                    "enrichment_s": round(level3_time, 2),
                    "consolidation_s": round(level5_time, 2),
                    "total_s": round(total_time, 2)
                },
                "memory_stats": memory_stats
            },
            "memory": memory
        }

    except Exception as e:
        logger.error(f"[PHASE3] ERROR: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "items": items,
            "metadata": {"session_id": session_id}
        }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "run_phase3_hierarchical_enrichment",
    "StratificationPhase",
    "ProductTypeGrouping",
    "DomainSpecificLLMPhase",
    "Level3ProductTypeProcessor"
]
