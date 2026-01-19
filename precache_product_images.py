# precache_product_images.py
# =============================================================================
# PRE-CACHE GENERIC PRODUCT TYPE IMAGES
# =============================================================================
#
# Run this script to pre-generate and cache all generic product type images.
# This prevents rate limit issues during live usage since all images will be cached.
#
# Usage:
#   python precache_product_images.py           # Cache all known product types
#   python precache_product_images.py --check   # Check which images are missing
#   python precache_product_images.py --force   # Regenerate all images
#
# =============================================================================

import sys
import os
import time
import argparse
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# =============================================================================
# KNOWN PRODUCT TYPES TO PRE-CACHE
# =============================================================================

PRODUCT_TYPES_TO_CACHE = [
    # Temperature Instruments
    "Thermocouple",
    "Multi-point Thermocouple",
    "Surface Mount Thermocouple",
    "RTD Temperature Sensor",
    "Temperature Transmitter",
    "Multi-channel Temperature Transmitter",
    "Thermowell",
    "High-Pressure Thermowell",
    "Temperature Indicator",
    
    # Pressure Instruments
    "Pressure Transmitter",
    "Differential Pressure Transmitter",
    "Pressure Gauge",
    "Pressure Switch",
    "Pressure Sensor",
    
    # Flow Instruments
    "Flow Meter",
    "Magnetic Flow Meter",
    "Ultrasonic Flow Meter",
    "Coriolis Flow Meter",
    "Vortex Flow Meter",
    "Turbine Flow Meter",
    "Flow Transmitter",
    
    # Level Instruments
    "Level Transmitter",
    "Radar Level Transmitter",
    "Ultrasonic Level Transmitter",
    "Level Switch",
    "Level Indicator",
    
    # Analyzers
    "pH Analyzer",
    "Conductivity Analyzer",
    "Dissolved Oxygen Analyzer",
    "Gas Analyzer",
    
    # Valves & Actuators
    "Control Valve",
    "Ball Valve",
    "Gate Valve",
    "Globe Valve",
    "Butterfly Valve",
    "Actuator",
    "Pneumatic Actuator",
    "Electric Actuator",
    "Positioner",
    
    # Electrical & Enclosures
    "Junction Box",
    "Multi-point Thermocouple Junction Box",
    "Terminal Box",
    "Industrial Enclosure",
    "Explosion-proof Enclosure",
    "Control Panel",
    
    # Power & Distribution
    "Power Supply",
    "Industrial Power Supply",
    "Redundant Power Supply",
    "Power Distribution Terminal Block",
    "UPS System",
    
    # Cables & Wiring
    "Thermocouple Extension Wire",
    "Thermocouple Extension Cable",
    "Instrumentation Signal Cable",
    "Cable Gland",
    "Cable Tray",
    "Cable Ties",
    
    # Mounting & Hardware
    "Mounting Hardware Kit",
    "Mounting Bracket",
    "Transmitter Mounting Bracket",
    "Surface Sensor Mounting Clamp Kit",
    "Pipe Mounting Bracket",
    "Wall Mounting Bracket",
    
    # Accessories
    "Gaskets and Bolting Kit",
    "Terminal Blocks",
    "Signal Cable Terminal Blocks",
    "Grounding Lugs Kit",
    "Engraved Stainless Steel Tags",
    "Identification Tags",
    "Junction Box Mounting Feet",
    
    # Calibration & Tools
    "HART Communicator",
    "Portable HART Communicator",
    "Calibrator",
    "Temperature Calibrator",
    "Pressure Calibrator",
    "Multi-function Calibrator",
    
    # Safety & Barriers
    "Safety Barrier",
    "Intrinsic Safety Barrier",
    "Galvanic Isolator",
    "Surge Protector",
    
    # Communication
    "Protocol Converter",
    "HART Multiplexer",
    "Fieldbus Interface",
    "Wireless Gateway"
]

def check_cached_images():
    """Check which product types already have cached images."""
    from generic_image_utils import get_cached_generic_image
    
    cached = []
    missing = []
    
    print("\n" + "="*60)
    print("CHECKING CACHED IMAGES")
    print("="*60 + "\n")
    
    for product_type in PRODUCT_TYPES_TO_CACHE:
        result = get_cached_generic_image(product_type)
        if result:
            cached.append(product_type)
            print(f"  ✅ {product_type}")
        else:
            missing.append(product_type)
            print(f"  ❌ {product_type}")
    
    print("\n" + "="*60)
    print(f"SUMMARY: {len(cached)} cached, {len(missing)} missing")
    print("="*60 + "\n")
    
    return cached, missing


def precache_images(force_regenerate=False, delay_seconds=10):
    """Pre-cache all product type images."""
    from generic_image_utils import get_cached_generic_image, fetch_generic_product_image
    
    print("\n" + "="*60)
    print("PRE-CACHING PRODUCT TYPE IMAGES")
    print(f"Force regenerate: {force_regenerate}")
    print(f"Delay between requests: {delay_seconds}s")
    print("="*60 + "\n")
    
    total = len(PRODUCT_TYPES_TO_CACHE)
    successful = 0
    skipped = 0
    failed = 0
    
    for i, product_type in enumerate(PRODUCT_TYPES_TO_CACHE, 1):
        print(f"\n[{i}/{total}] Processing: {product_type}")
        
        # Check if already cached
        if not force_regenerate:
            cached = get_cached_generic_image(product_type)
            if cached:
                print(f"  ✅ Already cached - skipping")
                skipped += 1
                continue
        
        # Generate and cache
        print(f"  🔄 Generating image...")
        start_time = time.time()
        
        result = fetch_generic_product_image(product_type)
        
        elapsed = time.time() - start_time
        
        if result:
            print(f"  ✅ Success ({elapsed:.1f}s)")
            successful += 1
        else:
            print(f"  ❌ Failed ({elapsed:.1f}s)")
            failed += 1
        
        # Wait before next request (unless this is the last one)
        if i < total:
            print(f"  ⏳ Waiting {delay_seconds}s before next request...")
            time.sleep(delay_seconds)
    
    print("\n" + "="*60)
    print("PRE-CACHING COMPLETE")
    print(f"  Successful: {successful}")
    print(f"  Skipped (already cached): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {total}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Pre-cache generic product type images")
    parser.add_argument("--check", action="store_true", help="Check which images are cached/missing")
    parser.add_argument("--force", action="store_true", help="Force regenerate all images (even if cached)")
    parser.add_argument("--delay", type=int, default=10, help="Delay between requests in seconds (default: 10)")
    
    args = parser.parse_args()
    
    if args.check:
        check_cached_images()
    else:
        precache_images(force_regenerate=args.force, delay_seconds=args.delay)


if __name__ == "__main__":
    main()
