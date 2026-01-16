
import logging
import sys
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add backend directory to path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    from azure_blob_config import azure_blob_manager
except ImportError as e:
    logger.error(f"Import failed: {e}")
    sys.exit(1)

def inspect_blob_structure():
    print("\n" + "="*80)
    print("INSPECTING AZURE BLOB STORAGE STRUCTURE")
    print("="*80)

    try:
        # Use simple prints to debug
        print(f"Container: {azure_blob_manager.container_name}")
        print(f"Base Path: {azure_blob_manager.base_path}")
        
        container_client = azure_blob_manager.container_client
        config_base_path = azure_blob_manager.base_path

        print(f"\n[1] Listing Top-Level Items in '{config_base_path}/':")
        # We want to see what's directly under Product-Recommender/
        # Use a delimiter to act like a folder listing
        top_level = container_client.walk_blobs(name_starts_with=f"{config_base_path}/", delimiter="/")
        
        for item in top_level:
            print(f" - {item.name}")

        # 2. Inspect 'vendors' Directory
        vendors_path = f"{config_base_path}/vendors/"
        print(f"\n[2] Inspecting '{vendors_path}':")
        
        # List blobs in vendors path (recursive but limited depth logic here essentially)
        # We will list all blobs starting with vendors/ to see structure
        # Limit to first 50 to avoid spam
        blobs = container_client.list_blobs(name_starts_with=vendors_path)
        
        count = 0
        vendors_found = set()
        
        for blob in blobs:
            # Parse structure: Product-Recommender/vendors/{VendorName}/...
            relative_path = blob.name.replace(vendors_path, "")
            parts = relative_path.split('/')
            if parts:
                vendors_found.add(parts[0])
            
            if count < 10:
                print(f" - {blob.name}")
            elif count == 10:
                print(" ... (output truncated)")
            count += 1
            
        print(f"\nTotal Blobs in 'vendors': {count}")
        print(f"Vendors Identified (folder names): {sorted(list(vendors_found))}")
        
        # 3. Check for specific Temperature Transmitter data
        print(f"\n[3] Searching for 'Temperature' or 'Transmitter' related files...")
        blobs = container_client.list_blobs(name_starts_with=vendors_path)
        related_files = []
        for blob in blobs:
            if 'temp' in blob.name.lower() or 'transmitter' in blob.name.lower():
                related_files.append(blob.name)
        
        if related_files:
            print(f"Found {len(related_files)} potentially related files:")
            for f in related_files:
                print(f" - {f}")
        else:
            print("No files matched keyword search.")

    except Exception as e:
        logger.error(f"Inspection failed: {e}")

def format_size(size_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def calculate_storage_stats():
    print("\n" + "="*80)
    print("CALCULATING TOTAL STORAGE USAGE")
    print("="*80)
    
    try:
        container_client = azure_blob_manager.container_client
        base_path = azure_blob_manager.base_path
        
        print(f"Scanning all blobs under '{base_path}'...")
        
        # List all blobs recursively
        blobs = container_client.list_blobs(name_starts_with=base_path)
        
        total_size = 0
        total_files = 0
        
        for blob in blobs:
            total_size += blob.size
            total_files += 1
            # print(f".", end="", flush=True) if total_files % 10 == 0 else None
            
        print(f"\n\nTotal Files: {total_files}")
        print(f"Total Size: {format_size(total_size)}")
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")

if __name__ == "__main__":
    inspect_blob_structure()
    calculate_storage_stats()
