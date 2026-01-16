# -*- coding: utf-8 -*-
"""
Check Azure Blob Storage structure and compare with expected structure
"""
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

def main():
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    container_name = os.getenv('AZURE_BLOB_SINGLE_CONTAINER', 'product-data')
    base_path = os.getenv('AZURE_BLOB_BASE_PATH', 'Product-Recommender')
    
    print("=" * 70)
    print("AZURE BLOB STORAGE STRUCTURE VERIFICATION")
    print("=" * 70)
    print(f"Container: {container_name}")
    print(f"Base Path: {base_path}")
    print()
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blobs = list(container_client.list_blobs())
    
    # Expected structure from WF.md documentation
    expected_folders = [
        "product_schemas",      # PPI-generated schemas (versioned)
        "specs",               # Legacy/manual schemas
        "advanced_parameters", # Cached advanced specs
        "vendors",             # Product data JSONs
        "documents",           # Downloaded PDFs
        "images",              # Cached product images
    ]
    
    print("=" * 70)
    print("EXPECTED vs ACTUAL STRUCTURE")
    print("=" * 70)
    
    for folder in expected_folders:
        folder_blobs = [b for b in blobs if b.name.startswith(f'{base_path}/{folder}/')]
        count = len(folder_blobs)
        status = "[OK]" if count > 0 else "[MISSING]"
        print(f"\n{status} {folder}/ ({count} blobs)")
        
        # Show sample files
        for b in folder_blobs[:3]:
            filename = b.name.replace(f'{base_path}/{folder}/', '')
            print(f"    - {filename}")
        if count > 3:
            print(f"    ... and {count-3} more")
    
    # Check all unique top-level folders
    print()
    print("=" * 70)
    print("ALL TOP-LEVEL FOLDERS")
    print("=" * 70)
    folders = set()
    for blob in blobs:
        if blob.name.startswith(base_path + '/'):
            parts = blob.name.replace(base_path + '/', '').split('/')
            if len(parts) > 0 and parts[0]:
                folders.add(parts[0])
    
    for f in sorted(folders):
        count = len([b for b in blobs if b.name.startswith(f'{base_path}/{f}/')])
        print(f"  /{f}/ - {count} blobs")
    
    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    missing = []
    for folder in expected_folders:
        folder_blobs = [b for b in blobs if b.name.startswith(f'{base_path}/{folder}/')]
        if len(folder_blobs) == 0:
            missing.append(folder)
    
    if missing:
        print(f"MISSING FOLDERS: {', '.join(missing)}")
    else:
        print("All expected folders exist!")

if __name__ == "__main__":
    main()
