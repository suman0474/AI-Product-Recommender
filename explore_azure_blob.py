"""
Azure Blob Storage Explorer
Lists all blobs in the container and their properties
"""

import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from datetime import datetime

load_dotenv()

def format_size(size_bytes):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def explore_azure_blob():
    """Explore Azure Blob Storage and list all contents"""

    print("=" * 80)
    print("AZURE BLOB STORAGE EXPLORER")
    print("=" * 80)

    # Get connection details
    connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    container_name = os.getenv('AZURE_BLOB_SINGLE_CONTAINER', 'product-data')
    base_path = os.getenv('AZURE_BLOB_BASE_PATH', 'Product-Recommender')

    if not connection_string:
        print("\n[ERROR] AZURE_STORAGE_CONNECTION_STRING not found in .env")
        return

    print(f"\nContainer: {container_name}")
    print(f"Base Path: {base_path}")
    print(f"Connection: {'*' * 20}...{connection_string[-20:]}")

    try:
        # Connect to Azure
        print("\n[INFO] Connecting to Azure Blob Storage...")
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        container_client = blob_service_client.get_container_client(container_name)

        # Get container properties
        props = container_client.get_container_properties()
        print(f"[SUCCESS] Connected to container: {props['name']}")
        print(f"   Last Modified: {props['last_modified']}")

        # List all blobs
        print("\n" + "=" * 80)
        print("LISTING ALL BLOBS")
        print("=" * 80)

        blobs = list(container_client.list_blobs(include=['metadata']))

        if not blobs:
            print("\n[WARNING] Container is empty - no blobs found")
            return

        print(f"\nTotal blobs found: {len(blobs)}")

        # Organize blobs by folder
        folders = {}
        total_size = 0

        for blob in blobs:
            total_size += blob.size

            # Extract folder path
            if '/' in blob.name:
                folder = '/'.join(blob.name.split('/')[:-1])
            else:
                folder = '(root)'

            if folder not in folders:
                folders[folder] = []

            folders[folder].append(blob)

        print(f"Total folders: {len(folders)}")
        print(f"Total size: {format_size(total_size)}")

        # Display by folder
        print("\n" + "=" * 80)
        print("FOLDER STRUCTURE")
        print("=" * 80)

        for folder in sorted(folders.keys()):
            folder_blobs = folders[folder]
            folder_size = sum(b.size for b in folder_blobs)

            print(f"\n[FOLDER] {folder}/")
            print(f"         Files: {len(folder_blobs)} | Size: {format_size(folder_size)}")
            print("   " + "-" * 76)

            # Show first 10 files in each folder
            for i, blob in enumerate(sorted(folder_blobs, key=lambda x: x.name)[:10], 1):
                filename = blob.name.split('/')[-1]
                metadata = blob.metadata or {}

                print(f"   {i}. {filename}")
                print(f"      Size: {format_size(blob.size)} | Modified: {blob.last_modified.strftime('%Y-%m-%d %H:%M:%S')}")

                if metadata:
                    meta_str = ' | '.join([f"{k}={v}" for k, v in list(metadata.items())[:3]])
                    print(f"      Metadata: {meta_str}")

            if len(folder_blobs) > 10:
                print(f"   ... and {len(folder_blobs) - 10} more files")

        # Check for base path specifically
        print("\n" + "=" * 80)
        print(f"BASE PATH: {base_path}")
        print("=" * 80)

        base_path_blobs = [b for b in blobs if b.name.startswith(base_path)]

        if base_path_blobs:
            print(f"\n[SUCCESS] Found {len(base_path_blobs)} blobs in base path")

            # Organize by collection
            collections = {}
            for blob in base_path_blobs:
                parts = blob.name.replace(base_path + '/', '').split('/')
                if len(parts) > 0:
                    collection = parts[0]
                    if collection not in collections:
                        collections[collection] = []
                    collections[collection].append(blob)

            print(f"Collections found: {len(collections)}")

            for collection in sorted(collections.keys()):
                coll_blobs = collections[collection]
                coll_size = sum(b.size for b in coll_blobs)
                print(f"\n   [COLLECTION] {collection}/")
                print(f"                Files: {len(coll_blobs)} | Size: {format_size(coll_size)}")

                # Show sample files
                for blob in sorted(coll_blobs, key=lambda x: x.name)[:5]:
                    filename = blob.name.split('/')[-1]
                    print(f"      - {filename} ({format_size(blob.size)})")

                if len(coll_blobs) > 5:
                    print(f"      ... and {len(coll_blobs) - 5} more files")
        else:
            print(f"\n[WARNING] No blobs found in base path: {base_path}")
            print("          This is expected for a new setup - data will be created when you use the application")

        # Summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"Connection: Successful")
        print(f"Container: {container_name}")
        print(f"Total Blobs: {len(blobs)}")
        print(f"Total Folders: {len(folders)}")
        print(f"Total Size: {format_size(total_size)}")
        print(f"Base Path Blobs: {len(base_path_blobs)}")

        if base_path_blobs:
            print(f"Collections: {', '.join(sorted(collections.keys()))}")

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explore_azure_blob()
