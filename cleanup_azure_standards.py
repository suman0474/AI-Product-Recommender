"""
Script to clean up instrumentation standards JSON files in Azure Blob Storage
Removes: statistics, extracted_date, and full_text sections
"""

import json
from azure_blob_config import azure_blob_manager
from azure.storage.blob import ContentSettings

def clean_standard_data(data):
    """Remove unwanted fields from standard data"""
    fields_to_remove = ['statistics', 'extracted_date', 'full_text', 'full_text_preview']

    if isinstance(data, dict):
        # If this is a collection of standards
        if any(key.startswith('instrumentation_') for key in data.keys()):
            cleaned_data = {}
            for standard_name, standard_content in data.items():
                cleaned_standard = {k: v for k, v in standard_content.items()
                                   if k not in fields_to_remove}
                cleaned_data[standard_name] = cleaned_standard
            return cleaned_data
        else:
            # Single standard or other dict
            return {k: v for k, v in data.items() if k not in fields_to_remove}

    return data

def clean_summary_data(data):
    """Remove unwanted fields from summary data"""
    if isinstance(data, dict):
        # Remove extraction_date from summary
        cleaned = {k: v for k, v in data.items() if k != 'extraction_date'}

        # Clean standards_overview if it exists
        if 'standards_overview' in cleaned:
            cleaned_standards = []
            for standard in cleaned['standards_overview']:
                clean_std = {k: v for k, v in standard.items() if k != 'statistics'}
                cleaned_standards.append(clean_std)
            cleaned['standards_overview'] = cleaned_standards

        return cleaned
    return data

def main():
    print("=" * 80)
    print("CLEANING AZURE BLOB STORAGE STANDARDS JSON FILES")
    print("=" * 80)

    # Configuration
    CONTAINER_NAME = "instrumentation-standards"
    FOLDER_NAME = "standards-20251230-095749"  # The folder we uploaded to

    try:
        # Connect to Azure
        if not azure_blob_manager.is_available:
            print("[ERROR] Azure Blob Storage is not available")
            return

        blob_service_client = azure_blob_manager.blob_service_client
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        print(f"\n[OK] Connected to Azure Blob Storage")
        print(f"Container: {CONTAINER_NAME}")
        print(f"Folder: {FOLDER_NAME}\n")

        # List all JSON files in the folder
        blobs = container_client.list_blobs(name_starts_with=f"{FOLDER_NAME}/")
        json_files = [blob.name for blob in blobs if blob.name.endswith('.json')]

        print(f"Found {len(json_files)} JSON files to clean\n")

        for blob_name in json_files:
            print(f"Processing: {blob_name}")

            # Download the blob
            blob_client = container_client.get_blob_client(blob_name)
            json_content = blob_client.download_blob().readall().decode('utf-8')
            data = json.loads(json_content)

            # Clean the data based on file type
            if 'summary' in blob_name:
                cleaned_data = clean_summary_data(data)
                print("  - Removed: extraction_date, statistics from overview")
            else:
                cleaned_data = clean_standard_data(data)
                print("  - Removed: statistics, extracted_date, full_text")

            # Convert back to JSON
            cleaned_json = json.dumps(cleaned_data, indent=2, ensure_ascii=False)

            # Upload back to Azure
            blob_client.upload_blob(
                cleaned_json,
                overwrite=True,
                content_settings=ContentSettings(content_type='application/json')
            )

            original_size = len(json_content)
            new_size = len(cleaned_json)
            savings = original_size - new_size

            print(f"  - Original size: {original_size:,} bytes")
            print(f"  - New size: {new_size:,} bytes")
            print(f"  - Saved: {savings:,} bytes ({(savings/original_size*100):.1f}%)")
            print(f"  [OK] Updated successfully\n")

        print("=" * 80)
        print("[SUCCESS] All files cleaned successfully")
        print("=" * 80)

    except Exception as e:
        print(f"[ERROR] Failed to clean files: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
