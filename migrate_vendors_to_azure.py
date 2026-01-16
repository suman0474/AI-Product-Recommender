import sys
import os
import bson
import json
import logging
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(os.getcwd())

from azure_blob_utils import azure_blob_file_manager
from azure_blob_config import Collections

def normalize_key(pt, vn):
    if not pt: pt = ""
    if not vn: vn = ""
    return f"{pt.strip().lower()}_{vn.strip().lower()}"

def merge_data(existing, new_item):
    """Merge new_item data into existing data"""
    if isinstance(existing, dict) and isinstance(new_item, dict):
        # Check for 'models' list
        if 'models' in existing and 'models' in new_item:
            existing['models'].extend(new_item['models'])
            return existing
        # Generic merge - what to do? overwrite keys?
        # For now, let's assume models structure is dominant
        existing.update(new_item)
        return existing
    elif isinstance(existing, list) and isinstance(new_item, list):
        existing.extend(new_item)
        return existing
    else:
        # Mismatch or unknown structure, keep existing (or warn)
        logger.warning("Could not merge data structures. Keeping existing.")
        return existing

def migrate():
    bson_path = r"D:\AI PR\AIPR\backend\documents\vendors.bson"
    
    if not os.path.exists(bson_path):
        logger.error(f"Source file not found: {bson_path}")
        return

    logger.info(f"Reading {bson_path}...")
    try:
        with open(bson_path, 'rb') as f:
            raw_docs = bson.decode_all(f.read())
    except Exception as e:
        logger.error(f"Failed to read BSON: {e}")
        return

    logger.info(f"Found {len(raw_docs)} documents. Aggregating...")

    # Aggregation
    grouped_docs = {} # Key: normalized_key, Value: { 'metadata': ..., 'data': ... }
    
    for doc in raw_docs:
        pt = doc.get('product_type')
        vn = doc.get('vendor_name')
        
        if not pt or not vn:
            logger.warning(f"Skipping doc with missing keys: pt={pt}, vn={vn}")
            continue
            
        key = normalize_key(pt, vn)
        
        doc_data = doc.get('data')
        doc_meta = doc.get('metadata', {})
        
        # Ensure metadata has core fields
        doc_meta['product_type'] = pt
        doc_meta['vendor_name'] = vn
        doc_meta['collection_type'] = Collections.VENDORS
        
        if key not in grouped_docs:
            grouped_docs[key] = {
                'metadata': doc_meta,
                'data': doc_data,
                'product_type': pt,
                'vendor_name': vn
            }
        else:
            # Merge data
            merged_content = merge_data(grouped_docs[key]['data'], doc_data)
            grouped_docs[key]['data'] = merged_content
            # Keep oldest or newest metadata? Azure handles 'upload_date' in upload function
            
    logger.info(f"aggregated into {len(grouped_docs)} unique vendor/product groups.")
    
    # Upload to Azure
    logger.info("Starting upload to Azure Blob Storage...")
    
    success_count = 0
    fail_count = 0
    
    def sanitize_header_value(val):
        """Ensure value is ASCII-safe for HTTP headers"""
        if not val:
            return ""
        # encode to ascii, ignoring errors, then decode back
        return str(val).encode('ascii', 'ignore').decode('ascii').strip()

    for key, item in grouped_docs.items():
        try:
            # Prepare clean metadata for the upload function
            # We copy specific fields we want
            clean_meta = {
               'product_type': sanitize_header_value(item['product_type']),
               'vendor_name': sanitize_header_value(item['vendor_name']),
               'collection_type': Collections.VENDORS
            }
            
            # Note: The 'item['metadata']' from BSON might contain original scraped metadata.
            # We preserve that in the JSON body via 'item['data']' merging or explicitly if we modified upload_json_data
            # But upload_json_data takes 'metadata' arg and uses it for BOTH blob metadata AND inside the JSON doc.
            # So passing clean_meta is safer for the blob, but we might lose rich metadata in the JSON?
            # actually upload_json_data:
            # document['metadata'] = { ... **metadata }
            # so we should include original metadata but filtered?
            
            # Let's trust that original metadata inside BSON is valuable, but for the 'product_type'/'vendor_name' keys 
            # used by Azure headers, we must ensure they are clean.
            
            # Merge original metadata but overwrite critical fields with clean versions
            upload_meta = item['metadata'].copy()
            upload_meta['product_type'] = clean_meta['product_type']
            upload_meta['vendor_name'] = clean_meta['vendor_name']
            
            # Also remove any keys from upload_meta that might be non-primitive if upload_json_data tries to put them in headers?
            # upload_json_data logic:
            # blob_metadata = { 'product_type': ..., 'vendor_name': ..., ... }
            # It ONLY picks specific keys for the headers. 
            # So as long as product_type and vendor_name are clean, we are good.
            
            doc_id = azure_blob_file_manager.upload_json_data(
                json_data=item['data'],
                metadata=upload_meta
            )
            success_count += 1
            if success_count % 10 == 0:
                logger.info(f"Uploaded {success_count}...")
        except Exception as e:
            logger.error(f"Failed to upload {key}: {e}")
            fail_count += 1
            
    logger.info("Migration Complete.")
    logger.info(f"Success: {success_count}")
    logger.info(f"Failed: {fail_count}")

if __name__ == "__main__":
    migrate()
