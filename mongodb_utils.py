"""
MongoDB Helper Utilities for File Operations
Provides high-level functions for storing and retrieving files using GridFS
"""

import os
import json
import io
import logging
from typing import Dict, Any, List, Optional, Union, BinaryIO
from datetime import datetime
from bson import ObjectId
from gridfs.errors import NoFile
from pymongo.errors import DuplicateKeyError

from mongodb_config import get_mongodb_connection, Collections

logger = logging.getLogger(__name__)

class MongoDBFileManager:
    """High-level file management for MongoDB with GridFS"""
    
    def __init__(self):
        self.conn = get_mongodb_connection()
        self.gridfs = self.conn['gridfs']
        self.collections = self.conn['collections']
        self.db = self.conn['database']  # Add database property for direct access
    
    # ==================== UPLOAD OPERATIONS ====================
    
    def upload_to_mongodb(self, file_path_or_data: Union[str, bytes, BinaryIO], 
                         metadata: Dict[str, Any]) -> str:
        """
        Upload file to MongoDB GridFS with metadata
        
        Args:
            file_path_or_data: File path, bytes, or file-like object
            metadata: File metadata including collection_type, product_type, vendor_name, etc.
            
        Returns:
            str: GridFS file ID
        """
        try:
            # Determine file data
            if isinstance(file_path_or_data, str):
                # File path
                with open(file_path_or_data, 'rb') as f:
                    file_data = f.read()
                filename = os.path.basename(file_path_or_data)
            elif isinstance(file_path_or_data, bytes):
                # Bytes data
                file_data = file_path_or_data
                filename = metadata.get('filename', 'unknown_file')
            else:
                # File-like object
                file_data = file_path_or_data.read()
                filename = metadata.get('filename', 'unknown_file')
            
            # Add standard metadata
            upload_metadata = {
                'filename': filename,
                'upload_date': datetime.utcnow(),
                'file_size': len(file_data),
                **metadata
            }
            
            # Upload to GridFS
            file_id = self.gridfs.put(file_data, **upload_metadata)
            
            # Store metadata in appropriate collection
            self._store_file_metadata(str(file_id), upload_metadata)
            
            logger.info(f"Successfully uploaded file to MongoDB: {filename} (ID: {file_id})")
            return str(file_id)
            
        except Exception as e:
            logger.error(f"Failed to upload file to MongoDB: {e}")
            raise
    
    def upload_json_data(self, json_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """
        Upload JSON data to MongoDB
        
        Args:
            json_data: JSON data to store
            metadata: Metadata including collection_type, product_type, vendor_name, etc.
            
        Returns:
            str: Document ID
        """
        try:
            collection_type = metadata.get('collection_type', 'documents')
            collection = self.collections[collection_type]
            
            # Add standard metadata
            document = {
                'data': json_data,
                'metadata': {
                    'upload_date': datetime.utcnow(),
                    **metadata
                },
                'product_type': metadata.get('product_type'),
                'vendor_name': metadata.get('vendor_name'),
                'file_type': 'json'
            }
            
            # Insert document
            result = collection.insert_one(document)
            doc_id = str(result.inserted_id)
            
            logger.info(f"Successfully uploaded JSON data to MongoDB collection '{collection_type}' (ID: {doc_id})")
            return doc_id
            
        except Exception as e:
            logger.error(f"Failed to upload JSON data to MongoDB: {e}")
            raise
    
    def _store_file_metadata(self, file_id: str, metadata: Dict[str, Any]):
        """Store file metadata in appropriate metadata collection"""
        try:
            collection_type = metadata.get('collection_type', 'documents')
            metadata_collection_name = f"{collection_type}_metadata"
            
            if metadata_collection_name in self.collections:
                metadata_doc = {
                    'file_id': file_id,
                    'created_at': datetime.utcnow(),
                    **metadata
                }
                
                self.collections[metadata_collection_name].insert_one(metadata_doc)
                
        except Exception as e:
            logger.warning(f"Failed to store file metadata: {e}")
    
    # ==================== RETRIEVAL OPERATIONS ====================
    
    def get_file_from_mongodb(self, collection_name: str, query: Dict[str, Any]) -> Optional[bytes]:
        """
        Retrieve file from MongoDB GridFS
        
        Args:
            collection_name: Collection type (documents, vendors, static, specs)
            query: Query to find the file
            
        Returns:
            bytes: File content or None if not found
        """
        try:
            # Strategy 1: Search GridFS directly (migration stores metadata at top level)
            grid_file = self.gridfs.find_one(query)
            if grid_file:
                return grid_file.read()
            
            # Strategy 2: Try to find in metadata collection
            metadata_collection = self.collections.get(f"{collection_name}_metadata")
            if metadata_collection is not None:
                metadata_doc = metadata_collection.find_one(query)
                if metadata_doc and 'file_id' in metadata_doc:
                    # Handle both string and ObjectId file_id
                    file_id = metadata_doc['file_id']
                    if isinstance(file_id, str):
                        file_id = ObjectId(file_id)
                    
                    grid_file = self.gridfs.get(file_id)
                    return grid_file.read()
            
            # Strategy 3: Enhanced GridFS search with collection_type filter
            if collection_name in ['documents', 'vendors', 'specs']:
                enhanced_query = {**query, 'collection_type': collection_name}
                grid_file = self.gridfs.find_one(enhanced_query)
                if grid_file:
                    return grid_file.read()
            
            return None
            
        except NoFile:
            logger.warning(f"File not found in MongoDB: {query}")
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve file from MongoDB: {e}")
            raise
    
    def get_json_data_from_mongodb(self, collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Retrieve JSON data from MongoDB collection
        
        Args:
            collection_name: Collection name
            query: Query to find the document
            
        Returns:
            Dict: JSON data or None if not found
        """
        try:
            collection = self.collections.get(collection_name)
            if collection is None:
                logger.error(f"Collection '{collection_name}' not found")
                return None
            
            document = collection.find_one(query)
            if document:
                # Handle both new format (with 'data' field) and legacy format
                if 'data' in document:
                    return document['data']
                else:
                    # Legacy format or direct storage - return the document minus MongoDB metadata
                    return {k: v for k, v in document.items() if k not in ['_id', 'metadata']}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve JSON data from MongoDB: {e}")
            raise
    
    def get_schema_from_mongodb(self, product_type: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve schema from MongoDB specs collection
        
        Args:
            product_type: Product type to get schema for
            
        Returns:
            Dict: Schema data or None if not found
        """
        try:
            # Normalize product type for search
            normalized_type = product_type.lower().replace(" ", "").replace("_", "")
            
            # Try multiple search strategies to find schema
            specs_collection = self.collections['specs']
            
            # Strategy 1: Exact match on top-level product_type
            query = {'product_type': product_type}
            schema_doc = specs_collection.find_one(query)
            
            if not schema_doc:
                # Strategy 2: Normalized match on metadata
                query = {'metadata.normalized_product_type': normalized_type}
                schema_doc = specs_collection.find_one(query)
            
            if not schema_doc:
                # Strategy 3: Case-insensitive regex match on multiple fields
                cursor = specs_collection.find({
                    '$or': [
                        {'product_type': {'$regex': product_type, '$options': 'i'}},
                        {'metadata.product_type': {'$regex': product_type, '$options': 'i'}},
                        {'metadata.normalized_product_type': {'$regex': normalized_type, '$options': 'i'}}
                    ]
                })
                
                for doc in cursor:
                    schema_doc = doc
                    break
            
            if schema_doc:
                # Handle both new format (with 'data' field) and legacy format
                if 'data' in schema_doc:
                    return schema_doc['data']
                else:
                    # Legacy format - return document minus MongoDB metadata
                    return {k: v for k, v in schema_doc.items() if k not in ['_id', 'metadata']}
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve schema from MongoDB: {e}")
            raise
    
    # ==================== QUERY OPERATIONS ====================
    
    def list_files(self, collection_name: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        List files in a collection with optional filters
        
        Args:
            collection_name: Collection name
            filters: Optional filters to apply
            
        Returns:
            List of file metadata
        """
        try:
            metadata_collection_name = f"{collection_name}_metadata"
            metadata_collection = self.collections.get(metadata_collection_name)
            
            if metadata_collection is None:
                return []
            
            query = filters or {}
            cursor = metadata_collection.find(query)
            
            return list(cursor)
            
        except Exception as e:
            logger.error(f"Failed to list files from MongoDB: {e}")
            return []
    
    def file_exists(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Check if file exists in MongoDB
        
        Args:
            collection_name: Collection name
            query: Query to check
            
        Returns:
            bool: True if file exists
        """
        try:
            metadata_collection_name = f"{collection_name}_metadata"
            metadata_collection = self.collections.get(metadata_collection_name)
            
            if metadata_collection is not None:
                return metadata_collection.find_one(query) is not None
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to check file existence in MongoDB: {e}")
            return False
    
    # ==================== DELETE OPERATIONS ====================
    
    def delete_file(self, collection_name: str, query: Dict[str, Any]) -> bool:
        """
        Delete file from MongoDB
        
        Args:
            collection_name: Collection name
            query: Query to find file to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            metadata_collection_name = f"{collection_name}_metadata"
            metadata_collection = self.collections.get(metadata_collection_name)
            
            if metadata_collection is not None:
                metadata_doc = metadata_collection.find_one(query)
                if metadata_doc and 'file_id' in metadata_doc:
                    # Delete from GridFS
                    file_id = ObjectId(metadata_doc['file_id'])
                    self.gridfs.delete(file_id)
                    
                    # Delete metadata
                    metadata_collection.delete_one({'_id': metadata_doc['_id']})
                    
                    logger.info(f"Successfully deleted file from MongoDB: {query}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete file from MongoDB: {e}")
            return False

# Global instance
mongodb_file_manager = MongoDBFileManager()

# ==================== CONVENIENCE FUNCTIONS ====================

def upload_to_mongodb(file_path_or_data: Union[str, bytes, BinaryIO], metadata: Dict[str, Any]) -> str:
    """Convenience function for uploading files"""
    return mongodb_file_manager.upload_to_mongodb(file_path_or_data, metadata)

def get_file_from_mongodb(collection_name: str, query: Dict[str, Any]) -> Optional[bytes]:
    """Convenience function for retrieving files"""
    return mongodb_file_manager.get_file_from_mongodb(collection_name, query)

def get_schema_from_mongodb(product_type: str) -> Optional[Dict[str, Any]]:
    """Convenience function for retrieving schemas"""
    return mongodb_file_manager.get_schema_from_mongodb(product_type)

def upload_json_to_mongodb(json_data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
    """Convenience function for uploading JSON data"""
    return mongodb_file_manager.upload_json_data(json_data, metadata)

def get_json_from_mongodb(collection_name: str, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Convenience function for retrieving JSON data"""
    return mongodb_file_manager.get_json_data_from_mongodb(collection_name, query)

# ==================== IMAGE CACHING FUNCTIONS ====================

def download_image_from_url(url: str, timeout: int = 30) -> Optional[tuple]:
    """
    Download image from URL and return binary data with metadata
    
    Args:
        url: Image URL to download
        timeout: Request timeout in seconds
        
    Returns:
        Tuple of (image_data: bytes, content_type: str, file_size: int) or None if failed
    """
    try:
        import requests
        
        # Validate URL scheme - reject unsupported schemes
        if not url or not isinstance(url, str):
            logger.warning(f"Invalid URL provided: {url}")
            return None
            
        # Check for unsupported URL schemes
        unsupported_schemes = ['x-raw-image://', 'data:', 'blob:', 'chrome://', 'about:']
        if any(url.startswith(scheme) for scheme in unsupported_schemes):
            logger.warning(f"Unsupported URL scheme, skipping: {url}")
            return None
        
        # Ensure URL starts with http:// or https://
        if not url.startswith(('http://', 'https://')):
            logger.warning(f"URL must start with http:// or https://, got: {url}")
            return None
        
        # Add headers to appear like a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/webp,image/apng,image/*,*/*;q=0.8',
        }
        
        logger.info(f"Downloading image from: {url}")
        response = requests.get(url, timeout=timeout, headers=headers, stream=True)
        response.raise_for_status()
        
        # Get content type
        content_type = response.headers.get('content-type', 'image/jpeg').lower()
        
        # Validate it's an image
        if 'image' not in content_type:
            logger.warning(f"URL does not return an image: {url} (content-type: {content_type})")
            return None
        
        # Download image data
        image_data = b''
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                image_data += chunk
        
        file_size = len(image_data)
        
        # Validate minimum size (avoid empty or corrupt images)
        if file_size < 100:  # Less than 100 bytes is suspicious
            logger.warning(f"Downloaded image is too small ({file_size} bytes): {url}")
            return None
        
        logger.info(f"Successfully downloaded image: {file_size} bytes, type: {content_type}")
        return (image_data, content_type, file_size)
        
    except requests.Timeout:
        logger.error(f"Timeout downloading image from: {url}")
        return None
    except requests.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error downloading image from {url}: {e}")
        return None

def get_cached_image(vendor_name: str, model_family: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached product image from MongoDB GridFS
    
    Args:
        vendor_name: Vendor/manufacturer name
        model_family: Model family name
        
    Returns:
        Dict containing image metadata with gridfs_file_id or None if not found
    """
    try:
        images_collection = mongodb_file_manager.collections['images']
        
        # Normalize search keys
        normalized_vendor = vendor_name.strip().lower()
        normalized_model = model_family.strip().lower()
        
        query = {
            'vendor_name_normalized': normalized_vendor,
            'model_family_normalized': normalized_model
        }
        
        cached_image = images_collection.find_one(query)
        
        if cached_image and cached_image.get('gridfs_file_id'):
            logger.info(f"Found cached image in GridFS for {vendor_name} - {model_family}")
            # Return image metadata with GridFS file reference
            return {
                'gridfs_file_id': str(cached_image.get('gridfs_file_id')),
                'title': cached_image.get('title'),
                'source': cached_image.get('source'),
                'domain': cached_image.get('domain'),
                'content_type': cached_image.get('content_type', 'image/jpeg'),
                'file_size': cached_image.get('file_size', 0),
                'original_url': cached_image.get('original_url'),
                'cached': True
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve cached image: {e}")
        return None

def cache_image(vendor_name: str, model_family: str, image_data: Dict[str, Any]) -> bool:
    """
    Download and store product image in MongoDB GridFS
    
    Args:
        vendor_name: Vendor/manufacturer name
        model_family: Model family name
        image_data: Image data dict containing url, title, source, etc.
        
    Returns:
        bool: True if successfully cached
    """
    try:
        images_collection = mongodb_file_manager.collections['images']
        gridfs = mongodb_file_manager.gridfs
        
        # Get image URL
        image_url = image_data.get('url')
        if not image_url:
            logger.warning(f"No URL provided for image caching: {vendor_name} - {model_family}")
            return False
        
        # Download the actual image
        download_result = download_image_from_url(image_url)
        if not download_result:
            logger.warning(f"Failed to download image from {image_url}, skipping cache")
            return False
        
        image_bytes, content_type, file_size = download_result
        
        # Normalize keys for indexing
        normalized_vendor = vendor_name.strip().lower()
        normalized_model = model_family.strip().lower()
        
        # Generate filename
        file_extension = content_type.split('/')[-1] if '/' in content_type else 'jpg'
        filename = f"{normalized_vendor}_{normalized_model}.{file_extension}"
        
        # Store image in GridFS
        gridfs_metadata = {
            'vendor_name': vendor_name,
            'model_family': model_family,
            'original_url': image_url,
            'source': image_data.get('source', ''),
            'domain': image_data.get('domain', '')
        }
        
        gridfs_file_id = gridfs.put(
            image_bytes,
            filename=filename,
            content_type=content_type,
            **gridfs_metadata
        )
        
        logger.info(f"Stored image in GridFS: {filename} (ID: {gridfs_file_id}, size: {file_size} bytes)")
        
        # Store metadata in images collection
        image_doc = {
            'vendor_name': vendor_name,
            'vendor_name_normalized': normalized_vendor,
            'model_family': model_family,
            'model_family_normalized': normalized_model,
            'gridfs_file_id': gridfs_file_id,
            'original_url': image_url,
            'title': image_data.get('title', ''),
            'source': image_data.get('source', ''),
            'domain': image_data.get('domain', ''),
            'content_type': content_type,
            'file_size': file_size,
            'filename': filename,
            'created_at': datetime.utcnow()
        }
        
        # Use upsert to avoid duplicates (update if exists, insert if not)
        result = images_collection.update_one(
            {
                'vendor_name_normalized': normalized_vendor,
                'model_family_normalized': normalized_model
            },
            {'$set': image_doc},
            upsert=True
        )
        
        if result.upserted_id or result.modified_count > 0:
            logger.info(f"Successfully cached image in GridFS for {vendor_name} - {model_family}")
            return True
        
        return False
        
    except Exception as e:
        logger.error(f"Failed to cache image in GridFS: {e}")
        return False

def get_available_vendors_from_mongodb() -> List[str]:
    """
    Get list of all available vendor names from MongoDB.
    """
    try:
        conn = get_mongodb_connection()
        products_collection = conn['collections']['vendors']
        
        # Get distinct vendor names from products collection
        vendor_names = products_collection.distinct("vendor_name")
        
        # Filter out None/empty values and sort
        valid_vendors = [vendor for vendor in vendor_names if vendor and str(vendor).strip()]
        valid_vendors.sort()
        
        logger.info(f"Retrieved {len(valid_vendors)} vendors from MongoDB")
        return valid_vendors
        
    except Exception as e:
        logger.error(f"Failed to get vendors from MongoDB: {e}")
        return []

def get_vendors_for_product_type(product_type: str) -> List[str]:
    """
    Get list of vendor names that have products for the specified product type.
    """
    try:
        conn = get_mongodb_connection()
        vendors_collection = conn['collections']['vendors']
        
        # Get smart analysis search categories based on product type
        from standardization_utils import get_analysis_search_categories
        search_categories = get_analysis_search_categories(product_type)
        
        logger.info(f"[VENDOR_LOADING] Searching for vendors with product categories: {search_categories}")
        
        # Query for vendors that have products in the relevant categories
        query = {
            '$or': [
                {'product_type': {'$regex': category, '$options': 'i'}}
                for category in search_categories
            ]
        }
        
        # Get distinct vendor names from matching documents
        vendor_names = vendors_collection.distinct("vendor_name", query)
        
        # Filter out None/empty values and sort
        valid_vendors = [vendor for vendor in vendor_names if vendor and str(vendor).strip()]
        valid_vendors.sort()
        
        logger.info(f"Retrieved {len(valid_vendors)} vendors for product type '{product_type}': {valid_vendors[:10]}...")
        return valid_vendors
        
    except Exception as e:
        logger.error(f"Failed to get vendors for product type '{product_type}': {e}")
        # Fallback to all vendors
        return get_available_vendors_from_mongodb()
