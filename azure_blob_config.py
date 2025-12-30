"""
Azure Blob Storage Configuration and Connection Management
Replaces MongoDB with Azure Blob Storage for data persistence
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Azure SDK imports
try:
    from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
    from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
    AZURE_SDK_AVAILABLE = True
except ImportError:
    AZURE_SDK_AVAILABLE = False
    logger.warning("Azure Storage SDK not installed. Run: pip install azure-storage-blob")


# Check if Azure Blob is enabled (allows disabling for local development)
AZURE_BLOB_ENABLED = os.getenv('AZURE_BLOB_ENABLED', 'true').lower() in ('true', '1', 'yes')


class AzureBlobManager:
    """Singleton Azure Blob Storage connection manager with lazy initialization"""

    _instance: Optional['AzureBlobManager'] = None
    _blob_service_client: Optional['BlobServiceClient'] = None
    _container_client: Optional['ContainerClient'] = None
    _connection_attempted: bool = False
    _connection_error: Optional[str] = None

    def __new__(cls) -> 'AzureBlobManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.container_name = os.getenv('AZURE_BLOB_SINGLE_CONTAINER', 'product-data')
            self.base_path = os.getenv('AZURE_BLOB_BASE_PATH', 'Product-Recommender')
            # NOTE: Connection is NOT established here - it's lazy loaded
            logger.info("AzureBlobManager initialized (connection will be established on first use)")

    def _setup_connection(self):
        """Setup Azure Blob Storage connection (lazy - called on first use)"""
        if self._connection_attempted:
            if self._connection_error:
                raise ConnectionError(self._connection_error)
            return

        self._connection_attempted = True

        if not AZURE_BLOB_ENABLED:
            self._connection_error = "Azure Blob Storage is disabled via AZURE_BLOB_ENABLED=false"
            logger.warning(self._connection_error)
            raise ConnectionError(self._connection_error)

        if not AZURE_SDK_AVAILABLE:
            self._connection_error = "Azure Storage SDK not available. Install with: pip install azure-storage-blob"
            raise ImportError(self._connection_error)

        try:
            # Get Azure Storage connection string from environment variables
            connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
            if not connection_string:
                self._connection_error = "AZURE_STORAGE_CONNECTION_STRING not found in environment variables"
                raise ValueError(self._connection_error)

            # Create BlobServiceClient with timeout settings
            self._blob_service_client = BlobServiceClient.from_connection_string(
                connection_string,
                connection_timeout=10,  # 10 second connection timeout
                read_timeout=30  # 30 second read timeout
            )

            # Get or create container
            self._container_client = self._blob_service_client.get_container_client(self.container_name)

            # Ensure container exists
            try:
                self._container_client.get_container_properties()
                logger.info(f"Connected to existing container: {self.container_name}")
            except ResourceNotFoundError:
                self._container_client.create_container()
                logger.info(f"Created new container: {self.container_name}")

            logger.info(f"Successfully connected to Azure Blob Storage container: {self.container_name}")
            logger.info(f"Base path: {self.base_path}")
            self._connection_error = None  # Clear any previous error

        except Exception as e:
            self._connection_error = f"Failed to connect to Azure Blob Storage: {e}"
            logger.error(self._connection_error)
            raise ConnectionError(self._connection_error)

    @property
    def is_available(self) -> bool:
        """Check if Azure Blob Storage is available without raising exception"""
        try:
            if not AZURE_BLOB_ENABLED or not AZURE_SDK_AVAILABLE:
                return False
            if not self._connection_attempted:
                self._setup_connection()
            return self._connection_error is None
        except Exception:
            return False

    @property
    def blob_service_client(self) -> 'BlobServiceClient':
        """Get Blob Service Client (lazy initialization)"""
        if self._blob_service_client is None:
            self._setup_connection()
        if self._blob_service_client is None:
            raise ConnectionError("Azure Blob Storage is not available")
        return self._blob_service_client

    @property
    def container_client(self) -> 'ContainerClient':
        """Get Container Client (lazy initialization)"""
        if self._container_client is None:
            self._setup_connection()
        if self._container_client is None:
            raise ConnectionError("Azure Blob Storage is not available")
        return self._container_client

    def get_blob_client(self, blob_path: str) -> 'BlobClient':
        """Get a blob client for a specific path"""
        full_path = f"{self.base_path}/{blob_path}"
        return self._container_client.get_blob_client(full_path)

    def get_collection_path(self, collection_name: str) -> str:
        """Get the full path for a collection"""
        return f"{self.base_path}/{collection_name}"

    def close(self):
        """Close Azure Blob connection (no-op for Azure SDK, kept for API compatibility)"""
        self._blob_service_client = None
        self._container_client = None
        logger.info("Azure Blob Storage connection closed")


# Global instance (lazy - does NOT connect on import)
azure_blob_manager = AzureBlobManager()


def is_azure_blob_available() -> bool:
    """Check if Azure Blob Storage is available for use"""
    return azure_blob_manager.is_available


# Collection names constants (matching MongoDB structure)
class Collections:
    """Azure Blob collection names (folder paths)"""
    # Main collections
    SPECS = "specs"                          # Product type schemas
    VENDORS = "vendors"                      # Vendor product data
    ADVANCED_PARAMETERS = "advanced_parameters"  # Cached advanced parameters
    IMAGES = "images"                        # Cached product images
    GENERIC_IMAGES = "generic_images"        # Cached generic product type images
    VENDOR_LOGOS = "vendor_logos"            # Cached vendor logos

    # User project management
    USER_PROJECTS = "user_projects"          # User saved projects

    # File storage
    FILES = "files"                          # Binary file storage (replaces GridFS)

    # Index files for fast lookups
    INDEXES = "indexes"                      # Index metadata


class AzureBlobCollection:
    """Wrapper class that mimics MongoDB collection interface for Azure Blob Storage"""

    def __init__(self, container_client, base_path: str, collection_name: str):
        self.container_client = container_client
        self.base_path = base_path
        self.collection_name = collection_name
        self.collection_path = f"{base_path}/{collection_name}"

    def find_one(self, query: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find a single document matching the query"""
        try:
            # List blobs and search for matching document
            blobs = self.container_client.list_blobs(
                name_starts_with=self.collection_path,
                include=['metadata']
            )

            for blob in blobs:
                if blob.name.endswith('.metadata') or not blob.name.endswith('.json'):
                    continue

                blob_metadata = blob.metadata or {}

                # Check if query matches
                matches = True
                for key, value in query.items():
                    blob_value = blob_metadata.get(key, '')
                    if isinstance(value, dict):
                        # Handle regex queries (simplified)
                        if '$regex' in value:
                            import re
                            pattern = value['$regex']
                            flags = re.IGNORECASE if value.get('$options') == 'i' else 0
                            if not re.search(pattern, str(blob_value), flags):
                                matches = False
                                break
                    else:
                        if str(value).lower() not in str(blob_value).lower():
                            matches = False
                            break

                if matches:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    data = json.loads(blob_client.download_blob().readall().decode('utf-8'))
                    data['_blob_name'] = blob.name
                    return data

            return None

        except Exception as e:
            logger.error(f"find_one error in {self.collection_name}: {e}")
            return None

    def find(self, query: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Find all documents matching the query"""
        try:
            query = query or {}
            results = []

            blobs = self.container_client.list_blobs(
                name_starts_with=self.collection_path,
                include=['metadata']
            )

            for blob in blobs:
                if blob.name.endswith('.metadata') or not blob.name.endswith('.json'):
                    continue

                blob_metadata = blob.metadata or {}

                # Check if query matches (empty query matches all)
                matches = True
                for key, value in query.items():
                    blob_value = blob_metadata.get(key, '')
                    if str(value).lower() not in str(blob_value).lower():
                        matches = False
                        break

                if matches:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    data = json.loads(blob_client.download_blob().readall().decode('utf-8'))
                    data['_blob_name'] = blob.name
                    results.append(data)

            return results

        except Exception as e:
            logger.error(f"find error in {self.collection_name}: {e}")
            return []

    def insert_one(self, document: Dict[str, Any]) -> Any:
        """Insert a single document"""
        try:
            from azure.storage.blob import ContentSettings
            import uuid

            # Generate document ID
            doc_id = document.get('_id', str(uuid.uuid4()))
            filename = f"{doc_id}.json"
            blob_path = f"{self.collection_path}/{filename}"

            # Prepare metadata
            blob_metadata = {}
            for key in ['product_type', 'vendor_name', 'product_type_normalized']:
                if key in document:
                    blob_metadata[key] = str(document[key])

            blob_metadata['created_at'] = datetime.utcnow().isoformat()

            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(
                json.dumps(document, indent=2, default=str),
                overwrite=True,
                metadata=blob_metadata,
                content_settings=ContentSettings(content_type='application/json')
            )

            # Return object with inserted_id attribute
            class InsertResult:
                def __init__(self, id):
                    self.inserted_id = id

            return InsertResult(doc_id)

        except Exception as e:
            logger.error(f"insert_one error in {self.collection_name}: {e}")
            raise

    def update_one(self, query: Dict[str, Any], update: Dict[str, Any], upsert: bool = False) -> Any:
        """Update a single document"""
        try:
            from azure.storage.blob import ContentSettings

            # Find existing document
            existing = self.find_one(query)

            if existing:
                # Update document
                if '$set' in update:
                    existing.update(update['$set'])
                else:
                    existing.update(update)

                blob_path = existing.get('_blob_name')
                if blob_path:
                    # Remove internal fields before saving
                    save_doc = {k: v for k, v in existing.items() if not k.startswith('_')}

                    blob_metadata = {}
                    for key in ['product_type', 'vendor_name', 'product_type_normalized']:
                        if key in save_doc:
                            blob_metadata[key] = str(save_doc[key])

                    blob_client = self.container_client.get_blob_client(blob_path)
                    blob_client.upload_blob(
                        json.dumps(save_doc, indent=2, default=str),
                        overwrite=True,
                        metadata=blob_metadata,
                        content_settings=ContentSettings(content_type='application/json')
                    )

                class UpdateResult:
                    def __init__(self):
                        self.modified_count = 1
                        self.upserted_id = None

                return UpdateResult()

            elif upsert:
                # Insert new document
                new_doc = {}
                new_doc.update(query)
                if '$set' in update:
                    new_doc.update(update['$set'])
                else:
                    new_doc.update(update)

                result = self.insert_one(new_doc)

                class UpsertResult:
                    def __init__(self, id):
                        self.modified_count = 0
                        self.upserted_id = id

                return UpsertResult(result.inserted_id)

            class NoMatchResult:
                def __init__(self):
                    self.modified_count = 0
                    self.upserted_id = None

            return NoMatchResult()

        except Exception as e:
            logger.error(f"update_one error in {self.collection_name}: {e}")
            raise

    def delete_one(self, query: Dict[str, Any]) -> bool:
        """Delete a single document"""
        try:
            existing = self.find_one(query)
            if existing and '_blob_name' in existing:
                blob_client = self.container_client.get_blob_client(existing['_blob_name'])
                blob_client.delete_blob()
                return True
            return False
        except Exception as e:
            logger.error(f"delete_one error in {self.collection_name}: {e}")
            return False

    def distinct(self, field: str, query: Dict[str, Any] = None) -> List[str]:
        """Get distinct values for a field"""
        try:
            documents = self.find(query or {})
            values = set()
            for doc in documents:
                if field in doc:
                    values.add(doc[field])
                elif 'metadata' in doc and field in doc['metadata']:
                    values.add(doc['metadata'][field])
            return list(values)
        except Exception as e:
            logger.error(f"distinct error in {self.collection_name}: {e}")
            return []

    def create_index(self, *args, **kwargs):
        """No-op for Azure Blob (indexes not needed)"""
        pass


class AzureBlobGridFS:
    """Wrapper class that mimics GridFS interface for Azure Blob Storage"""

    def __init__(self, container_client, base_path: str):
        self.container_client = container_client
        self.base_path = base_path
        self.files_path = f"{base_path}/files"

    def put(self, data: bytes, filename: str = None, content_type: str = None, **metadata) -> str:
        """Store a file and return its ID"""
        try:
            from azure.storage.blob import ContentSettings
            import uuid

            file_id = str(uuid.uuid4())
            filename = filename or f"file_{file_id}"
            blob_path = f"{self.files_path}/{file_id}_{filename}"

            blob_metadata = {
                'file_id': file_id,
                'filename': filename,
                'upload_date': datetime.utcnow().isoformat(),
                **{k: str(v) for k, v in metadata.items() if v is not None}
            }

            blob_client = self.container_client.get_blob_client(blob_path)
            blob_client.upload_blob(
                data,
                overwrite=True,
                metadata=blob_metadata,
                content_settings=ContentSettings(content_type=content_type or 'application/octet-stream')
            )

            return file_id

        except Exception as e:
            logger.error(f"GridFS put error: {e}")
            raise

    def get(self, file_id: str):
        """Get a file by ID"""
        try:
            # Search for file with matching ID
            blobs = self.container_client.list_blobs(
                name_starts_with=self.files_path,
                include=['metadata']
            )

            for blob in blobs:
                blob_metadata = blob.metadata or {}
                if blob_metadata.get('file_id') == str(file_id) or str(file_id) in blob.name:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    return blob_client.download_blob()

            raise FileNotFoundError(f"File not found: {file_id}")

        except Exception as e:
            logger.error(f"GridFS get error: {e}")
            raise

    def find_one(self, query: Dict[str, Any]):
        """Find a single file matching the query"""
        try:
            blobs = self.container_client.list_blobs(
                name_starts_with=self.files_path,
                include=['metadata']
            )

            for blob in blobs:
                blob_metadata = blob.metadata or {}

                matches = True
                for key, value in query.items():
                    if str(value).lower() not in str(blob_metadata.get(key, '')).lower():
                        matches = False
                        break

                if matches:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    return blob_client.download_blob()

            return None

        except Exception as e:
            logger.error(f"GridFS find_one error: {e}")
            return None

    def delete(self, file_id: str):
        """Delete a file by ID"""
        try:
            blobs = self.container_client.list_blobs(
                name_starts_with=self.files_path,
                include=['metadata']
            )

            for blob in blobs:
                blob_metadata = blob.metadata or {}
                if blob_metadata.get('file_id') == str(file_id) or str(file_id) in blob.name:
                    blob_client = self.container_client.get_blob_client(blob.name)
                    blob_client.delete_blob()
                    return True

            return False

        except Exception as e:
            logger.error(f"GridFS delete error: {e}")
            return False


def get_azure_blob_connection():
    """Get Azure Blob connection components (MongoDB API compatible)"""
    container_client = azure_blob_manager.container_client
    base_path = azure_blob_manager.base_path

    # Create collection wrappers
    collections = {
        'specs': AzureBlobCollection(container_client, base_path, Collections.SPECS),
        'vendors': AzureBlobCollection(container_client, base_path, Collections.VENDORS),
        'advanced_parameters': AzureBlobCollection(container_client, base_path, Collections.ADVANCED_PARAMETERS),
        'images': AzureBlobCollection(container_client, base_path, Collections.IMAGES),
        'generic_images': AzureBlobCollection(container_client, base_path, Collections.GENERIC_IMAGES),
        'vendor_logos': AzureBlobCollection(container_client, base_path, Collections.VENDOR_LOGOS),
        'user_projects': AzureBlobCollection(container_client, base_path, Collections.USER_PROJECTS),
    }

    # Create GridFS wrapper
    gridfs = AzureBlobGridFS(container_client, base_path)

    return {
        'blob_service_client': azure_blob_manager.blob_service_client,
        'container_client': container_client,
        'base_path': base_path,
        'container_name': azure_blob_manager.container_name,
        'collections': collections,
        'gridfs': gridfs,
        'database': None  # Not applicable for Azure Blob
    }


def ensure_collection_structure():
    """Create collection folder structure in Azure Blob Storage"""
    try:
        conn = get_azure_blob_connection()
        container_client = conn['container_client']
        base_path = conn['base_path']

        # Create placeholder files to ensure folder structure
        collections = [
            Collections.SPECS,
            Collections.VENDORS,
            Collections.ADVANCED_PARAMETERS,
            Collections.IMAGES,
            Collections.GENERIC_IMAGES,
            Collections.VENDOR_LOGOS,
            Collections.USER_PROJECTS,
            Collections.FILES,
            Collections.INDEXES
        ]

        for collection in collections:
            # Create a .metadata file in each collection folder
            metadata_path = f"{base_path}/{collection}/.metadata"
            blob_client = container_client.get_blob_client(metadata_path)

            try:
                blob_client.get_blob_properties()
            except ResourceNotFoundError:
                # Create the metadata file
                metadata = {
                    "collection": collection,
                    "created_at": datetime.utcnow().isoformat(),
                    "type": "collection_metadata"
                }
                blob_client.upload_blob(
                    json.dumps(metadata),
                    overwrite=True,
                    content_type='application/json'
                )
                logger.info(f"Created collection structure: {collection}")

        logger.info("Azure Blob collection structure ensured")

    except Exception as e:
        logger.error(f"Failed to create Azure Blob collection structure: {e}")
        raise


# NOTE: Collection structure is NOT initialized on import to allow lazy loading
# Call ensure_collection_structure() manually if you need to set up the folder structure
# Example: azure_blob_config.ensure_collection_structure()
