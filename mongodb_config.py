"""
MongoDB Configuration and Connection Management
Handles MongoDB connection, GridFS setup, and database operations
"""

import os
import logging
from typing import Optional, Dict, Any
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from gridfs import GridFS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MongoDBManager:
    """Singleton MongoDB connection manager with GridFS support"""
    
    _instance: Optional['MongoDBManager'] = None
    _client: Optional[MongoClient] = None
    _database: Optional[Database] = None
    _gridfs: Optional[GridFS] = None
    
    def __new__(cls) -> 'MongoDBManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self._setup_connection()
    
    def _setup_connection(self):
        """Setup MongoDB connection and GridFS"""
        try:
            # Get MongoDB URI from environment variables
            mongodb_uri = os.getenv('MONGODB_URI')
            # if not mongodb_uri:
            #     # Fallback to local MongoDB
            #     mongodb_uri = 'mongodb://localhost:27017/'
            #     logger.warning("MONGODB_URI not found in environment, using local MongoDB")
            
            # Database name
            db_name = os.getenv('MONGODB_DATABASE', 'product-recommender')
            
            # Create MongoDB client with timeout settings
            self._client = MongoClient(
            mongodb_uri
            )
            self._database = self._client[db_name]
            
            # Setup GridFS for file storage
            self._gridfs = GridFS(self._database)
            
            # Test connection
            self._client.admin.command('ping')
            logger.info(f"Successfully connected to MongoDB database: {db_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise
    
    @property
    def client(self) -> MongoClient:
        """Get MongoDB client"""
        if self._client is None:
            self._setup_connection()
        return self._client
    
    @property
    def database(self) -> Database:
        """Get MongoDB database"""
        if self._database is None:
            self._setup_connection()
        return self._database
    
    @property
    def gridfs(self) -> GridFS:
        """Get GridFS instance"""
        if self._gridfs is None:
            self._setup_connection()
        return self._gridfs
    
    def get_collection(self, name: str) -> Collection:
        """Get a specific collection"""
        return self.database[name]
    
    def close(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            self._client = None
            self._database = None
            self._gridfs = None
            logger.info("MongoDB connection closed")

# Global instance
mongodb_manager = MongoDBManager()

# Collection names constants
class Collections:
    """MongoDB collection names"""
    # Main collections
    SPECS = "specs"                          # Product type schemas
    VENDORS = "vendors"                      # Vendor product data
    ADVANCED_PARAMETERS = "advanced_parameters"  # Cached advanced parameters
    IMAGES = "images"                        # Cached product images
    GENERIC_IMAGES = "generic_images"        # Cached generic product type images
    VENDOR_LOGOS = "vendor_logos"            # Cached vendor logos
    
    # User project management
    USER_PROJECTS = "user_projects"          # User saved projects

def get_mongodb_connection():
    """Get MongoDB connection components"""
    return {
        'client': mongodb_manager.client,
        'database': mongodb_manager.database,
        'gridfs': mongodb_manager.gridfs,
        'collections': {
            # Main collections
            'specs': mongodb_manager.get_collection(Collections.SPECS),
            'vendors': mongodb_manager.get_collection(Collections.VENDORS),
            'advanced_parameters': mongodb_manager.get_collection(Collections.ADVANCED_PARAMETERS),
            'images': mongodb_manager.get_collection(Collections.IMAGES),
            'generic_images': mongodb_manager.get_collection(Collections.GENERIC_IMAGES),
            'vendor_logos': mongodb_manager.get_collection(Collections.VENDOR_LOGOS),
            
            # User project management
            'user_projects': mongodb_manager.get_collection(Collections.USER_PROJECTS),
        }
    }

def ensure_indexes():
    """Create necessary indexes for efficient querying"""
    try:
        conn = get_mongodb_connection()
        collections = conn['collections']
        db = conn['database']
        
        # Specs collection indexes
        collections['specs'].create_index([
            ("product_type", 1)
        ])
        collections['specs'].create_index([
            ("metadata.product_type", 1)
        ])
        
        # Vendors collection indexes
        collections['vendors'].create_index([
            ("product_type", 1),
            ("vendor", 1)
        ])
        collections['vendors'].create_index([
            ("metadata.product_type", 1),
            ("metadata.vendor_name", 1)
        ])
        
        # User projects collection indexes
        collections['user_projects'].create_index([
            ("user_id", 1),
            ("project_status", 1),
            ("updated_at", -1)
        ])
        collections['user_projects'].create_index([
            ("user_id", 1),
            ("project_name", 1)
        ])

        # Advanced parameters cache indexes
        # Advanced parameters cache indexes
        collections['advanced_parameters'].create_index([
            ("product_type", 1)
        ], unique=True)
        # Drop TTL index if it exists (we want permanent storage now)
        try:
            collections['advanced_parameters'].drop_index("created_at_1")
            logger.info("Dropped TTL index 'created_at_1' from advanced_parameters")
        except Exception:
            pass # Index probably doesn't exist
            
        # collections['advanced_parameters'].create_index(
        #     "created_at",
        #     expireAfterSeconds=60 * 60 * 24 * 30  # ~30 days
        # )
        
        # Images cache indexes
        collections['images'].create_index([
            ("vendor_name_normalized", 1),
            ("model_family_normalized", 1)
        ], unique=True)
        collections['images'].create_index(
            "created_at",
            expireAfterSeconds=60 * 60 * 24 * 90  # ~90 days
        )
        
        # Generic images cache indexes
        collections['generic_images'].create_index(
            "product_type_normalized",
            unique=True
        )
        collections['generic_images'].create_index(
            "created_at",
            expireAfterSeconds=60 * 60 * 24 * 90  # ~90 days
        )
        
        # Vendor logos cache indexes
        collections['vendor_logos'].create_index(
            "vendor_name_normalized",
            unique=True
        )
        collections['vendor_logos'].create_index(
            "created_at",
            expireAfterSeconds=60 * 60 * 24 * 90  # ~90 days
        )
        
        # GridFS indexes (fs.files and fs.chunks are auto-created by GridFS)
        # Add custom indexes for fs.files for better query performance
        fs_files = db['fs.files']
        fs_files.create_index([
            ("metadata.product_type", 1),
            ("metadata.vendor", 1)
        ])
        fs_files.create_index([
            ("metadata.collection_type", 1)
        ])
        
        logger.info("MongoDB indexes created successfully for all active collections")
        
    except Exception as e:
        logger.error(f"Failed to create MongoDB indexes: {e}")
        raise

# Initialize indexes on import
try:
    ensure_indexes()
except Exception as e:
    logger.warning(f"Could not initialize MongoDB indexes: {e}")
