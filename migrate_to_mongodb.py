"""
Migration Script: Local File Storage to MongoDB
Migrates existing local files (documents, vendors, specs) to MongoDB
Only migrates documents, specs, and vendors folders - no image handling.
"""

import os
import json
import logging
from typing import Dict, Any, List
from glob import glob
from pathlib import Path

from mongodb_utils import upload_to_mongodb, upload_json_to_mongodb, mongodb_file_manager
from mongodb_config import get_mongodb_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalToMongoDBMigrator:
    """Handles migration of local files to MongoDB"""
    
    def __init__(self):
        self.conn = get_mongodb_connection()
        self.stats = {
            'documents_migrated': 0,
            'vendors_migrated': 0,
            'specs_migrated': 0,
            'errors': 0
        }
    
    def migrate_all(self):
        """Migrate all local data to MongoDB"""
        logger.info("Starting migration from local filesystem to MongoDB...")
        logger.info("Migrating only documents, specs, and vendors folders")
        
        try:
            # Migrate in order: specs, documents, vendors
            self.migrate_specs()
            self.migrate_documents()
            self.migrate_vendors()
            
            # Print final statistics
            self.print_migration_stats()
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    def migrate_specs(self):
        """Migrate specs/*.json files to MongoDB"""
        logger.info("Migrating specs files...")
        specs_dir = "specs"
        
        if not os.path.exists(specs_dir):
            logger.info("No specs directory found, skipping...")
            return
        
        for json_file in glob(os.path.join(specs_dir, "*.json")):
            try:
                # Extract product type from filename
                filename = os.path.basename(json_file)
                product_type = filename.replace('.json', '').replace('_', ' ').title()
                
                # Load JSON data
                with open(json_file, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                
                # Prepare metadata
                metadata = {
                    'collection_type': 'specs',
                    'product_type': product_type,
                    'normalized_product_type': product_type.lower().replace(' ', '').replace('_', ''),
                    'filename': filename,
                    'file_type': 'json',
                    'schema_version': '1.0',
                    'migrated_from': json_file
                }
                
                # Upload to MongoDB
                doc_id = upload_json_to_mongodb(schema_data, metadata)
                logger.info(f"Migrated spec: {json_file} -> MongoDB ID: {doc_id}")
                self.stats['specs_migrated'] += 1
                
            except Exception as e:
                logger.error(f"Failed to migrate spec {json_file}: {e}")
                self.stats['errors'] += 1
    
    def migrate_documents(self):
        """Migrate documents/ folder structure to MongoDB"""
        logger.info("Migrating documents...")
        documents_dir = "documents"
        
        if not os.path.exists(documents_dir):
            logger.info("No documents directory found, skipping...")
            return
        
        # Structure: documents/vendor_name/product_type/files (OLD STRUCTURE)
        for vendor_dir in os.listdir(documents_dir):
            vendor_path = os.path.join(documents_dir, vendor_dir)
            if not os.path.isdir(vendor_path):
                continue
                
            for product_dir in os.listdir(vendor_path):
                product_path = os.path.join(vendor_path, product_dir)
                if not os.path.isdir(product_path):
                    continue
                
                # Process all files in this product directory
                for file_name in os.listdir(product_path):
                    file_path = os.path.join(product_path, file_name)
                    if os.path.isfile(file_path):
                        self._migrate_document_file(file_path, vendor_dir, product_dir, file_name)
    
    def _migrate_document_file(self, file_path: str, vendor_name: str, product_type: str, filename: str):
        """Migrate a single document file"""
        try:
            # Determine file type
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.pdf':
                # Handle PDF files
                # Structure: documents/{vendor}/{product_type}/{filename}
                safe_product_type = product_type.replace(' ', '_')
                safe_vendor = vendor_name.replace(' ', '_')
                metadata = {
                    'collection_type': 'documents',
                    'product_type': product_type.replace('_', ' '),
                    'vendor_name': vendor_name.replace('_', ' '),
                    'filename': filename,
                    'file_type': 'pdf',
                    'path': f'documents/{safe_vendor}/{safe_product_type}/{filename}',
                    'migrated_from': file_path
                }
                
                file_id = upload_to_mongodb(file_path, metadata)
                logger.info(f"Migrated PDF: {file_path} -> MongoDB ID: {file_id}")
                self.stats['documents_migrated'] += 1
                
            elif file_ext == '.json':
                # Handle JSON files
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                metadata = {
                    'collection_type': 'documents',
                    'product_type': product_type.replace('_', ' '),
                    'vendor_name': vendor_name.replace('_', ' '),
                    'filename': filename,
                    'file_type': 'json',
                    'migrated_from': file_path
                }
                
                doc_id = upload_json_to_mongodb(json_data, metadata)
                logger.info(f"Migrated JSON: {file_path} -> MongoDB ID: {doc_id}")
                self.stats['documents_migrated'] += 1
            
            else:
                # Handle other file types as binary
                metadata = {
                    'collection_type': 'documents',
                    'product_type': product_type.replace('_', ' '),
                    'vendor_name': vendor_name.replace('_', ' '),
                    'filename': filename,
                    'file_type': file_ext.replace('.', '') or 'unknown',
                    'migrated_from': file_path
                }
                
                file_id = upload_to_mongodb(file_path, metadata)
                logger.info(f"Migrated file: {file_path} -> MongoDB ID: {file_id}")
                self.stats['documents_migrated'] += 1
                
        except Exception as e:
            logger.error(f"Failed to migrate document {file_path}: {e}")
            self.stats['errors'] += 1
    
    def migrate_vendors(self):
        """Migrate vendors/ folder structure to MongoDB"""
        logger.info("Migrating vendors...")
        vendors_dir = "vendors"
        
        if not os.path.exists(vendors_dir):
            logger.info("No vendors directory found, skipping...")
            return
        
        # Structure: vendors/vendor_name/product_type/model_files.json (OLD STRUCTURE)
        for vendor_dir in os.listdir(vendors_dir):
            vendor_path = os.path.join(vendors_dir, vendor_dir)
            if not os.path.isdir(vendor_path):
                continue
                
            for product_dir in os.listdir(vendor_path):
                product_path = os.path.join(vendor_path, product_dir)
                if not os.path.isdir(product_path):
                    continue
                
                # Process all JSON files in this product directory
                for json_file in glob(os.path.join(product_path, "*.json")):
                    self._migrate_vendor_file(json_file, vendor_dir, product_dir)
    
    def _migrate_vendor_file(self, file_path: str, vendor_name: str, product_type: str):
        """Migrate a single vendor JSON file"""
        try:
            filename = os.path.basename(file_path)
            model_series = os.path.splitext(filename)[0]
            
            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                vendor_data = json.load(f)
            
            # Prepare metadata
            # Structure: vendors/{vendor}/{product_type}/{model}.json
            safe_product_type = product_type.replace(' ', '_')
            safe_vendor = vendor_name.replace(' ', '_')
            metadata = {
                'collection_type': 'vendors',  # Changed from 'products' to 'vendors'
                'product_type': product_type.replace('_', ' '),
                'vendor_name': vendor_name.replace('_', ' '),
                'model_series': model_series.replace('_', ' '),
                'filename': filename,
                'file_type': 'json',
                'path': f'vendors/{safe_vendor}/{safe_product_type}/{filename}',
                'migrated_from': file_path
            }
            
            # Upload to MongoDB
            doc_id = upload_json_to_mongodb(vendor_data, metadata)
            logger.info(f"Migrated vendor: {file_path} -> MongoDB ID: {doc_id}")
            self.stats['vendors_migrated'] += 1
            
        except Exception as e:
            logger.error(f"Failed to migrate vendor file {file_path}: {e}")
            self.stats['errors'] += 1
    
    def print_migration_stats(self):
        """Print migration statistics"""
        logger.info("=" * 60)
        logger.info("MIGRATION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Documents migrated: {self.stats['documents_migrated']}")
        logger.info(f"Vendors migrated: {self.stats['vendors_migrated']}")
        logger.info(f"Specs migrated: {self.stats['specs_migrated']}")
        logger.info(f"Errors encountered: {self.stats['errors']}")
        logger.info("Only documents, specs, and vendors folders were migrated")
        logger.info("=" * 60)
        
        total_migrated = (self.stats['documents_migrated'] + 
                         self.stats['vendors_migrated'] + 
                         self.stats['specs_migrated'])
        
        if total_migrated > 0:
            logger.info(f"Successfully migrated {total_migrated} items to MongoDB!")
        else:
            logger.info("No items were migrated. Check if local directories exist.")

def main():
    """Main migration function"""
    try:
        migrator = LocalToMongoDBMigrator()
        migrator.migrate_all()
        
        print("\n" + "="*60)
        print("MIGRATION INSTRUCTIONS")
        print("="*60)
        print("1. Verify the migration was successful by checking MongoDB collections")
        print("2. Test the application to ensure it works with MongoDB")
        print("3. Once confirmed, you can safely remove the local directories:")
        print("   - documents/")
        print("   - vendors/")
        print("   - specs/")
        print("4. Update your deployment to include MongoDB connection string")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        print("\nMigration failed! Please check the logs and fix any issues before retrying.")

if __name__ == "__main__":
    main()
