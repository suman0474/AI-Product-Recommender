"""
MongoDB Project Management
Handles project storage and retrieval using MongoDB
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from azure_blob_config import get_azure_blob_connection as get_mongodb_connection

logger = logging.getLogger(__name__)

class MongoDBProjectManager:
    """MongoDB-based project management"""
    
    def __init__(self):
        self.conn = get_mongodb_connection()
        self.projects_collection = self.conn['collections']['user_projects']
    
    def save_project(self, user_id: str, project_data: Dict[str, Any], project_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Save or update a project in MongoDB
        
        Args:
            user_id: User ID who owns the project
            project_data: Project data to save
            project_id: Optional project ID for updates
            
        Returns:
            Dict containing saved project info
        """
        try:
            # Prepare project document
            project_doc = {
                'user_id': user_id,
                'project_name': project_data.get('project_name', ''),
                'project_description': project_data.get('project_description', ''),
                'initial_requirements': project_data.get('initial_requirements', ''),
                'product_type': project_data.get('product_type', ''),
                'identified_instruments': project_data.get('identified_instruments', []),
                'identified_accessories': project_data.get('identified_accessories', []),
                'search_tabs': project_data.get('search_tabs', []),
                'conversation_histories': project_data.get('conversation_histories', {}),
                'collected_data': project_data.get('collected_data', {}),
                'current_step': project_data.get('current_step', ''),
                'analysis_results': project_data.get('analysis_results', {}),
                'project_status': project_data.get('project_status', 'active'),
                'updated_at': datetime.utcnow()
            }
            
            if project_id:
                # Update existing project
                project_doc['_id'] = ObjectId(project_id)
                result = self.projects_collection.replace_one(
                    {'_id': ObjectId(project_id), 'user_id': user_id},
                    project_doc
                )
                
                if result.matched_count == 0:
                    raise ValueError("Project not found or access denied")
                    
                saved_project_id = project_id
                logger.info(f"Updated project {project_id} for user {user_id}")
                
            else:
                # Create new project
                project_doc['created_at'] = datetime.utcnow()
                result = self.projects_collection.insert_one(project_doc)
                saved_project_id = str(result.inserted_id)
                logger.info(f"Created new project {saved_project_id} for user {user_id}")
            
            return {
                'id': saved_project_id,
                'project_name': project_doc['project_name'],
                'project_description': project_doc['project_description'],
                'created_at': project_doc.get('created_at', project_doc['updated_at']).isoformat(),
                'updated_at': project_doc['updated_at'].isoformat(),
                'project_status': project_doc['project_status']
            }
            
        except Exception as e:
            logger.error(f"Failed to save project: {e}")
            raise
    
    def get_user_projects(self, user_id: str, project_status: str = 'active') -> List[Dict[str, Any]]:
        """
        Get all projects for a user
        
        Args:
            user_id: User ID
            project_status: Project status filter (default: 'active')
            
        Returns:
            List of project summaries
        """
        try:
            cursor = self.projects_collection.find(
                {'user_id': user_id, 'project_status': project_status}
            ).sort('updated_at', -1)
            
            projects = []
            for project in cursor:
                # Calculate counts
                instruments_count = len(project.get('identified_instruments', []))
                accessories_count = len(project.get('identified_accessories', []))
                search_tabs_count = len(project.get('search_tabs', []))
                
                # Create requirements preview
                requirements = project.get('initial_requirements', '')
                requirements_preview = requirements[:200] + "..." if len(requirements) > 200 else requirements
                
                project_summary = {
                    'id': str(project['_id']),
                    'project_name': project.get('project_name', ''),
                    'project_description': project.get('project_description', ''),
                    'product_type': project.get('product_type', ''),
                    'instruments_count': instruments_count,
                    'accessories_count': accessories_count,
                    'search_tabs_count': search_tabs_count,
                    'current_step': project.get('current_step', ''),
                    'project_status': project.get('project_status', 'active'),
                    'created_at': project.get('created_at', datetime.utcnow()).isoformat(),
                    'updated_at': project.get('updated_at', datetime.utcnow()).isoformat(),
                    'requirements_preview': requirements_preview
                }
                projects.append(project_summary)
            
            logger.info(f"Retrieved {len(projects)} projects for user {user_id}")
            return projects
            
        except Exception as e:
            logger.error(f"Failed to get user projects: {e}")
            raise
    
    def get_project_details(self, user_id: str, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Get full project details
        
        Args:
            user_id: User ID
            project_id: Project ID
            
        Returns:
            Dict containing full project details or None if not found
        """
        try:
            project = self.projects_collection.find_one({
                '_id': ObjectId(project_id),
                'user_id': user_id
            })
            
            if not project:
                return None
            
            project_details = {
                'id': str(project['_id']),
                'project_name': project.get('project_name', ''),
                'project_description': project.get('project_description', ''),
                'initial_requirements': project.get('initial_requirements', ''),
                'product_type': project.get('product_type', ''),
                'identified_instruments': project.get('identified_instruments', []),
                'identified_accessories': project.get('identified_accessories', []),
                'search_tabs': project.get('search_tabs', []),
                'conversation_history': project.get('conversation_histories', {}),
                'collected_data': project.get('collected_data', {}),
                'current_step': project.get('current_step', ''),
                'analysis_results': project.get('analysis_results', {}),
                'project_status': project.get('project_status', 'active'),
                'created_at': project.get('created_at', datetime.utcnow()).isoformat(),
                'updated_at': project.get('updated_at', datetime.utcnow()).isoformat()
            }
            
            logger.info(f"Retrieved project details for {project_id}")
            return project_details
            
        except Exception as e:
            logger.error(f"Failed to get project details: {e}")
            raise
    
    def delete_project(self, user_id: str, project_id: str) -> bool:
        """
        Delete (archive) a project
        
        Args:
            user_id: User ID
            project_id: Project ID
            
        Returns:
            bool: True if successfully deleted
        """
        try:
            # Soft delete by changing status to archived
            result = self.projects_collection.update_one(
                {'_id': ObjectId(project_id), 'user_id': user_id},
                {
                    '$set': {
                        'project_status': 'archived',
                        'updated_at': datetime.utcnow()
                    }
                }
            )
            
            if result.matched_count == 0:
                return False
            
            logger.info(f"Archived project {project_id} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            raise

# Global instance
mongodb_project_manager = MongoDBProjectManager()

# Convenience functions
def save_project_to_mongodb(user_id: str, project_data: Dict[str, Any], project_id: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function for saving projects"""
    return mongodb_project_manager.save_project(user_id, project_data, project_id)

def get_user_projects_from_mongodb(user_id: str, project_status: str = 'active') -> List[Dict[str, Any]]:
    """Convenience function for getting user projects"""
    return mongodb_project_manager.get_user_projects(user_id, project_status)

def get_project_details_from_mongodb(user_id: str, project_id: str) -> Optional[Dict[str, Any]]:
    """Convenience function for getting project details"""
    return mongodb_project_manager.get_project_details(user_id, project_id)

def delete_project_from_mongodb(user_id: str, project_id: str) -> bool:
    """Convenience function for deleting projects"""
    return mongodb_project_manager.delete_project(user_id, project_id)