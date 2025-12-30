"""
MongoDB Project Management Module
Handles project storage and retrieval in MongoDB using GridFS
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from azure_blob_utils import MongoDBFileManager
from bson import ObjectId
import json

class MongoProjectManager:
    """Manages project operations in MongoDB with GridFS storage"""
    
    def __init__(self):
        self.file_manager = MongoDBFileManager()
        self.collection_name = 'user_projects'
        self.logger = logging.getLogger(__name__)
    
    def save_project(self, user_id: str, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save or update a project in MongoDB using GridFS
        
        Args:
            user_id: ID of the user who owns the project
            project_data: Project data to save
            
        Returns:
            Dictionary with project ID and metadata
        """
        try:
            project_id = project_data.get('project_id')
            current_time = datetime.utcnow()
            
            # Ensure detected product type is used, and avoid accidentally saving project_name as product_type
            project_name = (project_data.get('project_name') or '').strip()
            detected_product_type = project_data.get('detected_product_type')
            incoming_product_type = (project_data.get('product_type') or '').strip()

            if detected_product_type:
                product_type = detected_product_type.strip()
            else:
                # If incoming product_type exactly matches project_name, ignore it
                if incoming_product_type and project_name and incoming_product_type.lower() == project_name.lower():
                    self.logger.warning(f"Incoming product_type matches project_name ('{incoming_product_type}'). Clearing product_type to avoid incorrect storage for user {user_id}.")
                    product_type = ''
                else:
                    product_type = incoming_product_type

            # Build complete project data for GridFS storage (no serialization needed)
            complete_project_data = {
                'project_name': project_data.get('project_name', ''),
                'project_description': project_data.get('project_description', ''),
                'initial_requirements': project_data.get('initial_requirements', ''),
                'product_type': product_type,
                'pricing': project_data.get('pricing', {}),
                'identified_instruments': project_data.get('identified_instruments', []),
                'identified_accessories': project_data.get('identified_accessories', []),
                'search_tabs': project_data.get('search_tabs', []),
                'conversation_histories': project_data.get('conversation_histories', {}),
                'collected_data': project_data.get('collected_data', {}),
                'generic_images': project_data.get('generic_images', {}),
                'feedback_entries': project_data.get('feedback_entries', project_data.get('feedback', [])),
                'current_step': project_data.get('current_step', ''),
                'active_tab': project_data.get('active_tab', ''),
                'analysis_results': project_data.get('analysis_results', {}),
                'field_descriptions': project_data.get('field_descriptions', {}),
                'workflow_position': project_data.get('workflow_position', {}),
                'user_interactions': project_data.get('user_interactions', {}),
                'embedded_media': project_data.get('embedded_media', {}),
                'project_metadata': {
                    'schema_version': '2.0',
                    'storage_format': 'gridfs',
                    'data_structure_description': 'Complete project state stored in GridFS',
                    'supported_features': ['multi_tab_search', 'conversation_persistence', 'state_restoration', 'analysis_results'],
                    'last_updated_by': 'ai_product_recommender_system'
                }
            }
            
            # Convert to JSON bytes for GridFS storage
            project_json = json.dumps(complete_project_data, ensure_ascii=False, indent=2)
            project_bytes = project_json.encode('utf-8')
            
            # Log the product_type we plan to save for debugging
            self.logger.info(f"Saving project to GridFS - detected_product_type='{detected_product_type}' resolved_product_type='{product_type}' for user {user_id}")

            if project_id:
                # Update existing project
                try:
                    object_id = ObjectId(project_id)
                    existing_project = self.file_manager.db[self.collection_name].find_one({
                        '_id': object_id,
                        'user_id': str(user_id)
                    })
                    
                    if not existing_project:
                        raise ValueError("Project not found or access denied")
                    
                    # Check for duplicate project name if name changed
                    if existing_project.get('project_name', '').strip().lower() != project_name.lower():
                        existing_duplicate = self.file_manager.db[self.collection_name].find_one({
                            'user_id': str(user_id),
                            'project_name': project_name,
                            'project_status': 'active'
                        })
                        
                        if existing_duplicate and str(existing_duplicate.get('_id')) != project_id:
                            self.logger.warning(f"Duplicate project name '{project_name}' found for user {user_id}")
                            raise ValueError(f"Project name '{project_name}' already exists. Please choose a different name.")
                    
                    # Delete old GridFS file if it exists
                    old_gridfs_id = existing_project.get('gridfs_file_id')
                    if old_gridfs_id:
                        try:
                            self.file_manager.gridfs.delete(ObjectId(old_gridfs_id))
                            self.logger.info(f"Deleted old GridFS file {old_gridfs_id} for project {project_id}")
                        except Exception as e:
                            self.logger.warning(f"Failed to delete old GridFS file: {e}")
                    
                    # Upload new data to GridFS
                    gridfs_file_id = self.file_manager.gridfs.put(
                        project_bytes,
                        filename=f"project_{project_id}_{current_time.timestamp()}.json",
                        content_type='application/json',
                        user_id=str(user_id),
                        project_name=project_name,
                        upload_date=current_time
                    )
                    
                    # Update metadata document in collection
                    metadata_doc = {
                        'user_id': str(user_id),
                        'project_name': project_name,
                        'project_description': project_data.get('project_description', ''),
                        'product_type': product_type,
                        'gridfs_file_id': gridfs_file_id,
                        'storage_format': 'gridfs',
                        'project_status': 'active',
                        'updated_at': current_time,
                        'file_size': len(project_bytes)
                    }
                    
                    result = self.file_manager.db[self.collection_name].update_one(
                        {'_id': object_id, 'user_id': str(user_id)},
                        {'$set': metadata_doc}
                    )
                    
                    if result.matched_count == 0:
                        raise ValueError("Failed to update project")
                    
                    final_project_id = str(object_id)
                    self.logger.info(f"Updated project {final_project_id} in GridFS for user {user_id}")
                    
                except (ValueError, Exception) as e:
                    self.logger.error(f"Failed to update project {project_id}: {e}")
                    raise
            else:
                # Create new project - first check for duplicate name
                existing_duplicate = self.file_manager.db[self.collection_name].find_one({
                    'user_id': str(user_id),
                    'project_name': project_name,
                    'project_status': 'active'
                })
                
                if existing_duplicate:
                    self.logger.warning(f"Duplicate project name '{project_name}' found for user {user_id}")
                    raise ValueError(f"Project name '{project_name}' already exists. Please choose a different name.")
                
                # Upload data to GridFS first
                gridfs_file_id = self.file_manager.gridfs.put(
                    project_bytes,
                    filename=f"project_new_{current_time.timestamp()}.json",
                    content_type='application/json',
                    user_id=str(user_id),
                    project_name=project_name,
                    upload_date=current_time
                )
                
                # Create metadata document in collection
                metadata_doc = {
                    'user_id': str(user_id),
                    'project_name': project_name,
                    'project_description': project_data.get('project_description', ''),
                    'product_type': product_type,
                    'gridfs_file_id': gridfs_file_id,
                    'storage_format': 'gridfs',
                    'project_status': 'active',
                    'created_at': current_time,
                    'updated_at': current_time,
                    'file_size': len(project_bytes)
                }
                
                result = self.file_manager.db[self.collection_name].insert_one(metadata_doc)
                final_project_id = str(result.inserted_id)
                self.logger.info(f"Created new project {final_project_id} in GridFS for user {user_id}")
            
            return {
                'project_id': final_project_id,
                'project_name': project_name,
                'project_description': project_data.get('project_description', ''),
                'product_type': product_type,
                'pricing': project_data.get('pricing', {}),
                'feedback_entries': project_data.get('feedback_entries', project_data.get('feedback', [])),
                'created_at': metadata_doc.get('created_at', current_time).isoformat(),
                'updated_at': metadata_doc['updated_at'].isoformat(),
                'project_status': metadata_doc['project_status']
            }
            
        except Exception as e:
            self.logger.error(f"Failed to save project for user {user_id}: {e}")
            raise
    
    def get_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get all projects for a user (metadata only, no GridFS loading)
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of project summaries
        """
        try:
            projects_cursor = self.file_manager.db[self.collection_name].find(
                {
                    'user_id': str(user_id),
                    'project_status': 'active'
                },
                no_cursor_timeout=True
            ).sort('updated_at', -1)  # Most recently updated first
            
            project_list = []
            for project in projects_cursor:
                # For GridFS projects, we need to load the file to get counts
                gridfs_file_id = project.get('gridfs_file_id')
                
                if gridfs_file_id:
                    # GridFS project - load data to get counts
                    try:
                        grid_file = self.file_manager.gridfs.get(ObjectId(gridfs_file_id))
                        project_data = json.loads(grid_file.read().decode('utf-8'))
                        
                        identified_instruments = project_data.get('identified_instruments', [])
                        identified_accessories = project_data.get('identified_accessories', [])
                        search_tabs = project_data.get('search_tabs', [])
                        workflow_position = project_data.get('workflow_position', {})
                        user_interactions = project_data.get('user_interactions', {})
                        requirements = project_data.get('initial_requirements', '')
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load GridFS data for project {project['_id']}: {e}")
                        identified_instruments = []
                        identified_accessories = []
                        search_tabs = []
                        workflow_position = {}
                        user_interactions = {}
                        requirements = ''
                else:
                    # Legacy format - shouldn't happen with new implementation
                    identified_instruments = []
                    identified_accessories = []
                    search_tabs = []
                    workflow_position = {}
                    user_interactions = {}
                    requirements = project.get('initial_requirements', '')
                
                # Create requirements preview
                requirements_preview = requirements[:200] + "..." if len(requirements) > 200 else requirements
                
                project_summary = {
                    'id': str(project['_id']),
                    'project_name': project.get('project_name', ''),
                    'project_description': project.get('project_description', ''),
                    'product_type': project.get('product_type', ''),
                    'instruments_count': len(identified_instruments) if isinstance(identified_instruments, list) else 0,
                    'accessories_count': len(identified_accessories) if isinstance(identified_accessories, list) else 0,
                    'search_tabs_count': len(search_tabs) if isinstance(search_tabs, list) else 0,
                    'project_phase': workflow_position.get('project_phase', 'unknown'),
                    'conversations_count': user_interactions.get('conversations_count', 0),
                    'has_analysis': user_interactions.get('has_analysis', False),
                    'schema_version': '2.0',
                    'storage_format': 'gridfs',
                    'project_status': project.get('project_status', 'active'),
                    'created_at': project.get('created_at', datetime.utcnow()).isoformat(),
                    'updated_at': project.get('updated_at', datetime.utcnow()).isoformat(),
                    'requirements_preview': requirements_preview,
                    'file_size': project.get('file_size', 0)
                }
                project_list.append(project_summary)
            
            self.logger.info(f"Retrieved {len(project_list)} projects for user {user_id}")
            return project_list
            
        except Exception as e:
            self.logger.error(f"Failed to get projects for user {user_id}: {e}")
            raise
    
    def get_project_details(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get full project details for loading from GridFS
        
        Args:
            project_id: ID of the project
            user_id: ID of the user (for security)
            
        Returns:
            Complete project data
        """
        try:
            object_id = ObjectId(project_id)
            project_meta = self.file_manager.db[self.collection_name].find_one({
                '_id': object_id,
                'user_id': str(user_id)
            })
            
            if not project_meta:
                raise ValueError("Project not found or access denied")
            
            # Get GridFS file ID
            gridfs_file_id = project_meta.get('gridfs_file_id')
            
            if not gridfs_file_id:
                raise ValueError("Project data not found in GridFS")
            
            # Load complete project data from GridFS
            grid_file = self.file_manager.gridfs.get(ObjectId(gridfs_file_id))
            project_data = json.loads(grid_file.read().decode('utf-8'))
            
            # Add metadata fields
            project_data['id'] = str(project_meta['_id'])
            project_data['created_at'] = project_meta.get('created_at', datetime.utcnow()).isoformat()
            project_data['updated_at'] = project_meta.get('updated_at', datetime.utcnow()).isoformat()
            project_data['project_status'] = project_meta.get('project_status', 'active')
            
            self.logger.info(f"Retrieved project {project_id} from GridFS for user {user_id}")
            return project_data
            
        except Exception as e:
            self.logger.error(f"Failed to get project {project_id} for user {user_id}: {e}")
            raise

    def append_feedback_to_project(self, project_id: str, user_id: str, feedback_entry: Dict[str, Any]) -> bool:
        """
        Append a feedback entry to a project's feedback_entries array in GridFS
        """
        try:
            # Load current project data
            project_data = self.get_project_details(project_id, user_id)
            
            # Append feedback
            if 'feedback_entries' not in project_data:
                project_data['feedback_entries'] = []
            project_data['feedback_entries'].append(feedback_entry)
            
            # Save back (this will update the GridFS file)
            project_data['project_id'] = project_id
            self.save_project(user_id, project_data)
            
            self.logger.info(f"Appended feedback to project {project_id} for user {user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to append feedback to project {project_id}: {e}")
            raise
    
    def delete_project(self, project_id: str, user_id: str) -> bool:
        """
        Permanently delete a project from MongoDB and GridFS
        
        Args:
            project_id: ID of the project
            user_id: ID of the user (for security)
            
        Returns:
            True if successful
        """
        try:
            object_id = ObjectId(project_id)
            
            # Get project metadata to find GridFS file
            project_meta = self.file_manager.db[self.collection_name].find_one({
                '_id': object_id,
                'user_id': str(user_id)
            })
            
            if not project_meta:
                raise ValueError("Project not found or access denied")
            
            # Delete GridFS file first
            gridfs_file_id = project_meta.get('gridfs_file_id')
            if gridfs_file_id:
                try:
                    self.file_manager.gridfs.delete(ObjectId(gridfs_file_id))
                    self.logger.info(f"Deleted GridFS file {gridfs_file_id} for project {project_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete GridFS file: {e}")
            
            # Delete metadata document
            result = self.file_manager.db[self.collection_name].delete_one({
                '_id': object_id,
                'user_id': str(user_id)
            })
            
            if result.deleted_count == 0:
                raise ValueError("Project not found or access denied")
            
            self.logger.info(f"Permanently deleted project {project_id} for user {user_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete project {project_id} for user {user_id}: {e}")
            raise

# Create global instance
mongo_project_manager = MongoProjectManager()