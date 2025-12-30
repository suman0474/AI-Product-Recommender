"""
Generic Product Type Image Utilities
Handles fetching, caching, and retrieving generic product type images
using MongoDB cache and LLM-based image generation.
"""

import logging
import os
import io
import time
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from PIL import Image
import google.genai as genai
from google.genai import types

logger = logging.getLogger(__name__)

# Align environment keys with main.py
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY1") or os.getenv("GOOGLE_API_KEY")

# Configure Gemini client for image generation
_gemini_client = None
if GOOGLE_API_KEY:
    try:
        _gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
        logger.info("[INIT] Gemini client initialized for image generation")
    except Exception as e:
        logger.warning(f"[INIT] Failed to initialize Gemini client: {e}")

# Rate limiting for Gemini Imagen API (10 requests per minute)
_rate_limit_lock = threading.Lock()
_last_request_time = 0
_request_count = 0
_request_window_start = 0
MAX_REQUESTS_PER_MINUTE = 8  # Set to 8 to be safe (limit is 10)
REQUEST_WINDOW = 60  # 60 seconds

def _wait_for_rate_limit():
    """
    Enforce rate limiting for Gemini Imagen API
    Waits if we've exceeded the request quota
    """
    global _request_count, _request_window_start, _last_request_time
    
    with _rate_limit_lock:
        current_time = time.time()
        
        # Reset counter if we're in a new window
        if current_time - _request_window_start >= REQUEST_WINDOW:
            _request_count = 0
            _request_window_start = current_time
            logger.info(f"[RATE_LIMIT] New request window started")
        
        # If we've hit the limit, wait until the window resets
        if _request_count >= MAX_REQUESTS_PER_MINUTE:
            wait_time = REQUEST_WINDOW - (current_time - _request_window_start)
            if wait_time > 0:
                logger.warning(f"[RATE_LIMIT] Quota reached ({_request_count}/{MAX_REQUESTS_PER_MINUTE}). Waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
                # Reset after waiting
                _request_count = 0
                _request_window_start = time.time()
        
        # Increment counter
        _request_count += 1
        _last_request_time = time.time()
        logger.info(f"[RATE_LIMIT] Request {_request_count}/{MAX_REQUESTS_PER_MINUTE} in current window")

def _generate_image_with_llm(product_type: str, retry_count: int = 0, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Generate a generic product image using Gemini's Imagen model
    
    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")
        retry_count: Current retry attempt (for internal use)
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dict containing generated image data or None if generation failed
    """
    global _gemini_client
    
    if not _gemini_client:
        logger.error("[LLM_IMAGE_GEN] Gemini client not initialized. Check GOOGLE_API_KEY.")
        return None
    
    try:
        # Enforce rate limiting before making request
        _wait_for_rate_limit()
        
        logger.info(f"[LLM_IMAGE_GEN] Generating image for '{product_type}' using Gemini Imagen model...")
        
        # Construct the prompt for image generation
        # Ensure no text and no background color as per user requirement
        prompt = f"A professional 3D render of a {product_type}, studio lighting, high resolution, isolated on a transparent background, no shadow, clean edges."
        
        logger.info(f"[LLM_IMAGE_GEN] Prompt: {prompt}")
        
        # Use Gemini's Imagen 4.0 model for image generation
        response = _gemini_client.models.generate_images(
            model='imagen-4.0-generate-001',
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                # aspect_ratio can be "1:1", "3:4", "4:3", "9:16", "16:9"
                # For generic product images, square is best
            )
        )
        
        if not response.generated_images:
            logger.warning(f"[LLM_IMAGE_GEN] No images generated for '{product_type}'")
            return None
        
        # Get the first generated image
        generated_image = response.generated_images[0]
        
        # The image bytes are directly available from the response
        # The google.genai library provides the image as bytes
        image_bytes = generated_image.image.image_bytes
        
        logger.info(f"[LLM_IMAGE_GEN] ✓ Successfully generated image for '{product_type}' (size: {len(image_bytes)} bytes)")
        
        return {
            'image_bytes': image_bytes,
            'content_type': 'image/png',
            'file_size': len(image_bytes),
            'source': 'gemini_imagen',
            'prompt': prompt
        }
        
    except Exception as e:
        error_str = str(e)
        
        # Check if it's a rate limit error (429)
        if '429' in error_str or 'RESOURCE_EXHAUSTED' in error_str or 'quota' in error_str.lower():
            if retry_count < max_retries:
                # Extract retry delay from error message if available
                retry_delay = 60  # Default to 60 seconds
                if 'retry in' in error_str.lower():
                    try:
                        import re
                        match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str.lower())
                        if match:
                            retry_delay = float(match.group(1))
                    except:
                        pass
                
                # Exponential backoff: wait longer with each retry
                wait_time = min(retry_delay * (2 ** retry_count), 120)  # Cap at 2 minutes
                logger.warning(f"[LLM_IMAGE_GEN] Rate limit hit for '{product_type}'. Retry {retry_count + 1}/{max_retries} after {wait_time:.1f}s...")
                time.sleep(wait_time)
                
                # Retry the request
                return _generate_image_with_llm(product_type, retry_count + 1, max_retries)
            else:
                logger.error(f"[LLM_IMAGE_GEN] Max retries ({max_retries}) exceeded for '{product_type}' due to rate limiting")
                return None
        
        logger.error(f"[LLM_IMAGE_GEN] Failed to generate image with Gemini: {e}")
        logger.exception(e)
        return None

def _cache_generated_image(product_type: str, image_data: Dict[str, Any]) -> bool:
    """
    Cache LLM-generated image directly to MongoDB GridFS
    
    Args:
        product_type: Product type name
        image_data: Generated image data containing image_bytes, content_type, etc.
        
    Returns:
        bool: True if successfully cached
    """
    try:
        from azure_blob_config import get_azure_blob_connection as get_mongodb_connection

        logger.info(f"[CACHE_LLM] Caching LLM-generated image for: {product_type}")

        conn = get_mongodb_connection()
        generic_images_collection = conn['collections']['generic_images']
        gridfs = conn['gridfs']
        
        # Get image bytes
        image_bytes = image_data.get('image_bytes')
        if not image_bytes:
            logger.warning(f"[CACHE_LLM] No image bytes provided for caching: {product_type}")
            return False
        
        content_type = image_data.get('content_type', 'image/png')
        file_size = image_data.get('file_size', len(image_bytes))
        
        # Normalize product type for indexing
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        
        # Generate filename
        file_extension = content_type.split('/')[-1] if '/' in content_type else 'png'
        filename = f"generic_{normalized_type}_llm.{file_extension}"
        
        # Store image in GridFS
        gridfs_metadata = {
            'product_type': product_type,
            'source': image_data.get('source', 'gemini_imagen'),
            'image_type': 'generic_llm_generated',
            'generation_prompt': image_data.get('prompt', '')
        }
        
        gridfs_file_id = gridfs.put(
            image_bytes,
            filename=filename,
            content_type=content_type,
            **gridfs_metadata
        )
        
        logger.info(f"[CACHE_LLM] ✓ Stored LLM-generated image in GridFS: {filename} (ID: {gridfs_file_id}, size: {file_size} bytes)")
        
        # Store metadata in generic_images collection
        image_doc = {
            'product_type': product_type,
            'product_type_normalized': normalized_type,
            'gridfs_file_id': gridfs_file_id,
            'source': image_data.get('source', 'gemini_imagen'),
            'content_type': content_type,
            'file_size': file_size,
            'filename': filename,
            'generation_method': 'llm',
            'generation_prompt': image_data.get('prompt', ''),
            'created_at': datetime.utcnow()
        }
        
        # Use upsert to avoid duplicates
        result = generic_images_collection.update_one(
            {'product_type_normalized': normalized_type},
            {'$set': image_doc},
            upsert=True
        )
        
        if result.upserted_id:
            logger.info(f"[CACHE_LLM] ✓ NEW document inserted in generic_images collection (ID: {result.upserted_id})")
            return True
        elif result.modified_count > 0:
            logger.info(f"[CACHE_LLM] ✓ UPDATED existing document in generic_images collection")
            return True
        else:
            logger.warning(f"[CACHE_LLM] ⚠ No changes made to generic_images collection")
            return True
        
    except Exception as e:
        logger.error(f"[CACHE_LLM] Failed to cache LLM-generated image: {e}")
        return False

# --- Main Utilities ---

def get_cached_generic_image(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached generic product type image from MongoDB GridFS
    
    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")
        
    Returns:
        Dict containing image metadata with gridfs_file_id or None if not found
    """
    try:
        from azure_blob_config import get_azure_blob_connection as get_mongodb_connection

        conn = get_mongodb_connection()
        generic_images_collection = conn['collections']['generic_images']
        
        # Normalize product type for search
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")
        
        query = {
            'product_type_normalized': normalized_type
        }
        
        cached_image = generic_images_collection.find_one(query)
        
        if cached_image and cached_image.get('gridfs_file_id'):
            logger.info(f"Found cached generic image in GridFS for product type: {product_type}")
            # Return image metadata with GridFS file reference
            return {
                'gridfs_file_id': str(cached_image.get('gridfs_file_id')),
                'product_type': cached_image.get('product_type'),
                'source': cached_image.get('source'),
                'content_type': cached_image.get('content_type', 'image/jpeg'),
                'file_size': cached_image.get('file_size', 0),
                'original_url': cached_image.get('original_url'),
                'generation_method': cached_image.get('generation_method', 'llm'),
                'cached': True
            }
        
        return None
        
    except Exception as e:
        logger.error(f"Failed to retrieve cached generic image: {e}")
        return None


def fetch_generic_product_image(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Fetch generic product type image with MongoDB caching and LLM generation
    
    Flow:
    1. Check MongoDB cache first
    2. If not found, generate image using Gemini LLM (can run in parallel)
    3. Cache the LLM-generated image in MongoDB
    
    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")
        
    Returns:
        Dict containing image URL and metadata, or None if generation failed
    
    Note: This function is designed to be called in parallel from multiple requests.
    The LLM generation happens independently for each request, allowing parallel processing.
    """
    logger.info(f"[FETCH] Fetching generic image for product type: {product_type}")
    
    # Step 1: Check MongoDB cache first
    cached_image = get_cached_generic_image(product_type)
    if cached_image:
        logger.info(f"[FETCH] ✓ Using cached generic image from GridFS for '{product_type}'")
        gridfs_file_id = cached_image.get('gridfs_file_id')
        backend_url = f"/api/images/{gridfs_file_id}"
        
        return {
            'url': backend_url,
            'product_type': product_type,
            'source': cached_image.get('source', 'gemini_imagen'),
            'cached': True,
            'gridfs_file_id': gridfs_file_id,
            'generation_method': cached_image.get('generation_method', 'llm')
        }
    
    # Step 2: Cache miss - generate image using LLM
    # NOTE: This can run in parallel when multiple requests come in simultaneously
    logger.info(f"[FETCH] Cache miss for '{product_type}', generating image with LLM...")
    
    generated_image_data = _generate_image_with_llm(product_type)
    
    if generated_image_data:
        # Cache the LLM-generated image
        cache_success = _cache_generated_image(product_type, generated_image_data)
        
        if cache_success:
            # Return the cached LLM-generated image
            cached_image = get_cached_generic_image(product_type)
            if cached_image:
                gridfs_file_id = cached_image.get('gridfs_file_id')
                backend_url = f"/api/images/{gridfs_file_id}"
                
                logger.info(f"[FETCH] ✓ Successfully generated and cached LLM image for '{product_type}'")
                
                return {
                    'url': backend_url,
                    'product_type': product_type,
                    'source': 'gemini_imagen',
                    'cached': True,
                    'gridfs_file_id': gridfs_file_id,
                    'generation_method': 'llm'
                }
        
        logger.warning(f"[FETCH] LLM image generated but caching failed for '{product_type}'")
    else:
        logger.error(f"[FETCH] LLM image generation failed for '{product_type}'")
    
    # Complete failure - no image available
    logger.error(f"[FETCH] Failed to retrieve/generate image for '{product_type}'")
    return None
