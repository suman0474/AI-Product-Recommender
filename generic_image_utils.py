"""
Generic Product Type Image Utilities
Handles fetching, caching, and retrieving generic product type images
using ONLY Azure Blob Storage with LLM-based generation via Gemini Imagen 4.0.

ARCHITECTURE (Azure-Only):
1. Check Azure Blob Storage for cached image
2. If not found, check if generation is in-progress (request deduplication)
3. Generate via Gemini Imagen 4.0 LLM (NO rate limiting between requests)
4. Store result in Azure Blob Storage ONLY
5. Return URL/metadata to caller

RATE LIMIT HANDLING: Exponential backoff for 429 errors.
REQUEST DEDUPLICATION: Only one LLM call per product type for concurrent requests.
STORAGE: Azure Blob Storage (primary and only storage).
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

# =============================================================================
# REQUEST DEDUPLICATION SYSTEM & THROTTLING
# =============================================================================
# Prevents duplicate concurrent LLM calls for the same product type.
# Only the first request generates; subsequent concurrent requests wait for the result.

_generation_lock = threading.Lock()  # Lock for single execution
_pending_requests: Dict[str, threading.Event] = {}  # Track pending requests
_pending_results: Dict[str, Any] = {}  # Store results for pending requests
_pending_lock = threading.Lock()

# FIX #3: Rate limiting throttling - prevents thundering herd
_last_request_time = 0  # Timestamp of last LLM request
_throttle_lock = threading.Lock()
_MIN_REQUEST_INTERVAL = 2.0  # Minimum 2 seconds between requests to avoid rate limits


def _apply_request_throttle():
    """
    FIX #3: Apply request throttling with exponential backoff and jitter.

    Ensures minimum delay between LLM requests to avoid rate limiting.
    Uses jitter to prevent thundering herd when multiple requests arrive simultaneously.

    Returns:
        float: Actual wait time applied (seconds)
    """
    import random

    global _last_request_time
    current_time = time.time()

    with _throttle_lock:
        time_since_last = current_time - _last_request_time

        if time_since_last < _MIN_REQUEST_INTERVAL:
            # Add jitter (±25% of wait time) to prevent synchronization
            wait_time = _MIN_REQUEST_INTERVAL - time_since_last
            jitter = random.uniform(0, 0.25 * wait_time)
            actual_wait = wait_time + jitter

            logger.info(
                f"[FIX3] Request throttling: waiting {actual_wait:.2f}s "
                f"(jitter: {jitter:.2f}s) to maintain {_MIN_REQUEST_INTERVAL}s interval"
            )
            time.sleep(actual_wait)
            _last_request_time = time.time()
            return actual_wait
        else:
            _last_request_time = current_time
            return 0.0


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
        logger.info(f"[LLM_IMAGE_GEN] Generating image for '{product_type}' using Gemini Imagen model...")

        # FIX #3: Apply request throttling to prevent rate limiting
        throttle_wait = _apply_request_throttle()
        if throttle_wait > 0:
            logger.info(f"[FIX3] Applied throttling: {throttle_wait:.2f}s")

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
                # FIX #3: Extract retry delay from error message if available
                retry_delay = 60  # Default to 60 seconds
                if 'retry in' in error_str.lower():
                    try:
                        import re
                        match = re.search(r'retry in (\d+(?:\.\d+)?)', error_str.lower())
                        if match:
                            retry_delay = float(match.group(1))
                    except:
                        pass

                # FIX #3: More aggressive exponential backoff with jitter
                # Exponential: 60s -> 120s -> 240s, capped at 5 minutes
                base_wait = retry_delay * (2 ** retry_count)
                jitter = time.time() % 1.0  # Add 0-1s jitter
                wait_time = min(base_wait + jitter, 300)  # Cap at 5 minutes

                logger.warning(
                    f"[FIX3] Rate limit hit for '{product_type}'. "
                    f"Retry {retry_count + 1}/{max_retries} after {wait_time:.1f}s "
                    f"(base: {base_wait:.1f}s, jitter: {jitter:.1f}s)..."
                )
                time.sleep(wait_time)

                # Retry the request
                return _generate_image_with_llm(product_type, retry_count + 1, max_retries)
            else:
                logger.error(f"[LLM_IMAGE_GEN] Max retries ({max_retries}) exceeded for '{product_type}' due to rate limiting")
                return None
        
        logger.error(f"[LLM_IMAGE_GEN] Failed to generate image with Gemini: {e}")
        logger.exception(e)
        return None


# --- Main Utilities ---

def get_generic_image_from_azure(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve cached generic product type image from Azure Blob Storage

    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")

    Returns:
        Dict containing image metadata or None if not found
    """
    try:
        from azure_blob_config import Collections, azure_blob_manager
        from azure.core.exceptions import ResourceNotFoundError
        import json

        # Normalize product type for Azure path
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

        # Construct metadata path (relative to base path)
        metadata_path = f"{Collections.GENERIC_IMAGES}/{normalized_type}.json"

        # Try to get metadata from Azure Blob
        blob_client = azure_blob_manager.get_blob_client(metadata_path)
        metadata_bytes = blob_client.download_blob().readall()

        # Parse metadata
        metadata = json.loads(metadata_bytes.decode('utf-8'))
        logger.info(f"[AZURE_CHECK] ✓ Found cached generic image in Azure Blob for: {product_type}")

        return {
            'azure_blob_path': metadata.get('image_blob_path'),
            'product_type': metadata.get('product_type'),
            'source': metadata.get('source'),
            'content_type': metadata.get('content_type', 'image/png'),
            'file_size': metadata.get('file_size', 0),
            'generation_method': metadata.get('generation_method', 'llm'),
            'cached': True,
            'storage_location': 'azure_blob'
        }

    except ResourceNotFoundError:
        logger.debug(f"[AZURE_CHECK] No cached generic image in Azure Blob for: {product_type}")
        return None
    except Exception as e:
        logger.debug(f"[AZURE_CHECK] Failed to retrieve generic image from Azure Blob: {e}")
        return None


def cache_generic_image_to_azure(product_type: str, image_data: Dict[str, Any]) -> bool:
    """
    Cache LLM-generated image to Azure Blob Storage

    Args:
        product_type: Product type name
        image_data: Generated image data containing image_bytes, content_type, etc.

    Returns:
        bool: True if successfully cached
    """
    try:
        from azure_blob_config import Collections, azure_blob_manager
        from azure.storage.blob import ContentSettings
        import json

        logger.info(f"[CACHE_AZURE] Caching generic image to Azure Blob for: {product_type}")

        # Get image bytes
        image_bytes = image_data.get('image_bytes')
        if not image_bytes:
            logger.warning(f"[CACHE_AZURE] No image bytes provided for caching: {product_type}")
            return False

        content_type = image_data.get('content_type', 'image/png')
        file_size = image_data.get('file_size', len(image_bytes))

        # Normalize product type for path
        normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

        # Determine file extension
        file_extension = content_type.split('/')[-1] if '/' in content_type else 'png'

        # Upload image blob
        image_blob_path = f"{Collections.GENERIC_IMAGES}/{normalized_type}.{file_extension}"
        image_blob_client = azure_blob_manager.get_blob_client(image_blob_path)
        image_blob_client.upload_blob(
            image_bytes,
            overwrite=True,
            content_settings=ContentSettings(content_type=content_type),
            metadata={
                'product_type': product_type,
                'normalized_type': normalized_type,
                'generation_method': 'llm'
            }
        )

        logger.info(f"[CACHE_AZURE] ✓ Stored image in Azure Blob: {image_blob_path}")

        # Upload metadata JSON
        metadata_doc = {
            'product_type': product_type,
            'product_type_normalized': normalized_type,
            'image_blob_path': image_blob_path,
            'source': image_data.get('source', 'gemini_imagen'),
            'content_type': content_type,
            'file_size': file_size,
            'generation_method': 'llm',
            'generation_prompt': image_data.get('prompt', ''),
            'created_at': datetime.utcnow().isoformat()
        }

        metadata_blob_path = f"{Collections.GENERIC_IMAGES}/{normalized_type}.json"
        metadata_json = json.dumps(metadata_doc, indent=2)

        metadata_blob_client = azure_blob_manager.get_blob_client(metadata_blob_path)
        metadata_blob_client.upload_blob(
            metadata_json,
            overwrite=True,
            content_settings=ContentSettings(content_type='application/json'),
            metadata={
                'product_type': product_type,
                'normalized_type': normalized_type,
                'metadata_file': 'true'
            }
        )

        logger.info(f"[CACHE_AZURE] ✓ Stored metadata in Azure Blob: {metadata_blob_path}")
        return True

    except Exception as e:
        logger.error(f"[CACHE_AZURE] Failed to cache generic image to Azure Blob: {e}")
        logger.exception(e)
        return False




def fetch_generic_product_image(product_type: str) -> Optional[Dict[str, Any]]:
    """
    Fetch generic product type image with Azure Blob caching and LLM generation.

    FLOW (Azure-Only):
    1. Check Azure Blob Storage cache first
    2. Check if generation is already in-progress for this product type (deduplication)
    3. If not, generate image using Gemini Imagen 4.0 LLM (NO rate limiting)
    4. Cache the LLM-generated image to Azure Blob Storage ONLY

    Args:
        product_type: Product type name (e.g., "Pressure Transmitter")

    Returns:
        Dict containing image URL and metadata, or None if generation failed
    """
    logger.info(f"[FETCH] Fetching generic image for product type: {product_type}")

    # Normalize product type for deduplication
    normalized_type = product_type.strip().lower().replace(" ", "").replace("_", "").replace("-", "")

    # Step 1: Check Azure Blob Storage for cached image
    azure_image = get_generic_image_from_azure(product_type)
    if azure_image:
        logger.info(f"[FETCH] ✓ Using cached generic image from Azure Blob for '{product_type}'")
        backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

        return {
            'url': backend_url,
            'product_type': product_type,
            'source': azure_image.get('source', 'gemini_imagen'),
            'cached': True,
            'generation_method': azure_image.get('generation_method', 'llm')
        }

    # Step 2: Check if generation is already in-progress for this product type
    with _pending_lock:
        if normalized_type in _pending_requests:
            # Another request is already generating this image - wait for it
            logger.info(f"[FETCH] Generation already in-progress for '{product_type}', waiting...")
            event = _pending_requests[normalized_type]
        else:
            # We are the first - mark as in-progress
            event = threading.Event()
            _pending_requests[normalized_type] = event
            _pending_results[normalized_type] = None

    # If we're waiting for another request
    if event.is_set() or (normalized_type in _pending_requests and _pending_requests[normalized_type] != event):
        # Wait for the event with timeout
        wait_result = event.wait(timeout=120)  # Wait up to 2 minutes

        if wait_result:
            # Check cache again - the other request should have populated it
            azure_image = get_generic_image_from_azure(product_type)
            if azure_image:
                logger.info(f"[FETCH] ✓ Got result from parallel request for '{product_type}'")
                backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

                return {
                    'url': backend_url,
                    'product_type': product_type,
                    'source': azure_image.get('source', 'gemini_imagen'),
                    'cached': True,
                    'generation_method': azure_image.get('generation_method', 'llm')
                }

        logger.warning(f"[FETCH] Parallel request wait timed out for '{product_type}'")
        return None

    # Step 3: We are the primary generator - generate image using LLM (NO rate limiting)
    logger.info(f"[FETCH] Cache miss for '{product_type}', generating image with Gemini Imagen 4.0...")

    try:
        generated_image_data = _generate_image_with_llm(product_type)

        if generated_image_data:
            # Step 4: Cache to Azure Blob Storage ONLY
            azure_cache_success = cache_generic_image_to_azure(product_type, generated_image_data)

            if azure_cache_success:
                logger.info(f"[FETCH] ✓ Successfully generated and cached LLM image for '{product_type}'")

                # Retrieve the cached image to get the URL
                azure_image = get_generic_image_from_azure(product_type)
                if azure_image:
                    backend_url = f"/api/images/{azure_image.get('azure_blob_path', '')}"

                    # Notify waiting requests
                    with _pending_lock:
                        if normalized_type in _pending_requests:
                            _pending_requests[normalized_type].set()
                            del _pending_requests[normalized_type]
                        if normalized_type in _pending_results:
                            del _pending_results[normalized_type]

                    return {
                        'url': backend_url,
                        'product_type': product_type,
                        'source': 'gemini_imagen',
                        'cached': False,  # Newly generated
                        'generation_method': 'llm'
                    }

            logger.warning(f"[FETCH] LLM image generated but caching failed for '{product_type}'")
        else:
            logger.error(f"[FETCH] LLM image generation failed for '{product_type}'")

    finally:
        # Clean up pending request tracking
        with _pending_lock:
            if normalized_type in _pending_requests:
                _pending_requests[normalized_type].set()  # Signal completion (even on failure)
                del _pending_requests[normalized_type]
            if normalized_type in _pending_results:
                del _pending_results[normalized_type]

    # Complete failure - no image available
    logger.error(f"[FETCH] Failed to retrieve/generate image for '{product_type}'")
    return None

