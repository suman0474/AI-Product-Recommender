"""
Compression Manager
Handles compression of embeddings and state for network efficiency
Supports float16 (2x compression) and int8 (4x compression) quantization
"""
import logging
import numpy as np
from typing import List, Union, Optional, Tuple

logger = logging.getLogger(__name__)


class CompressionManager:
    """Manages compression of embeddings and state for network efficiency."""

    @staticmethod
    def compress_embeddings(
        embeddings: Union[List[float], List[List[float]]],
        precision: str = "float16"
    ) -> bytes:
        """
        Compress embedding vectors using quantization.

        Reduces embedding size by 50% (float16) or 75% (int8) while maintaining
        accuracy suitable for semantic search.

        Args:
            embeddings: Single or batch of embeddings
            precision: "float16" (2x compression) or "int8" (4x compression)

        Returns:
            Compressed bytes representation
        """
        arr = np.array(embeddings, dtype=np.float32)

        if precision == "float16":
            # 2x compression: float32 -> float16
            # Maintains good precision for embeddings
            compressed = arr.astype(np.float16)
            logger.debug(
                f"[COMPRESSION] Compressed embeddings to float16 "
                f"({arr.nbytes} -> {compressed.nbytes} bytes)"
            )
        elif precision == "int8":
            # 4x compression: normalize to -128..127 range
            # Some precision loss but acceptable for embeddings
            min_val = arr.min()
            max_val = arr.max()

            # Normalize to 0..1 range, then to -128..127
            if max_val > min_val:
                normalized = (arr - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(arr)

            compressed = (normalized * 255 - 128).astype(np.int8)
            logger.debug(
                f"[COMPRESSION] Compressed embeddings to int8 "
                f"({arr.nbytes} -> {compressed.nbytes} bytes)"
            )
        else:
            raise ValueError(f"Unknown precision: {precision}")

        return compressed.tobytes()

    @staticmethod
    def decompress_embeddings(
        compressed: bytes,
        precision: str = "float16",
        original_shape: Optional[Tuple[int, ...]] = None
    ) -> List[float]:
        """
        Decompress embedding vectors.

        Args:
            compressed: Compressed bytes
            precision: Compression type used ("float16" or "int8")
            original_shape: Original shape for reshape (e.g., (10, 768))

        Returns:
            Decompressed embeddings as list of floats
        """
        if precision == "float16":
            arr = np.frombuffer(compressed, dtype=np.float16)
            decompressed = arr.astype(np.float32)
        elif precision == "int8":
            arr = np.frombuffer(compressed, dtype=np.int8)
            # Denormalize from -128..127 to 0..1 to original range
            decompressed = (arr.astype(np.float32) + 128) / 255.0
        else:
            raise ValueError(f"Unknown precision: {precision}")

        if original_shape:
            decompressed = decompressed.reshape(original_shape)

        return decompressed.astype(np.float32).tolist()

    @staticmethod
    def get_compression_ratio(
        original_size_bytes: int,
        precision: str = "float16"
    ) -> float:
        """
        Get compression ratio for given precision.

        Args:
            original_size_bytes: Size of uncompressed data
            precision: Compression type

        Returns:
            Ratio of compressed/original (e.g., 0.5 for 50% compression)
        """
        if precision == "float16":
            return 0.5  # 2x compression
        elif precision == "int8":
            return 0.25  # 4x compression
        else:
            return 1.0

    @staticmethod
    def compress_json_response(json_obj: dict) -> dict:
        """
        Compress JSON response by removing verbose fields.

        Args:
            json_obj: JSON object to compress

        Returns:
            Compressed JSON object
        """
        # Remove verbose fields that increase response size
        fields_to_remove = ['_internal', 'debug_info', 'metadata', 'verbose_log']

        for field in fields_to_remove:
            json_obj.pop(field, None)

        logger.debug("[COMPRESSION] Removed verbose fields from JSON response")
        return json_obj


# Compression profile presets
COMPRESSION_PROFILES = {
    "aggressive": {
        "precision": "int8",
        "description": "Maximum compression (75%), some precision loss"
    },
    "balanced": {
        "precision": "float16",
        "description": "Good balance (50% compression), minimal precision loss"
    },
    "lossless": {
        "precision": "float32",
        "description": "No compression, full precision"
    }
}


def get_compression_profile(profile: str) -> dict:
    """
    Get compression profile settings.

    Args:
        profile: "aggressive", "balanced", or "lossless"

    Returns:
        Profile settings dictionary
    """
    if profile not in COMPRESSION_PROFILES:
        logger.warning(f"Unknown profile: {profile}, using 'balanced'")
        profile = "balanced"

    return COMPRESSION_PROFILES[profile]
