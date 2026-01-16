"""
Configuration management for backend services.

Loads settings from environment variables (.env file) and provides
centralized configuration access for Pinecone, LLM models, ChromaDB,
and general application settings.
"""

from dotenv import load_dotenv
import os
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class PineconeConfig:
    """Centralized Pinecone configuration management"""

    # API Configuration
    API_KEY = os.getenv("PINECONE_API_KEY")

    # Index Configuration
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "agentic-quickstart-test")
    DEFAULT_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "example-namespace")

    # Search Settings
    DEFAULT_TOP_K = int(os.getenv("PINECONE_TOP_K", "10"))
    RERANK_MODEL = os.getenv("PINECONE_RERANK_MODEL", "bge-reranker-v2-m3")

    # Batch Settings (according to CLAUDE.md)
    BATCH_SIZE = int(os.getenv("PINECONE_BATCH_SIZE", "96"))  # For text records
    BATCH_DELAY = float(os.getenv("PINECONE_BATCH_DELAY", "0.1"))  # Delay between batches

    # Retry Settings
    MAX_RETRIES = int(os.getenv("PINECONE_MAX_RETRIES", "5"))
    RETRY_BASE_DELAY = float(os.getenv("PINECONE_RETRY_BASE_DELAY", "1.0"))
    RETRY_MAX_DELAY = float(os.getenv("PINECONE_RETRY_MAX_DELAY", "60.0"))

    # Indexing Settings
    INDEXING_TIMEOUT = int(os.getenv("PINECONE_INDEXING_TIMEOUT", "60"))
    INDEXING_POLL_INTERVAL = int(os.getenv("PINECONE_INDEXING_POLL_INTERVAL", "2"))

    @classmethod
    def validate(cls):
        """
        Validate required configuration settings.

        Raises:
            ValueError: If required configuration is missing
        """
        if not cls.API_KEY:
            raise ValueError(
                "PINECONE_API_KEY is required. "
                "Please set it in your .env file or environment variables."
            )

        # Validate numeric settings
        if cls.BATCH_SIZE < 1 or cls.BATCH_SIZE > 96:
            raise ValueError(
                f"PINECONE_BATCH_SIZE must be between 1 and 96 (got {cls.BATCH_SIZE})"
            )

        if cls.MAX_RETRIES < 1:
            raise ValueError(
                f"PINECONE_MAX_RETRIES must be at least 1 (got {cls.MAX_RETRIES})"
            )

        if cls.DEFAULT_TOP_K < 1:
            raise ValueError(
                f"PINECONE_TOP_K must be at least 1 (got {cls.DEFAULT_TOP_K})"
            )

    @classmethod
    def print_config(cls):
        """Print current configuration (safe - hides API key)"""
        print("=" * 60)
        print("PINECONE CONFIGURATION")
        print("=" * 60)
        print(f"API Key: {'*' * 20}{cls.API_KEY[-4:] if cls.API_KEY else 'NOT SET'}")
        print(f"Index Name: {cls.INDEX_NAME}")
        print(f"Default Namespace: {cls.DEFAULT_NAMESPACE}")
        print(f"Default Top K: {cls.DEFAULT_TOP_K}")
        print(f"Rerank Model: {cls.RERANK_MODEL}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Batch Delay: {cls.BATCH_DELAY}s")
        print(f"Max Retries: {cls.MAX_RETRIES}")
        print(f"Retry Base Delay: {cls.RETRY_BASE_DELAY}s")
        print(f"Retry Max Delay: {cls.RETRY_MAX_DELAY}s")
        print(f"Indexing Timeout: {cls.INDEXING_TIMEOUT}s")
        print(f"Indexing Poll Interval: {cls.INDEXING_POLL_INTERVAL}s")
        print("=" * 60)


class AgenticConfig:
    """Centralized configuration for agentic workflows and LLM operations"""

    # LLM Model Configuration (Multi-tier approach)
    DEFAULT_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gemini-2.5-flash")
    PRO_MODEL = os.getenv("PRO_LLM_MODEL", "gemini-2.5-pro")
    LITE_MODEL = os.getenv("LITE_LLM_MODEL", "gemini-2.5-flash")
    FALLBACK_MODEL = os.getenv("FALLBACK_LLM_MODEL", "gemini-2.5-flash")

    # Google API Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

    # Model Temperature Settings
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
    CREATIVE_TEMPERATURE = float(os.getenv("CREATIVE_TEMPERATURE", "0.7"))

    # Timeout Configuration (in seconds)
    DEFAULT_TIMEOUT = int(os.getenv("DEFAULT_TIMEOUT", "30"))
    LONG_TIMEOUT = int(os.getenv("LONG_TIMEOUT", "120"))
    PARALLEL_ANALYSIS_TIMEOUT = int(os.getenv("PARALLEL_ANALYSIS_TIMEOUT", "300"))

    # Parallel Processing Configuration
    MAX_WORKERS = int(os.getenv("MAX_WORKERS", "5"))
    MAX_VENDORS_TO_ANALYZE = int(os.getenv("MAX_VENDORS_TO_ANALYZE", "5"))

    # ChromaDB Configuration
    CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
    CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
    CHROMADB_TIMEOUT = int(os.getenv("CHROMADB_TIMEOUT", "30"))

    # Retry Configuration
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_DELAY = float(os.getenv("RETRY_DELAY", "1.0"))

    # Workflow Configuration
    MAX_WORKFLOW_RETRIES = int(os.getenv("MAX_WORKFLOW_RETRIES", "2"))
    WORKFLOW_TIMEOUT = int(os.getenv("WORKFLOW_TIMEOUT", "300"))

    # RAG Configuration
    RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
    RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.7"))

    @classmethod
    def validate(cls):
        """
        Validate required configuration settings.

        Raises:
            ValueError: If required configuration is missing or invalid
        """
        if not cls.GOOGLE_API_KEY:
            raise ValueError(
                "GOOGLE_API_KEY is required. "
                "Please set it in your .env file or environment variables."
            )

        # Validate temperature ranges
        if not (0.0 <= cls.DEFAULT_TEMPERATURE <= 2.0):
            raise ValueError(
                f"DEFAULT_TEMPERATURE must be between 0.0 and 2.0 (got {cls.DEFAULT_TEMPERATURE})"
            )

        if not (0.0 <= cls.CREATIVE_TEMPERATURE <= 2.0):
            raise ValueError(
                f"CREATIVE_TEMPERATURE must be between 0.0 and 2.0 (got {cls.CREATIVE_TEMPERATURE})"
            )

        # Validate positive integers
        if cls.MAX_WORKERS < 1:
            raise ValueError(f"MAX_WORKERS must be at least 1 (got {cls.MAX_WORKERS})")

        if cls.DEFAULT_TIMEOUT < 1:
            raise ValueError(f"DEFAULT_TIMEOUT must be at least 1 (got {cls.DEFAULT_TIMEOUT})")

        if cls.RAG_TOP_K < 1:
            raise ValueError(f"RAG_TOP_K must be at least 1 (got {cls.RAG_TOP_K})")

    @classmethod
    def print_config(cls):
        """Print current configuration (safe - hides API key)"""
        logger.info("=" * 60)
        logger.info("AGENTIC CONFIGURATION")
        logger.info("=" * 60)
        logger.info(f"Google API Key: {'*' * 20}{cls.GOOGLE_API_KEY[-4:] if cls.GOOGLE_API_KEY else 'NOT SET'}")
        logger.info(f"Default Model: {cls.DEFAULT_MODEL}")
        logger.info(f"Pro Model: {cls.PRO_MODEL}")
        logger.info(f"Lite Model: {cls.LITE_MODEL}")
        logger.info(f"Fallback Model: {cls.FALLBACK_MODEL}")
        logger.info(f"Default Temperature: {cls.DEFAULT_TEMPERATURE}")
        logger.info(f"Creative Temperature: {cls.CREATIVE_TEMPERATURE}")
        logger.info(f"Default Timeout: {cls.DEFAULT_TIMEOUT}s")
        logger.info(f"Long Timeout: {cls.LONG_TIMEOUT}s")
        logger.info(f"Parallel Analysis Timeout: {cls.PARALLEL_ANALYSIS_TIMEOUT}s")
        logger.info(f"Max Workers: {cls.MAX_WORKERS}")
        logger.info(f"Max Vendors to Analyze: {cls.MAX_VENDORS_TO_ANALYZE}")
        logger.info(f"ChromaDB Host: {cls.CHROMADB_HOST}")
        logger.info(f"ChromaDB Port: {cls.CHROMADB_PORT}")
        logger.info(f"ChromaDB Timeout: {cls.CHROMADB_TIMEOUT}s")
        logger.info(f"RAG Top K: {cls.RAG_TOP_K}")
        logger.info(f"RAG Score Threshold: {cls.RAG_SCORE_THRESHOLD}")
        logger.info("=" * 60)


# Validate configuration on import
try:
    PineconeConfig.validate()
except ValueError as e:
    logger.error(f"Pinecone Configuration Error: {e}")
    logger.error("Please check your .env file and environment variables.")

try:
    AgenticConfig.validate()
except ValueError as e:
    logger.error(f"Agentic Configuration Error: {e}")
    logger.error("Please check your .env file and environment variables.")
