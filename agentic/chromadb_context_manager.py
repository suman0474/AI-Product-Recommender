"""ChromaDB-specific context manager for connection pooling."""
from agentic.context_managers import ConnectionPoolManager
import chromadb
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)

class ChromaDBConnectionManager(ConnectionPoolManager):
    """
    Manage ChromaDB HTTP client connections with pooling.
    Ensures safe connection lifecycle and reusability.
    """
    
    def __init__(self, host: str = "localhost", port: int = 8000, max_connections: int = 5):
        super().__init__(f"chromadb_{host}:{port}", max_connections)
        self.host = host
        self.port = port
        self._http_client: Optional[Any] = None
    
    def __enter__(self):
        super().__enter__()
        # In a real pooling scenario we might get a client from a pool
        # Here we initialize the client when acquiring stats
        try:
            self._http_client = chromadb.HttpClient(host=self.host, port=self.port)
            logger.info(f"Connected to ChromaDB at {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Allow cleanup
        self._http_client = None
        return super().__exit__(exc_type, exc_val, exc_tb)
    
    def get_collection(self, name: str):
        """Get a collection with connection tracking."""
        if not self._http_client:
             raise RuntimeError("ChromaDB client not initialized. Use 'with' block.")
             
        conn_id = f"collection_{name}"
        self.acquire_connection(conn_id)
        
        try:
            return self._http_client.get_collection(name)
        except Exception as e:
            self.release_connection(conn_id)
            raise e

    def get_or_create_collection(self, name: str):
        """Get or create collection with tracking."""
        if not self._http_client:
             raise RuntimeError("ChromaDB client not initialized. Use 'with' block.")

        conn_id = f"collection_{name}"
        self.acquire_connection(conn_id)
        
        try:
            return self._http_client.get_or_create_collection(name)
        except Exception as e:
            self.release_connection(conn_id)
            raise e
