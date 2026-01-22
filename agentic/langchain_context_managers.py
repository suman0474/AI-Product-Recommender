"""
Framework-specific context managers for LangChain operations.
These are built on top of Python's context manager protocol but are optimized
for LangChain-specific workflows like data loading, embedding, and LLM chaining.
"""

import logging
import time
from typing import Any, Optional, List, Dict
from contextlib import contextmanager
from threading import RLock

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


class LangChainDataSourceLoader:
    """
    Framework-specific context manager for LangChain data source operations.
    Manages: document loading, chunking, embedding, and cleanup.
    """

    def __init__(self, source_type: str, source_path: str):
        """
        Initialize a LangChain data source loader.

        Args:
            source_type: Type of source (file, directory, database, api)
            source_path: Path or identifier for the data source
        """
        self.source_type = source_type
        self.source_path = source_path
        self.documents: List[Document] = []
        self.document_count = 0
        self._lock = RLock()
        self._is_active = False
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Initialize LangChain data source"""
        with self._lock:
            self._is_active = True
            self.start_time = time.time()
            logger.info(
                f"Initializing LangChain {self.source_type} data source: "
                f"{self.source_path}"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup LangChain data source resources"""
        with self._lock:
            self._is_active = False
            duration = time.time() - self.start_time if self.start_time else 0

            if exc_type:
                logger.error(
                    f"Error in LangChain {self.source_type} loader: {exc_val}. "
                    f"Loaded {self.document_count} documents before failure."
                )
            else:
                logger.info(
                    f"LangChain {self.source_type} data source closed successfully. "
                    f"Processed {self.document_count} documents in {duration:.2f}s"
                )

            # Clear documents from memory if large
            self.documents.clear()

        return False  # Don't suppress exceptions

    def add_document(self, doc: Document):
        """Add a document to the loader"""
        with self._lock:
            if not self._is_active:
                raise RuntimeError("Data source loader is not active")
            self.documents.append(doc)
            self.document_count += 1

    def add_documents(self, docs: List[Document]):
        """Add multiple documents"""
        with self._lock:
            if not self._is_active:
                raise RuntimeError("Data source loader is not active")
            self.documents.extend(docs)
            self.document_count += len(docs)

    def get_documents(self) -> List[Document]:
        """Get all loaded documents"""
        with self._lock:
            return self.documents.copy()


class LangChainEmbeddingManager:
    """
    Framework-specific context manager for LangChain embedding operations.
    Manages: embedding model initialization, batch processing, and vector cleanup.
    """

    def __init__(self, model_name: str = "models/embedding-001"):
        """
        Initialize LangChain embedding manager.

        Args:
            model_name: Google Generative AI embedding model
        """
        self.model_name = model_name
        self.embedding_model: Optional[GoogleGenerativeAIEmbeddings] = None
        self._lock = RLock()
        self._is_active = False
        self.embedded_documents = 0
        self.batch_size = 32

    def __enter__(self):
        """Initialize embedding model"""
        with self._lock:
            try:
                self.embedding_model = GoogleGenerativeAIEmbeddings(
                    model=self.model_name
                )
                self._is_active = True
                logger.info(f"Initialized LangChain embedding model: {self.model_name}")
                return self
            except Exception as e:
                logger.error(f"Failed to initialize embedding model: {e}")
                self._is_active = False
                raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup embedding resources"""
        with self._lock:
            self._is_active = False

            if exc_type:
                logger.error(
                    f"Error during embedding operation: {exc_val}. "
                    f"Embedded {self.embedded_documents} documents before failure."
                )
            else:
                logger.info(
                    f"LangChain embedding manager closed. "
                    f"Total documents embedded: {self.embedded_documents}"
                )

            # Clear model from memory
            self.embedding_model = None

        return False

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Embed documents using LangChain embedding model.

        Args:
            documents: List of documents to embed

        Returns:
            List of embedding vectors
        """
        with self._lock:
            if not self._is_active or not self.embedding_model:
                raise RuntimeError("Embedding manager is not active")

            try:
                texts = [doc.page_content for doc in documents]
                embeddings = self.embedding_model.embed_documents(texts)
                self.embedded_documents += len(documents)

                logger.info(
                    f"Embedded {len(documents)} documents using {self.model_name}"
                )
                return embeddings

            except Exception as e:
                logger.error(f"Error embedding documents: {e}")
                raise


class LangChainWorkflowExecutor:
    """
    Framework-specific context manager for LangChain workflow execution.
    Manages: workflow initialization, node execution, state management, and cleanup.
    """

    def __init__(self, workflow_name: str):
        """
        Initialize LangChain workflow executor.

        Args:
            workflow_name: Name of the workflow being executed
        """
        self.workflow_name = workflow_name
        self.workflow_state: Dict[str, Any] = {}
        self.execution_stats = {
            'nodes_executed': 0,
            'start_time': None,
            'end_time': None,
            'errors': []
        }
        self._lock = RLock()
        self._is_active = False

    def __enter__(self):
        """Initialize workflow execution context"""
        with self._lock:
            self._is_active = True
            self.execution_stats['start_time'] = time.time()
            logger.info(f"Starting LangChain workflow execution: {self.workflow_name}")
            return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup workflow execution resources"""
        with self._lock:
            self._is_active = False
            self.execution_stats['end_time'] = time.time()
            duration = (
                self.execution_stats['end_time'] - self.execution_stats['start_time']
            )

            if exc_type:
                self.execution_stats['errors'].append(str(exc_val))
                logger.error(
                    f"Workflow {self.workflow_name} failed after "
                    f"{self.execution_stats['nodes_executed']} nodes: {exc_val}"
                )
            else:
                logger.info(
                    f"LangChain workflow {self.workflow_name} completed successfully. "
                    f"Executed {self.execution_stats['nodes_executed']} nodes "
                    f"in {duration:.2f}s"
                )

            # Clear state
            self.workflow_state.clear()

        return False

    def execute_node(self, node_name: str, node_fn, *args, **kwargs) -> Any:
        """
        Execute a node in the workflow.

        Args:
            node_name: Name of the node
            node_fn: Function to execute
            *args, **kwargs: Arguments to pass to the node function

        Returns:
            Result of node execution
        """
        with self._lock:
            if not self._is_active:
                raise RuntimeError(f"Workflow {self.workflow_name} is not active")

            try:
                logger.debug(f"Executing node: {node_name}")
                result = node_fn(*args, **kwargs)
                self.execution_stats['nodes_executed'] += 1
                self.workflow_state[node_name] = result
                return result

            except Exception as e:
                logger.error(f"Error executing node {node_name}: {e}")
                self.execution_stats['errors'].append(f"{node_name}: {str(e)}")
                raise

    def get_state(self) -> Dict[str, Any]:
        """Get current workflow state"""
        with self._lock:
            return self.workflow_state.copy()


class LangChainLLMClientManager:
    """
    Framework-specific context manager for LangChain LLM client operations.
    Manages: LLM model initialization, connection pooling, and safe cleanup.
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        max_retries: int = 3,
        timeout_seconds: int = 60
    ):
        """
        Initialize LangChain LLM client manager.

        Args:
            model_name: Google Generative AI model name
            max_retries: Maximum number of retries for failed requests
            timeout_seconds: Timeout for LLM operations
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.llm_client: Optional[ChatGoogleGenerativeAI] = None
        self._lock = RLock()
        self._is_active = False
        self.request_count = 0
        self.error_count = 0

    def __enter__(self):
        """Initialize LLM client"""
        with self._lock:
            try:
                self.llm_client = ChatGoogleGenerativeAI(
                    model=self.model_name,
                    timeout=self.timeout_seconds,
                    max_retries=self.max_retries
                )
                self._is_active = True
                logger.info(f"Initialized LangChain LLM client: {self.model_name}")
                return self
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                self._is_active = False
                raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup LLM client resources"""
        with self._lock:
            self._is_active = False

            logger.info(
                f"LangChain LLM client {self.model_name} closed. "
                f"Requests: {self.request_count}, Errors: {self.error_count}"
            )

            # Clear client from memory
            self.llm_client = None

        return False

    def invoke(self, prompt: str) -> str:
        """
        Invoke LLM with a prompt.

        Args:
            prompt: Input prompt for the LLM

        Returns:
            LLM response
        """
        with self._lock:
            if not self._is_active or not self.llm_client:
                raise RuntimeError("LLM client is not active")

            try:
                self.request_count += 1
                response = self.llm_client.invoke(prompt)
                return response.content

            except Exception as e:
                self.error_count += 1
                logger.error(f"Error invoking LLM: {e}")
                raise


# Convenience context managers for common LangChain patterns
@contextmanager
def langchain_data_pipeline(source_type: str, source_path: str):
    """
    Convenience context manager for complete LangChain data pipeline.

    Usage:
        with langchain_data_pipeline("file", "data.txt") as pipeline:
            docs = pipeline.get_documents()
    """
    loader = LangChainDataSourceLoader(source_type, source_path)
    try:
        loader.__enter__()
        yield loader
    finally:
        loader.__exit__(None, None, None)


@contextmanager
def langchain_embedding_pipeline(model_name: str = "models/embedding-001"):
    """
    Convenience context manager for LangChain embedding pipeline.

    Usage:
        with langchain_embedding_pipeline() as embedder:
            embeddings = embedder.embed_documents(docs)
    """
    embedder = LangChainEmbeddingManager(model_name)
    try:
        embedder.__enter__()
        yield embedder
    finally:
        embedder.__exit__(None, None, None)


@contextmanager
def langchain_workflow_execution(workflow_name: str):
    """
    Convenience context manager for LangChain workflow execution.

    Usage:
        with langchain_workflow_execution("my_workflow") as executor:
            result = executor.execute_node("step1", my_function)
    """
    executor = LangChainWorkflowExecutor(workflow_name)
    try:
        executor.__enter__()
        yield executor
    finally:
        executor.__exit__(None, None, None)


# Usage examples:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example 1: Data source loading
    print("\n=== Example 1: LangChain Data Source Loading ===")
    with LangChainDataSourceLoader("file", "data.txt") as loader:
        doc = Document(page_content="Sample content")
        loader.add_document(doc)
        print(f"Loaded {len(loader.get_documents())} documents")

    # Example 2: Embedding operations
    print("\n=== Example 2: LangChain Embedding Operations ===")
    try:
        with LangChainEmbeddingManager() as embedder:
            docs = [Document(page_content="Test document")]
            embeddings = embedder.embed_documents(docs)
            print(f"Generated {len(embeddings)} embeddings")
    except Exception as e:
        print(f"Note: Embedding requires valid API keys: {e}")

    # Example 3: Workflow execution
    print("\n=== Example 3: LangChain Workflow Execution ===")
    with LangChainWorkflowExecutor("sample_workflow") as executor:
        result = executor.execute_node("step1", lambda: "Step 1 complete")
        result = executor.execute_node("step2", lambda: "Step 2 complete")
        print(f"Workflow state: {executor.get_state()}")

    # Example 4: LLM client operations
    print("\n=== Example 4: LangChain LLM Client Operations ===")
    try:
        with LangChainLLMClientManager("gemini-2.5-flash") as llm:
            response = llm.invoke("Hello, LLM!")
            print(f"LLM Response: {response}")
    except Exception as e:
        print(f"Note: LLM requires valid API keys: {e}")
