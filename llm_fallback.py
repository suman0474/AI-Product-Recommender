"""
LLM Fallback Utility
Provides automatic fallback from Google Gemini to OpenAI when Gemini fails
"""
import os
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model mappings: Gemini -> OpenAI equivalent
MODEL_MAPPINGS = {
    "gemini-2.5-flash": "gpt-4o-mini",
    "gemini-2.5-pro": "gpt-4o",
    "gemini-2.0-flash-exp": "gpt-4o-mini",
    "gemini-2.0-flash": "gpt-4o-mini",
    "gemini-1.5-pro": "gpt-4o",
    "gemini-1.5-flash": "gpt-4o-mini",
}


def get_openai_equivalent(gemini_model: str) -> str:
    """
    Get the OpenAI model equivalent for a Gemini model

    Args:
        gemini_model: Gemini model name

    Returns:
        OpenAI model name
    """
    return MODEL_MAPPINGS.get(gemini_model, "gpt-4o-mini")


def create_llm_with_fallback(
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.1,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> Any:
    """
    Create an LLM instance with automatic fallback from Gemini to OpenAI

    Args:
        model: Gemini model name (will be mapped to OpenAI if fallback is needed)
        temperature: Temperature for generation
        google_api_key: Google API key (optional, uses env var if not provided)
        openai_api_key: OpenAI API key (optional, uses env var if not provided)
        max_tokens: Maximum tokens for generation
        **kwargs: Additional arguments to pass to the LLM

    Returns:
        LLM instance (either ChatGoogleGenerativeAI or ChatOpenAI)
    """
    google_key = google_api_key or GOOGLE_API_KEY
    openai_key = openai_api_key or OPENAI_API_KEY

    # Try Google Gemini first
    if google_key:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            logger.info(f"[LLM_FALLBACK] Attempting to use Google Gemini: {model}")

            llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                google_api_key=google_key,
                max_output_tokens=max_tokens,
                **kwargs
            )

            # Test the model with a simple call to verify it works
            try:
                _ = llm.invoke("test")
                logger.info(f"[LLM_FALLBACK] Successfully initialized Google Gemini: {model}")
                return llm
            except Exception as test_error:
                logger.warning(f"[LLM_FALLBACK] Gemini model test failed: {test_error}")
                raise test_error

        except Exception as e:
            logger.warning(f"[LLM_FALLBACK] Failed to initialize Gemini ({model}): {e}")
            logger.info("[LLM_FALLBACK] Attempting fallback to OpenAI...")
    else:
        logger.warning("[LLM_FALLBACK] Google API key not available, skipping Gemini")

    # Fallback to OpenAI
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI

            openai_model = get_openai_equivalent(model)
            logger.info(f"[LLM_FALLBACK] Using OpenAI fallback: {openai_model} (equivalent to {model})")

            llm = ChatOpenAI(
                model=openai_model,
                temperature=temperature,
                openai_api_key=openai_key,
                max_tokens=max_tokens,
                **kwargs
            )

            logger.info(f"[LLM_FALLBACK] Successfully initialized OpenAI: {openai_model}")
            return llm

        except Exception as e:
            logger.error(f"[LLM_FALLBACK] Failed to initialize OpenAI ({openai_model}): {e}")
            raise RuntimeError(f"Both Gemini and OpenAI initialization failed. Last error: {e}")
    else:
        logger.error("[LLM_FALLBACK] OpenAI API key not available for fallback")
        raise RuntimeError("Gemini failed and OpenAI API key not available for fallback")


def create_llm_langchain(
    model: str = "gemini-2.0-flash-exp",
    temperature: float = 0.1,
    google_api_key: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    **kwargs
) -> Any:
    """
    Alias for create_llm_with_fallback for LangChain compatibility

    This is the recommended function to use across the codebase.
    """
    return create_llm_with_fallback(
        model=model,
        temperature=temperature,
        google_api_key=google_api_key,
        openai_api_key=openai_api_key,
        **kwargs
    )


class FallbackLLMClient:
    """
    Wrapper class for non-LangChain LLM usage with fallback support
    Similar to GeminiClient but with OpenAI fallback
    """

    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.google_api_key = api_key or GOOGLE_API_KEY
        self.openai_api_key = openai_api_key or OPENAI_API_KEY
        self.client = None
        self.client_type = None  # 'gemini' or 'openai'

        self._initialize_client()

    def _initialize_client(self):
        """Initialize the LLM client with fallback logic"""
        # Try Gemini first
        if self.google_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.google_api_key)
                self.client = genai.GenerativeModel(self.model_name)
                self.client_type = 'gemini'
                logger.info(f"[FallbackLLMClient] Using Gemini: {self.model_name}")
                return
            except Exception as e:
                logger.warning(f"[FallbackLLMClient] Gemini initialization failed: {e}")

        # Fallback to OpenAI
        if self.openai_api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.openai_api_key)
                self.client_type = 'openai'
                self.openai_model = get_openai_equivalent(self.model_name)
                logger.info(f"[FallbackLLMClient] Using OpenAI fallback: {self.openai_model}")
                return
            except Exception as e:
                logger.error(f"[FallbackLLMClient] OpenAI initialization failed: {e}")
                raise RuntimeError(f"Both Gemini and OpenAI initialization failed")

        raise RuntimeError("No valid API keys available for LLM initialization")

    def invoke(self, prompt: str, **kwargs) -> str:
        """
        Invoke the LLM with a prompt

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments

        Returns:
            Generated text response
        """
        try:
            if self.client_type == 'gemini':
                response = self.client.generate_content(
                    prompt,
                    generation_config={
                        'temperature': self.temperature,
                        **kwargs
                    }
                )
                return response.text

            elif self.client_type == 'openai':
                response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    **kwargs
                )
                return response.choices[0].message.content

        except Exception as e:
            logger.error(f"[FallbackLLMClient] Error invoking LLM: {e}")

            # Try to failover if using Gemini
            if self.client_type == 'gemini' and self.openai_api_key:
                logger.info("[FallbackLLMClient] Attempting runtime failover to OpenAI...")
                try:
                    from openai import OpenAI
                    client = OpenAI(api_key=self.openai_api_key)
                    openai_model = get_openai_equivalent(self.model_name)

                    response = client.chat.completions.create(
                        model=openai_model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.temperature,
                        **kwargs
                    )

                    # Update client for future calls
                    self.client = client
                    self.client_type = 'openai'
                    self.openai_model = openai_model
                    logger.info(f"[FallbackLLMClient] Runtime failover successful to {openai_model}")

                    return response.choices[0].message.content
                except Exception as fallback_error:
                    logger.error(f"[FallbackLLMClient] Runtime failover failed: {fallback_error}")

            raise e


# Convenience functions
def get_default_llm(temperature: float = 0.1, model: str = "gemini-2.0-flash-exp") -> Any:
    """Get a default LLM instance with fallback"""
    return create_llm_with_fallback(model=model, temperature=temperature)


def get_llm_for_task(task_type: str = "general", temperature: float = 0.1) -> Any:
    """
    Get an LLM optimized for a specific task type

    Args:
        task_type: Type of task ('general', 'fast', 'precise', 'creative')
        temperature: Temperature setting

    Returns:
        LLM instance with fallback
    """
    task_configs = {
        "general": {"model": "gemini-2.0-flash-exp", "temp": 0.1},
        "fast": {"model": "gemini-2.0-flash", "temp": 0.0},
        "precise": {"model": "gemini-2.5-flash", "temp": 0.0},
        "creative": {"model": "gemini-2.5-pro", "temp": 0.7},
    }

    config = task_configs.get(task_type, task_configs["general"])
    return create_llm_with_fallback(
        model=config["model"],
        temperature=temperature if temperature != 0.1 else config["temp"]
    )
