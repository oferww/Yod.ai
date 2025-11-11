"""
LLM Provider Abstraction Layer

Supports multiple LLM providers (Cohere, Google AI Studio) with a unified interface.
The provider is selected via the LLM_PROVIDER environment variable.
"""

import os
import logging
import asyncio
from typing import Generator, AsyncGenerator, Optional, List, Dict, Any
from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage

logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def stream(self, messages: List[BaseMessage]) -> Generator[str, None, None]:
        """Stream responses from the LLM.
        
        Args:
            messages: List of messages (SystemMessage, HumanMessage, etc.)
            
        Yields:
            str: Chunks of the response text
        """
        pass
    
    @abstractmethod
    async def astream(self, messages: List[BaseMessage]) -> AsyncGenerator[str, None]:
        """Async stream responses from the LLM.
        
        Args:
            messages: List of messages (SystemMessage, HumanMessage, etc.)
            
        Yields:
            str: Chunks of the response text
        """
        pass
    
    @abstractmethod
    def invoke(self, messages: List[BaseMessage]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Get a complete response from the LLM (non-streaming).
        
        Args:
            messages: List of messages (SystemMessage, HumanMessage, etc.)
            
        Returns:
            tuple: (response_text, token_metadata) where token_metadata contains:
                - input_tokens: Number of prompt tokens
                - output_tokens: Number of response tokens
        """
        pass
    
    @abstractmethod
    async def ainvoke(self, messages: List[BaseMessage]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Get a complete response from the LLM asynchronously (non-streaming, non-blocking).
        
        Args:
            messages: List of messages (SystemMessage, HumanMessage, etc.)
            
        Returns:
            tuple: (response_text, token_metadata) where token_metadata contains:
                - input_tokens: Number of prompt tokens
                - output_tokens: Number of response tokens
        """
        pass
    
    @abstractmethod
    def get_token_counts(self) -> Optional[Dict[str, Any]]:
        """Get token counts from the last response (for streaming).
        
        Returns:
            dict with input_tokens and output_tokens, or None if not available
        """
        pass


class CohereLLMProvider(BaseLLMProvider):
    """Cohere LLM provider using langchain_cohere."""
    
    def __init__(self, api_key: str, model: str = "command-a-vision-07-2025", 
                 temperature: float = 0.7, max_tokens: int = 1000):
        """Initialize Cohere provider.
        
        Args:
            api_key: Cohere API key
            model: Model name
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        from langchain_cohere import ChatCohere
        
        self.llm = ChatCohere(
            cohere_api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # Expose the underlying LangChain LLM for memory compatibility
        self._langchain_llm = self.llm
        # Store last token counts
        self._last_token_counts = None
    
    def stream(self, messages: List[BaseMessage]) -> Generator[str, None, None]:
        """Stream responses from Cohere."""
        print("[COHERE_STREAM] Starting stream", flush=True)
        chunk_count = 0
        for chunk in self.llm.stream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                chunk_count += 1
                # Extract token counts from chunk if available
                if hasattr(chunk, 'additional_kwargs') and 'token_count' in chunk.additional_kwargs:
                    token_info = chunk.additional_kwargs['token_count']
                    self._last_token_counts = {
                        'input_tokens': token_info.get('prompt_tokens', 0),
                        'output_tokens': token_info.get('response_tokens', 0),
                    }
                    print(f"[COHERE_STREAM] Chunk {chunk_count}: Extracted tokens - input={self._last_token_counts['input_tokens']}, output={self._last_token_counts['output_tokens']}", flush=True)
                else:
                    print(f"[COHERE_STREAM] Chunk {chunk_count}: No token_count in additional_kwargs", flush=True)
                yield content
        print(f"[COHERE_STREAM] Stream complete. Total chunks: {chunk_count}, Final token counts: {self._last_token_counts}", flush=True)
    
    async def astream(self, messages: List[BaseMessage]) -> AsyncGenerator[str, None]:
        """Async stream responses from Cohere."""
        print("[COHERE_ASTREAM] Starting async stream", flush=True)
        chunk_count = 0
        async for chunk in self.llm.astream(messages):
            content = chunk.content if hasattr(chunk, 'content') else str(chunk)
            if content:
                chunk_count += 1
                # Extract token counts from chunk if available
                if hasattr(chunk, 'additional_kwargs') and 'token_count' in chunk.additional_kwargs:
                    token_info = chunk.additional_kwargs['token_count']
                    self._last_token_counts = {
                        'input_tokens': token_info.get('prompt_tokens', 0),
                        'output_tokens': token_info.get('response_tokens', 0),
                    }
                    print(f"[COHERE_ASTREAM] Chunk {chunk_count}: Extracted tokens - input={self._last_token_counts['input_tokens']}, output={self._last_token_counts['output_tokens']}", flush=True)
                else:
                    print(f"[COHERE_ASTREAM] Chunk {chunk_count}: No token_count in additional_kwargs", flush=True)
                yield content
        print(f"[COHERE_ASTREAM] Stream complete. Total chunks: {chunk_count}, Final token counts: {self._last_token_counts}", flush=True)
    
    def invoke(self, messages: List[BaseMessage]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Get complete response from Cohere."""
        print("[COHERE_INVOKE] Starting invoke", flush=True)
        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract token counts from response
        token_metadata = None
        if hasattr(response, 'additional_kwargs') and 'token_count' in response.additional_kwargs:
            token_info = response.additional_kwargs['token_count']
            token_metadata = {
                'input_tokens': token_info.get('prompt_tokens', 0),
                'output_tokens': token_info.get('response_tokens', 0),
            }
            self._last_token_counts = token_metadata
            print(f"[COHERE_INVOKE] Extracted tokens - input={token_metadata['input_tokens']}, output={token_metadata['output_tokens']}", flush=True)
        else:
            print("[COHERE_INVOKE] No token_count in response.additional_kwargs", flush=True)
        
        print(f"[COHERE_INVOKE] Invoke complete. Response length: {len(response_text)}, Token metadata: {token_metadata}", flush=True)
        return response_text, token_metadata
    
    async def ainvoke(self, messages: List[BaseMessage]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Get complete response from Cohere asynchronously (non-blocking)."""
        print("[COHERE_AINVOKE] Starting async invoke", flush=True)
        response = await self.llm.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract token counts from response
        token_metadata = None
        if hasattr(response, 'additional_kwargs') and 'token_count' in response.additional_kwargs:
            token_info = response.additional_kwargs['token_count']
            token_metadata = {
                'input_tokens': token_info.get('prompt_tokens', 0),
                'output_tokens': token_info.get('response_tokens', 0),
            }
            self._last_token_counts = token_metadata
            print(f"[COHERE_AINVOKE] Extracted tokens - input={token_metadata['input_tokens']}, output={token_metadata['output_tokens']}", flush=True)
        else:
            print("[COHERE_AINVOKE] No token_count in response.additional_kwargs", flush=True)
        
        print(f"[COHERE_AINVOKE] Async invoke complete. Response length: {len(response_text)}, Token metadata: {token_metadata}", flush=True)
        return response_text, token_metadata
    
    def get_token_counts(self) -> Optional[Dict[str, Any]]:
        """Get token counts from the last response."""
        print(f"[COHERE_GET_TOKENS] Retrieving token counts: {self._last_token_counts}", flush=True)
        return self._last_token_counts
    
    def get_langchain_llm(self):
        """Get the underlying LangChain LLM for memory compatibility."""
        return self._langchain_llm


class GoogleAIProvider(BaseLLMProvider):
    """Google AI Studio (Gemini) provider using langchain_google_genai."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro", 
                 temperature: float = 0.7, max_tokens: int = 1000):
        """Initialize Google AI provider.
        
        Args:
            api_key: Google AI API key
            model: Model name (e.g., gemini-1.5-pro, gemini-1.5-flash)
                  Note: gemini-2.5-flash has known streaming truncation issues
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
        """
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        # Expose the underlying LangChain LLM for memory compatibility
        self._langchain_llm = self.llm
        # Store last token counts
        self._last_token_counts = None
        # Store messages for token count retrieval after streaming
        self._last_messages = None
    
    def stream(self, messages: List[BaseMessage]) -> Generator[str, None, None]:
        """Stream responses from Google AI.
        
        Note: Google's streaming API does not provide token counts in chunks.
        Token counts are retrieved after streaming completes via invoke().
        """
        print("[GOOGLE_STREAM] Starting stream", flush=True)
        chunk_count = 0
        total_chars = 0
        # Store messages for later token count retrieval
        self._last_messages = messages
        try:
            for chunk in self.llm.stream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    chunk_count += 1
                    total_chars += len(content)
                    print(f"[GOOGLE_STREAM] Chunk {chunk_count}: {len(content)} chars, total: {total_chars} chars", flush=True)
                    yield content
            print(f"[GOOGLE_STREAM] Stream complete. Total chunks: {chunk_count}, Total chars: {total_chars}", flush=True)
            print(f"[GOOGLE_STREAM] Note: Token counts will be retrieved via invoke() after streaming", flush=True)
        except Exception as e:
            print(f"[GOOGLE_STREAM] ERROR during streaming: {e}", flush=True)
            raise
    
    async def astream(self, messages: List[BaseMessage]) -> AsyncGenerator[str, None]:
        """Async stream responses from Google AI.
        
        Note: Google's streaming API does not provide token counts in chunks.
        Token counts are retrieved after streaming completes via invoke().
        """
        print("[GOOGLE_ASTREAM] Starting async stream", flush=True)
        chunk_count = 0
        total_chars = 0
        # Store messages for later token count retrieval
        self._last_messages = messages
        try:
            async for chunk in self.llm.astream(messages):
                content = chunk.content if hasattr(chunk, 'content') else str(chunk)
                if content:
                    chunk_count += 1
                    total_chars += len(content)
                    print(f"[GOOGLE_ASTREAM] Chunk {chunk_count}: {len(content)} chars, total: {total_chars} chars", flush=True)
                    yield content
            print(f"[GOOGLE_ASTREAM] Stream complete. Total chunks: {chunk_count}, Total chars: {total_chars}", flush=True)
            print(f"[GOOGLE_ASTREAM] Note: Token counts will be retrieved via invoke() after streaming", flush=True)
        except Exception as e:
            print(f"[GOOGLE_ASTREAM] ERROR during async streaming: {e}", flush=True)
            raise
    
    def invoke(self, messages: List[BaseMessage]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Get complete response from Google AI."""
        print("[GOOGLE_INVOKE] Starting invoke", flush=True)
        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract token counts from response (usage_metadata)
        token_metadata = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            token_metadata = {
                'input_tokens': response.usage_metadata.get('input_tokens', 0),
                'output_tokens': response.usage_metadata.get('output_tokens', 0),
            }
            self._last_token_counts = token_metadata
            print(f"[GOOGLE_INVOKE] Extracted tokens - input={token_metadata['input_tokens']}, output={token_metadata['output_tokens']}", flush=True)
        else:
            print("[GOOGLE_INVOKE] No usage_metadata in response", flush=True)
        
        print(f"[GOOGLE_INVOKE] Invoke complete. Response length: {len(response_text)}, Token metadata: {token_metadata}", flush=True)
        return response_text, token_metadata
    
    async def ainvoke(self, messages: List[BaseMessage]) -> tuple[str, Optional[Dict[str, Any]]]:
        """Get complete response from Google AI asynchronously (non-blocking)."""
        print("[GOOGLE_AINVOKE] Starting async invoke", flush=True)
        response = await self.llm.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Extract token counts from response (usage_metadata)
        token_metadata = None
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            token_metadata = {
                'input_tokens': response.usage_metadata.get('input_tokens', 0),
                'output_tokens': response.usage_metadata.get('output_tokens', 0),
            }
            self._last_token_counts = token_metadata
            print(f"[GOOGLE_AINVOKE] Extracted tokens - input={token_metadata['input_tokens']}, output={token_metadata['output_tokens']}", flush=True)
        else:
            print("[GOOGLE_AINVOKE] No usage_metadata in response", flush=True)
        
        print(f"[GOOGLE_AINVOKE] Async invoke complete. Response length: {len(response_text)}, Token metadata: {token_metadata}", flush=True)
        return response_text, token_metadata
    
    def get_token_counts(self) -> Optional[Dict[str, Any]]:
        """Get token counts from the last response.
        
        For Google AI, token counts are not available during streaming.
        This method calls invoke() on the stored messages to get token counts.
        """
        # If we already have token counts cached, return them
        if self._last_token_counts:
            print(f"[GOOGLE_GET_TOKENS] Returning cached token counts: {self._last_token_counts}", flush=True)
            return self._last_token_counts
        
        # If we have stored messages, invoke to get token counts
        if self._last_messages:
            print(f"[GOOGLE_GET_TOKENS] No cached tokens, invoking to get token counts...", flush=True)
            try:
                response = self.llm.invoke(self._last_messages)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Extract token counts from response
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    self._last_token_counts = {
                        'input_tokens': response.usage_metadata.get('input_tokens', 0),
                        'output_tokens': response.usage_metadata.get('output_tokens', 0),
                    }
                    print(f"[GOOGLE_GET_TOKENS] ✓ Retrieved tokens via invoke: input={self._last_token_counts['input_tokens']}, output={self._last_token_counts['output_tokens']}", flush=True)
                    return self._last_token_counts
                else:
                    print(f"[GOOGLE_GET_TOKENS] ✗ No usage_metadata in invoke response", flush=True)
                    return None
            except Exception as e:
                print(f"[GOOGLE_GET_TOKENS] ✗ Error invoking for token counts: {e}", flush=True)
                return None
        
        print(f"[GOOGLE_GET_TOKENS] ✗ No cached tokens and no stored messages", flush=True)
        return None
    
    def get_langchain_llm(self):
        """Get the underlying LangChain LLM for memory compatibility."""
        return self._langchain_llm


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    @staticmethod
    def create_provider(
        provider_type: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> BaseLLMProvider:
        """Create an LLM provider instance.
        
        Args:
            provider_type: Provider type ('cohere' or 'google'). 
                          Defaults to LLM_PROVIDER env var or 'cohere'.
            api_key: API key for the provider. Defaults to appropriate env var.
            model: Model name. Defaults to appropriate env var or provider default.
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            BaseLLMProvider: Configured provider instance
            
        Raises:
            ValueError: If provider_type is not supported or API key is missing
        """
        # Determine provider type
        if provider_type is None:
            provider_type = os.getenv("LLM_PROVIDER", "cohere").lower()
        else:
            provider_type = provider_type.lower()
        
        print(f"[LLM_PROVIDER] Creating provider: {provider_type}")
        
        if provider_type == "cohere":
            # Get API key
            if api_key is None:
                api_key = os.getenv("COHERE_API_KEY")
                if not api_key:
                    raise ValueError("COHERE_API_KEY environment variable is required for Cohere provider")
            
            # Get model
            if model is None:
                model = os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025")
            
            print(f"[LLM_PROVIDER] Using Cohere model: {model}")
            return CohereLLMProvider(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        elif provider_type == "google":
            # Get API key
            if api_key is None:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable is required for Google provider")
            
            # Get model
            if model is None:
                model = os.getenv("GOOGLE_MODEL", "gemini-1.5-pro")
            
            print(f"[LLM_PROVIDER] Using Google AI model: {model}")
            return GoogleAIProvider(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_type}. Supported: 'cohere', 'google'")
    
    @staticmethod
    def get_provider_config() -> dict:
        """Get current provider configuration from environment.
        
        Returns:
            dict: Configuration with provider, model, and api_key_set status
        """
        provider = os.getenv("LLM_PROVIDER", "cohere").lower()
        
        if provider == "cohere":
            return {
                "provider": "cohere",
                "model": os.getenv("COHERE_CHAT_MODEL", "command-a-vision-07-2025"),
                "api_key_set": bool(os.getenv("COHERE_API_KEY"))
            }
        elif provider == "google":
            return {
                "provider": "google",
                "model": os.getenv("GOOGLE_MODEL", "gemini-1.5-flash"),
                "api_key_set": bool(os.getenv("GOOGLE_API_KEY"))
            }
        else:
            return {
                "provider": provider,
                "model": None,
                "api_key_set": False,
                "error": f"Unsupported provider: {provider}"
            }
