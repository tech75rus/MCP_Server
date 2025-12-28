"""
LLM API Client for AI Agent RAG System

This module provides a client for interacting with various LLM APIs.
Supports OpenAI-compatible APIs, Anthropic, and local models.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict

import httpx
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """Configuration for LLM API connection."""
    
    api_base: str = Field(
        default=os.getenv("LLM_API_BASE", "http://localhost:11434"),
        description="Base URL for the LLM API"
    )
    api_key: Optional[str] = Field(
        default=os.getenv("LLM_API_KEY"),
        description="API key for authentication (if required)"
    )
    model: str = Field(
        default=os.getenv("LLM_MODEL", "llama3.2"),
        description="Model name to use"
    )
    temperature: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", "0.7")),
        description="Temperature for generation (0.0 to 1.0)"
    )
    max_tokens: int = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", "2048")),
        description="Maximum tokens to generate"
    )
    timeout: int = Field(
        default=int(os.getenv("LLM_TIMEOUT", "60")),
        description="Request timeout in seconds"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class LLMResponse(BaseModel):
    """Response from LLM API."""
    
    content: str = Field(description="Generated text content")
    model: str = Field(description="Model used for generation")
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage information"
    )
    finish_reason: Optional[str] = Field(
        default=None,
        description="Reason why generation finished"
    )
    raw_response: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Raw response from API"
    )


class LLMClient:
    """
    Client for interacting with LLM APIs.
    
    Supports:
    - OpenAI-compatible APIs (Ollama, vLLM, etc.)
    - Anthropic Claude API
    - Custom endpoints
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client.
        
        Args:
            config: LLM configuration. If None, uses defaults from environment.
        """
        self.config = config or LLMConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.api_base,
            timeout=self.config.timeout,
            headers=self._get_headers()
        )
        logger.info(f"Initialized LLMClient with config: {self.config}")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AI-Agent-RAG/1.0"
        }
        
        if self.config.api_key:
            # Support both OpenAI and Anthropic style headers
            if "anthropic" in self.config.api_base.lower():
                headers["x-api-key"] = self.config.api_key
                headers["anthropic-version"] = "2023-06-01"
            else:
                headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        return headers
    
    def _prepare_openai_request(
        self, 
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Dict[str, Any]:
        """Prepare request in OpenAI format."""
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        }
    
    def _prepare_anthropic_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Dict[str, Any]:
        """Prepare request in Anthropic format."""
        return {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream
        }
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Send chat completion request to LLM API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            
        Returns:
            LLMResponse object with generated content
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        try:
            # Add system prompt if provided
            if system_prompt:
                messages = [{"role": "system", "content": system_prompt}] + messages
            
            # Detect API type and prepare request
            if "anthropic" in self.config.api_base.lower():
                request_data = self._prepare_anthropic_request(messages, stream)
                endpoint = "/v1/messages"
            else:
                request_data = self._prepare_openai_request(messages, stream)
                endpoint = "/v1/chat/completions"
            
            logger.debug(f"Sending request to {endpoint}: {request_data}")
            
            # Send request
            response = await self.client.post(
                endpoint,
                json=request_data,
                headers=self._get_headers()
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Extract content based on API type
            if "anthropic" in self.config.api_base.lower():
                content = result["content"][0]["text"]
                model = result["model"]
                usage = result.get("usage")
                finish_reason = None  # Anthropic doesn't provide this in same format
            else:
                content = result["choices"][0]["message"]["content"]
                model = result["model"]
                usage = result.get("usage")
                finish_reason = result["choices"][0].get("finish_reason")
            
            return LLMResponse(
                content=content,
                model=model,
                usage=usage,
                finish_reason=finish_reason,
                raw_response=result
            )
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during LLM API call: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during LLM API call: {e}")
            raise
    
    async def generate_with_context(
        self,
        query: str,
        context: List[str],
        system_prompt: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate response with RAG context.
        
        Args:
            query: User query
            context: List of relevant context chunks
            system_prompt: Optional system prompt
            
        Returns:
            LLMResponse with generated answer
        """
        # Prepare context
        context_text = "\n\n".join([
            f"[Context {i+1}]: {chunk}"
            for i, chunk in enumerate(context)
        ])
        
        # Create messages
        messages = [
            {
                "role": "user",
                "content": f"""Based on the following context, answer the question.

Context:
{context_text}

Question: {query}

Answer:"""
            }
        ]
        
        # Use custom system prompt or default
        if not system_prompt:
            system_prompt = """You are a helpful AI assistant. Use the provided context to answer the question accurately.
If the context doesn't contain relevant information, say so and provide a general answer based on your knowledge.
Be concise and factual."""
        
        return await self.chat_completion(messages, system_prompt)
    
    async def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None
    ):
        """
        Stream chat completion response.
        
        Args:
            messages: List of message dictionaries
            system_prompt: Optional system prompt
            
        Yields:
            Chunks of generated text
        """
        # Add system prompt if provided
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        # Prepare streaming request
        if "anthropic" in self.config.api_base.lower():
            request_data = self._prepare_anthropic_request(messages, stream=True)
            endpoint = "/v1/messages"
        else:
            request_data = self._prepare_openai_request(messages, stream=True)
            endpoint = "/v1/chat/completions"
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as stream_client:
            async with stream_client.stream(
                "POST",
                f"{self.config.api_base}{endpoint}",
                json=request_data,
                headers=self._get_headers()
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.strip():
                        yield line
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


# Convenience functions
async def get_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """
    Get an LLM client instance.
    
    Args:
        config: Optional LLM configuration
        
    Returns:
        LLMClient instance
    """
    return LLMClient(config)


async def query_llm(
    query: str,
    context: Optional[List[str]] = None,
    config: Optional[LLMConfig] = None
) -> str:
    """
    Convenience function to query LLM with optional context.
    
    Args:
        query: User query
        context: Optional list of context chunks
        config: Optional LLM configuration
        
    Returns:
        Generated response as string
    """
    client = LLMClient(config)
    
    try:
        if context:
            response = await client.generate_with_context(query, context)
        else:
            messages = [{"role": "user", "content": query}]
            response = await client.chat_completion(messages)
        
        return response.content
    finally:
        await client.close()


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example():
        """Example usage of the LLM client."""
        # Example 1: Basic chat
        print("Example 1: Basic chat completion")
        client = LLMClient()
        
        messages = [
            {"role": "user", "content": "What is the capital of France?"}
        ]
        
        try:
            response = await client.chat_completion(messages)
            print(f"Response: {response.content}")
            print(f"Model: {response.model}")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            await client.close()
        
        # Example 2: With RAG context
        print("\nExample 2: With RAG context")
        
        context = [
            "Paris is the capital and most populous city of France.",
            "France is a country located in Western Europe.",
            "The Eiffel Tower is a famous landmark in Paris."
        ]
        
        response_text = await query_llm(
            "What is the capital of France?",
            context=context
        )
        print(f"Response with context: {response_text}")
    
    asyncio.run(example())
