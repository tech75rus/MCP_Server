"""
API Endpoints

FastAPI endpoints for the AI Agent API.
"""

import time
import logging
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_core import get_rag_system, query_rag, ingest_document
from api_client import LLMClient, LLMConfig

logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., description="User query")
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve")
    use_rag: bool = Field(default=True, description="Whether to use RAG context")
    stream: bool = Field(default=False, description="Whether to stream the response")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    context_chunks: List[str] = Field(default=[], description="Relevant context chunks")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents")
    confidence: float = Field(default=0.0, description="Confidence score")
    processing_time: float = Field(..., description="Time taken to process query")


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    
    success: bool = Field(..., description="Whether ingestion was successful")
    file_path: str = Field(..., description="Path to ingested file")
    chunks_count: int = Field(default=0, description="Number of chunks created")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    rag_ready: bool = Field(..., description="Whether RAG system is ready")
    llm_ready: bool = Field(..., description="Whether LLM client is ready")
    collection_info: Optional[Dict[str, Any]] = Field(default=None, description="Collection information")


class LLMConfigRequest(BaseModel):
    """Request model for LLM configuration."""
    
    api_base: Optional[str] = Field(default=None, description="Base URL for the LLM API")
    api_key: Optional[str] = Field(default=None, description="API key for authentication")
    model: Optional[str] = Field(default=None, description="Model name to use")
    temperature: Optional[float] = Field(default=None, description="Temperature for generation")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    timeout: Optional[int] = Field(default=None, description="Request timeout in seconds")


# Endpoints
@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    try:
        rag_system = get_rag_system()
        
        # Test LLM connection
        llm_ready = False
        try:
            llm_client = LLMClient()
            # Simple test query
            test_response = await llm_client.chat_completion([
                {"role": "user", "content": "test"}
            ])
            llm_ready = True
            await llm_client.close()
        except Exception as e:
            logger.warning(f"LLM test failed: {e}")
        
        # Get collection info
        collection_info = await rag_system.get_collection_info()
        
        return HealthResponse(
            status="healthy",
            rag_ready=True,
            llm_ready=llm_ready,
            collection_info=collection_info
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            rag_ready=False,
            llm_ready=False,
            collection_info=None
        )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    start_time = time.time()
    
    try:
        if request.use_rag:
            # Use RAG with context
            result = await query_rag(request.query, request.top_k)
            
            response = QueryResponse(
                answer=result.answer,
                context_chunks=result.context_chunks,
                sources=result.sources,
                confidence=result.confidence,
                processing_time=time.time() - start_time
            )
        else:
            # Direct LLM query without RAG
            llm_client = LLMClient()
            llm_response = await llm_client.chat_completion([
                {"role": "user", "content": request.query}
            ])
            await llm_client.close()
            
            response = QueryResponse(
                answer=llm_response.content,
                processing_time=time.time() - start_time
            )
        
        logger.info(f"Query processed in {response.processing_time:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/ingest")
async def ingest_file(file: UploadFile = File(...)):
    """Ingest a document from file upload."""
    import tempfile
    import os
    
    # Check file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        # Read uploaded file
        content = await file.read()
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Ingest the document
        result = await ingest_document(temp_path)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        if result["success"]:
            return IngestResponse(
                success=True,
                file_path=file.filename,
                chunks_count=result["chunks_count"]
            )
        else:
            return IngestResponse(
                success=False,
                file_path=file.filename,
                error=result.get("error", "Unknown error")
            )
        
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/ingest/path")
async def ingest_path(file_path: str = Form(...)):
    """Ingest a document from file path."""
    try:
        result = await ingest_document(file_path)
        
        if result["success"]:
            return IngestResponse(
                success=True,
                file_path=result["file_path"],
                chunks_count=result["chunks_count"]
            )
        else:
            return IngestResponse(
                success=False,
                file_path=result["file_path"],
                error=result.get("error", "Unknown error")
            )
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/collection/info")
async def get_collection_info():
    """Get information about the vector collection."""
    try:
        rag_system = get_rag_system()
        info = await rag_system.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/collection/clear")
async def clear_collection():
    """Clear all documents from the collection."""
    try:
        rag_system = get_rag_system()
        success = await rag_system.clear_collection()
        
        return {
            "success": success,
            "message": "Collection cleared" if success else "Failed to clear collection"
        }
        
    except Exception as e:
        logger.error(f"Failed to clear collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/config")
async def update_llm_config(config: LLMConfigRequest):
    """Update LLM configuration."""
    try:
        # Create new config with provided values
        llm_config = LLMConfig()
        
        if config.api_base is not None:
            llm_config.api_base = config.api_base
        if config.api_key is not None:
            llm_config.api_key = config.api_key
        if config.model is not None:
            llm_config.model = config.model
        if config.temperature is not None:
            llm_config.temperature = config.temperature
        if config.max_tokens is not None:
            llm_config.max_tokens = config.max_tokens
        if config.timeout is not None:
            llm_config.timeout = config.timeout
        
        # Test the new configuration
        llm_client = LLMClient(llm_config)
        test_response = await llm_client.chat_completion([
            {"role": "user", "content": "test"}
        ])
        await llm_client.close()
        
        # TODO: Persist the configuration
        
        return {
            "success": True,
            "message": "LLM configuration updated successfully",
            "config": llm_config.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Failed to update LLM config: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update LLM config: {str(e)}")


@router.get("/llm/models")
async def get_available_models():
    """Get available LLM models."""
    # This would typically query the LLM API for available models
    # For now, return some common models
    
    models = {
        "ollama": [
            "llama3.2",
            "mistral",
            "codellama",
            "neural-chat"
        ],
        "openai": [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-3.5-turbo"
        ],
        "anthropic": [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307"
        ]
    }
    
    return models


# Streaming endpoint for query
@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Stream query results."""
    from fastapi.responses import StreamingResponse
    import json
    
    async def generate():
        """Generate streaming response."""
        try:
            if request.use_rag:
                # Get RAG result first
                result = await query_rag(request.query, request.top_k)
                
                # Stream the answer
                yield f"data: {json.dumps({'type': 'answer', 'content': result.answer})}\n\n"
                
                # Stream context chunks
                for i, chunk in enumerate(result.context_chunks):
                    yield f"data: {json.dumps({'type': 'context', 'index': i, 'content': chunk[:200]})}\n\n"
                
            else:
                # Stream LLM response
                llm_client = LLMClient()
                
                async for chunk in llm_client.stream_chat_completion([
                    {"role": "user", "content": request.query}
                ]):
                    if chunk.strip():
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
                
                await llm_client.close()
                
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )
