"""
Pydantic Models for API

Data models for request/response validation in the API.
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Service status")
    rag_ready: bool = Field(..., description="Whether RAG system is ready")
    llm_ready: bool = Field(..., description="Whether LLM client is ready")
    collection_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Collection information"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of relevant chunks to retrieve"
    )
    use_rag: bool = Field(
        default=True,
        description="Whether to use RAG context"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    
    @validator('query')
    def validate_query(cls, v):
        """Validate query string."""
        v = v.strip()
        if not v:
            raise ValueError('Query cannot be empty')
        return v


class SourceDocument(BaseModel):
    """Model for source document information."""
    
    chunk_preview: str = Field(..., description="Preview of the chunk")
    source: str = Field(..., description="Source file path")
    page: Optional[int] = Field(default=None, description="Page number")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index")
    total_chunks: Optional[int] = Field(default=None, description="Total chunks in document")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    context_chunks: List[str] = Field(
        default=[],
        description="Relevant context chunks"
    )
    sources: List[SourceDocument] = Field(
        default=[],
        description="Source documents"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score"
    )
    processing_time: float = Field(
        ...,
        ge=0.0,
        description="Time taken to process query in seconds"
    )
    query_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the query"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )


class IngestRequest(BaseModel):
    """Request model for ingest endpoint."""
    
    file_path: str = Field(..., description="Path to document file")
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate file path."""
        import os
        v = v.strip()
        if not v:
            raise ValueError('File path cannot be empty')
        if not os.path.exists(v):
            raise ValueError(f'File does not exist: {v}')
        if not v.lower().endswith('.pdf'):
            raise ValueError('Only PDF files are supported')
        return v


class IngestResponse(BaseModel):
    """Response model for ingest endpoint."""
    
    success: bool = Field(..., description="Whether ingestion was successful")
    file_path: str = Field(..., description="Path to ingested file")
    chunks_count: int = Field(
        default=0,
        ge=0,
        description="Number of chunks created"
    )
    document_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the document"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Document metadata"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )


class LLMConfigRequest(BaseModel):
    """Request model for LLM configuration."""
    
    api_base: Optional[str] = Field(
        default=None,
        description="Base URL for the LLM API"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for authentication"
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name to use"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Temperature for generation"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=10000,
        description="Maximum tokens to generate"
    )
    timeout: Optional[int] = Field(
        default=None,
        ge=1,
        le=300,
        description="Request timeout in seconds"
    )


class LLMConfigResponse(BaseModel):
    """Response model for LLM configuration."""
    
    success: bool = Field(..., description="Whether configuration was successful")
    message: str = Field(..., description="Response message")
    config: Dict[str, Any] = Field(..., description="Current configuration")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )


class CollectionInfo(BaseModel):
    """Model for collection information."""
    
    name: str = Field(..., description="Collection name")
    count: int = Field(..., ge=0, description="Number of documents in collection")
    embedding_dimension: Optional[int] = Field(
        default=None,
        description="Embedding dimension"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Collection metadata"
    )
    persist_directory: Optional[str] = Field(
        default=None,
        description="Persist directory"
    )


class ClearCollectionResponse(BaseModel):
    """Response model for clear collection endpoint."""
    
    success: bool = Field(..., description="Whether operation was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )


class ErrorResponse(BaseModel):
    """Model for error responses."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Error detail")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Error timestamp"
    )


class StreamingChunk(BaseModel):
    """Model for streaming response chunks."""
    
    type: str = Field(..., description="Chunk type (answer, context, chunk, error)")
    content: str = Field(..., description="Chunk content")
    index: Optional[int] = Field(default=None, description="Index for context chunks")
    timestamp: float = Field(
        default_factory=lambda: datetime.now().timestamp(),
        description="Chunk timestamp"
    )


# Models for batch operations
class BatchIngestRequest(BaseModel):
    """Request model for batch ingest endpoint."""
    
    directory_path: str = Field(..., description="Path to directory containing PDFs")
    
    @validator('directory_path')
    def validate_directory_path(cls, v):
        """Validate directory path."""
        import os
        v = v.strip()
        if not v:
            raise ValueError('Directory path cannot be empty')
        if not os.path.isdir(v):
            raise ValueError(f'Directory does not exist: {v}')
        return v


class BatchIngestResult(BaseModel):
    """Model for batch ingest result."""
    
    file_path: str = Field(..., description="File path")
    success: bool = Field(..., description="Whether ingestion was successful")
    chunks_count: int = Field(default=0, description="Number of chunks created")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class BatchIngestResponse(BaseModel):
    """Response model for batch ingest endpoint."""
    
    total_files: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successful ingestions")
    failed: int = Field(..., description="Number of failed ingestions")
    results: List[BatchIngestResult] = Field(..., description="Individual results")
    total_chunks: int = Field(..., description="Total chunks created")
    processing_time: float = Field(..., description="Total processing time in seconds")
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="Response timestamp"
    )
