"""
MCP Server for Continue CLI

Model Context Protocol server that provides RAG capabilities to Continue IDE.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_core import RAGSystem, RAGConfig, get_rag_system, query_rag, ingest_document
from api_client import LLMClient, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., description="User query")
    top_k: int = Field(default=5, description="Number of relevant chunks to retrieve")
    use_rag: bool = Field(default=True, description="Whether to use RAG context")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    answer: str = Field(..., description="Generated answer")
    context_chunks: List[str] = Field(default=[], description="Relevant context chunks")
    sources: List[Dict[str, Any]] = Field(default=[], description="Source documents")
    confidence: float = Field(default=0.0, description="Confidence score")
    processing_time: float = Field(..., description="Time taken to process query")


class IngestRequest(BaseModel):
    """Request model for ingest endpoint."""
    
    file_path: str = Field(..., description="Path to document file")


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


# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting MCP Server...")
    
    # Initialize RAG system
    try:
        rag_config = RAGConfig()
        rag_system = get_rag_system(rag_config)
        logger.info("RAG system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down MCP Server...")
    if rag_system:
        await rag_system.close()


# Create FastAPI app
app = FastAPI(
    title="AI Agent MCP Server",
    description="MCP Server providing RAG capabilities for Continue CLI",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint."""
    return {
        "message": "AI Agent MCP Server",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/query",
            "/ingest",
            "/collection/info",
            "/collection/clear"
        ]
    }


@app.get("/health")
async def health() -> HealthResponse:
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


@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    """Query the RAG system."""
    import time
    
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


@app.post("/ingest")
async def ingest(request: IngestRequest) -> IngestResponse:
    """Ingest a document into the RAG system."""
    try:
        result = await ingest_document(request.file_path)
        
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


@app.get("/collection/info")
async def get_collection_info() -> Dict[str, Any]:
    """Get information about the vector collection."""
    try:
        rag_system = get_rag_system()
        info = await rag_system.get_collection_info()
        return info
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/collection/clear")
async def clear_collection() -> Dict[str, Any]:
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


# MCP Protocol Endpoints (for Continue CLI)
@app.get("/mcp/tools")
async def get_mcp_tools() -> Dict[str, Any]:
    """Get available MCP tools."""
    return {
        "tools": [
            {
                "name": "query_rag",
                "description": "Query the RAG system with a question",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The question to ask"},
                        "top_k": {"type": "integer", "default": 5, "description": "Number of relevant chunks"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "ingest_document",
                "description": "Ingest a document into the RAG system",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the document file"}
                    },
                    "required": ["file_path"]
                }
            },
            {
                "name": "get_collection_info",
                "description": "Get information about the vector collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    }


@app.post("/mcp/execute")
async def execute_mcp_tool(request: Dict[str, Any]) -> Dict[str, Any]:
    """Execute an MCP tool."""
    tool_name = request.get("name")
    arguments = request.get("arguments", {})
    
    try:
        if tool_name == "query_rag":
            query_text = arguments.get("query")
            top_k = arguments.get("top_k", 5)
            
            if not query_text:
                raise ValueError("Query text is required")
            
            result = await query_rag(query_text, top_k)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result.answer
                    }
                ],
                "metadata": {
                    "context_chunks": result.context_chunks,
                    "sources": result.sources,
                    "confidence": result.confidence
                }
            }
            
        elif tool_name == "ingest_document":
            file_path = arguments.get("file_path")
            
            if not file_path:
                raise ValueError("File path is required")
            
            result = await ingest_document(file_path)
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Document ingested: {result['file_path']}" if result["success"] \
                               else f"Failed to ingest: {result.get('error')}"
                    }
                ],
                "metadata": result
            }
            
        elif tool_name == "get_collection_info":
            rag_system = get_rag_system()
            info = await rag_system.get_collection_info()
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Collection '{info.get('name')}' has {info.get('count', 0)} documents"
                    }
                ],
                "metadata": info
            }
            
        else:
            raise HTTPException(status_code=404, detail=f"Tool not found: {tool_name}")
        
    except Exception as e:
        logger.error(f"MCP tool execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# CLI entry point
def main():
    """CLI entry point for running the server."""
    import uvicorn
    
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"Starting MCP Server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        reload=False  # Set to True for development
    )


if __name__ == "__main__":
    main()
