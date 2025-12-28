"""
RAG Core Module

This module orchestrates the entire RAG pipeline:
1. Document loading and processing
2. Text chunking
3. Embedding generation
4. Vector storage and retrieval
5. LLM integration
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Import services
from services.pdf_loader import PDFLoader
from services.text_chunker import TextChunker
from services.embedding_service import EmbeddingService
from services.chroma_service import ChromaService
from api_client import LLMClient, LLMConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG system."""
    
    # Document processing
    pdf_extract_images: bool = Field(
        default=os.getenv("PDF_EXTRACT_IMAGES", "false").lower() == "true"
    )
    pdf_extract_tables: bool = Field(
        default=os.getenv("PDF_EXTRACT_TABLES", "true").lower() == "true"
    )
    
    # Text chunking
    chunk_size: int = Field(
        default=int(os.getenv("CHUNK_SIZE", "1000"))
    )
    chunk_overlap: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", "200"))
    )
    
    # Embeddings
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    
    # Vector database
    chroma_persist_directory: str = Field(
        default=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/chroma_db")
    )
    chroma_collection_name: str = Field(
        default=os.getenv("CHROMA_COLLECTION_NAME", "documents")
    )
    
    # LLM
    llm_config: LLMConfig = Field(default_factory=LLMConfig)


class QueryResult(BaseModel):
    """Result of a RAG query."""
    
    query: str = Field(description="Original query")
    answer: str = Field(description="Generated answer")
    context_chunks: List[str] = Field(
        description="Relevant context chunks used for generation"
    )
    sources: List[Dict[str, Any]] = Field(
        description="Source documents metadata"
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score (0.0 to 1.0)"
    )


class RAGSystem:
    """
    Main RAG system orchestrator.
    
    Coordinates all components:
    - Document processing
    - Vector database
    - Retrieval
    - LLM generation
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize RAG system with configuration."""
        self.config = config or RAGConfig()
        
        # Initialize components
        self.pdf_loader = PDFLoader(
            extract_images=self.config.pdf_extract_images,
            extract_tables=self.config.pdf_extract_tables
        )
        
        self.text_chunker = TextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        
        self.embedding_service = EmbeddingService(
            model_name=self.config.embedding_model
        )
        
        self.chroma_service = ChromaService(
            persist_directory=self.config.chroma_persist_directory,
            collection_name=self.config.chroma_collection_name,
            embedding_function=self.embedding_service.get_embedding_function()
        )
        
        self.llm_client = LLMClient(self.config.llm_config)
        
        logger.info("RAG system initialized")
    
    async def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a single document into the RAG system.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Ingesting document: {file_path}")
            
            # 1. Load and parse document
            document = self.pdf_loader.load_pdf(file_path)
            logger.info(f"Loaded document: {document.metadata}")
            
            # 2. Extract text
            text = self.pdf_loader.extract_text(document)
            logger.info(f"Extracted text length: {len(text)} characters")
            
            # 3. Chunk text
            chunks = self.text_chunker.chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            # 4. Generate embeddings
            embeddings = self.embedding_service.embed_texts(chunks)
            logger.info(f"Generated {len(embeddings)} embeddings")
            
            # 5. Store in vector database
            metadatas = [
                {
                    "source": file_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    **document.metadata
                }
                for i in range(len(chunks))
            ]
            
            ids = [f"{Path(file_path).stem}_{i}" for i in range(len(chunks))]
            
            self.chroma_service.add_documents(
                texts=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Successfully ingested {file_path}")
            
            return {
                "success": True,
                "file_path": file_path,
                "chunks_count": len(chunks),
                "metadata": document.metadata
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document {file_path}: {e}")
            return {
                "success": False,
                "file_path": file_path,
                "error": str(e)
            }
    
    async def query(self, query: str, top_k: int = 5) -> QueryResult:
        """
        Query the RAG system.
        
        Args:
            query: User query
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            QueryResult with answer and context
        """
        try:
            logger.info(f"Processing query: {query}")
            
            # 1. Generate query embedding
            query_embedding = self.embedding_service.embed_text(query)
            
            # 2. Retrieve relevant chunks
            results = self.chroma_service.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            if not results["documents"] or not results["documents"][0]:
                logger.warning("No relevant documents found")
                # Fallback to LLM without context
                response = await self.llm_client.chat_completion([
                    {"role": "user", "content": query}
                ])
                
                return QueryResult(
                    query=query,
                    answer=response.content,
                    context_chunks=[],
                    sources=[],
                    confidence=0.0
                )
            
            # 3. Prepare context
            retrieved_chunks = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            
            # Calculate confidence based on similarity scores
            # Convert distances to similarity scores (assuming cosine distance)
            similarities = [1 - (dist / 2) for dist in distances]  # Approximate
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # 4. Generate answer with context
            response = await self.llm_client.generate_with_context(
                query=query,
                context=retrieved_chunks
            )
            
            # 5. Prepare sources
            sources = []
            for i, metadata in enumerate(metadatas):
                sources.append({
                    "chunk": retrieved_chunks[i][:200] + "...",  # Preview
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get("page", 0),
                    "similarity": similarities[i]
                })
            
            return QueryResult(
                query=query,
                answer=response.content,
                context_chunks=retrieved_chunks,
                sources=sources,
                confidence=avg_similarity
            )
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector collection."""
        return self.chroma_service.get_collection_info()
    
    async def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        return self.chroma_service.clear_collection()
    
    async def close(self):
        """Cleanup resources."""
        await self.llm_client.close()
        self.chroma_service.close()


# Singleton instance for easy access
_rag_system: Optional[RAGSystem] = None


def get_rag_system(config: Optional[RAGConfig] = None) -> RAGSystem:
    """
    Get or create singleton RAG system instance.
    
    Args:
        config: Optional RAG configuration
        
    Returns:
        RAGSystem instance
    """
    global _rag_system
    
    if _rag_system is None:
        _rag_system = RAGSystem(config)
    
    return _rag_system


async def query_rag(query: str, top_k: int = 5) -> QueryResult:
    """
    Convenience function to query RAG system.
    
    Args:
        query: User query
        top_k: Number of relevant chunks to retrieve
        
    Returns:
        QueryResult with answer and context
    """
    rag_system = get_rag_system()
    return await rag_system.query(query, top_k)


async def ingest_document(file_path: str) -> Dict[str, Any]:
    """
    Convenience function to ingest a document.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Dictionary with ingestion results
    """
    rag_system = get_rag_system()
    return await rag_system.ingest_document(file_path)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example():
        """Example usage of the RAG system."""
        # Initialize system
        rag = RAGSystem()
        
        # Get collection info
        info = await rag.get_collection_info()
        print(f"Collection info: {info}")
        
        # Example query (without ingested documents)
        try:
            result = await rag.query("What is machine learning?")
            print(f"\nQuery: {result.query}")
            print(f"Answer: {result.answer}")
            print(f"Confidence: {result.confidence:.2f}")
            print(f"Sources: {len(result.sources)}")
        except Exception as e:
            print(f"Query error: {e}")
        
        await rag.close()
    
    asyncio.run(example())
