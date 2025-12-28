"""
ChromaDB Service

Handles vector database operations for storing and retrieving embeddings.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.api.types import (
    EmbeddingFunction,
    Embeddings,
    Documents,
    Metadatas,
    IDs
)

logger = logging.getLogger(__name__)


class ChromaService:
    """Service for ChromaDB vector database operations."""
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        embedding_function: Optional[EmbeddingFunction] = None,
        host: Optional[str] = None,
        port: Optional[int] = None
    ):
        """
        Initialize ChromaDB service.
        
        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
            embedding_function: Function to generate embeddings
            host: ChromaDB server host (if using client/server mode)
            port: ChromaDB server port
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        
        # Create directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Initialize ChromaDB client
        if host and port:
            # Client/server mode
            logger.info(f"Connecting to ChromaDB at {host}:{port}")
            self.client = chromadb.HttpClient(
                host=host,
                port=port,
                settings=Settings()
            )
        else:
            # Persistent mode
            logger.info(f"Using persistent ChromaDB at {persist_directory}")
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"ChromaService initialized with collection: {collection_name}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """Get existing collection or create new one."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
            return collection
            
        except Exception as e:
            # Collection doesn't exist, create new one
            logger.info(f"Creating new collection: {self.collection_name}")
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}  # Cosine similarity
            )
            return collection
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the collection.
        
        Args:
            texts: List of document texts
            embeddings: Optional pre-computed embeddings
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        # Generate IDs if not provided
        if ids is None:
            import hashlib
            import time
            ids = []
            for i, text in enumerate(texts):
                text_hash = hashlib.md5(text.encode()).hexdigest()[:16]
                ids.append(f"doc_{text_hash}_{int(time.time())}_{i}")
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in texts]
        
        # Add to collection
        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(texts)} documents to collection")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the collection.
        
        Args:
            query_texts: Query texts (will be embedded)
            query_embeddings: Pre-computed query embeddings
            n_results: Number of results to return
            where: Filter by metadata
            where_document: Filter by document content
            
        Returns:
            Query results
        """
        if not query_texts and not query_embeddings:
            raise ValueError("Either query_texts or query_embeddings must be provided")
        
        try:
            results = self.collection.query(
                query_texts=query_texts,
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            logger.debug(f"Query returned {len(results.get('documents', [[]])[0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
    
    def get_all_documents(self, limit: int = 1000) -> Dict[str, Any]:
        """
        Get all documents from the collection.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            All documents with metadata
        """
        try:
            # ChromaDB doesn't have a direct 'get all' method
            # We can query with a dummy embedding to get all documents
            dummy_embedding = [[0.0] * 384]  # Assuming 384-dim embeddings
            
            results = self.collection.query(
                query_embeddings=dummy_embedding,
                n_results=limit
            )
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting all documents: {e}")
            raise
    
    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Delete documents from the collection.
        
        Args:
            ids: IDs of documents to delete
            where: Filter by metadata
        """
        try:
            self.collection.delete(
                ids=ids,
                where=where
            )
            logger.info(f"Deleted documents: ids={ids}, where={where}")
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            raise
    
    def update_document(
        self,
        id: str,
        text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update a document in the collection.
        
        Args:
            id: Document ID
            text: New text content
            embedding: New embedding
            metadata: New metadata
        """
        try:
            self.collection.update(
                ids=[id],
                documents=[text] if text else None,
                embeddings=[embedding] if embedding else None,
                metadatas=[metadata] if metadata else None
            )
            logger.info(f"Updated document: {id}")
            
        except Exception as e:
            logger.error(f"Error updating document {id}: {e}")
            raise
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            # Get count
            count = self.collection.count()
            
            # Get metadata
            metadata = self.collection.metadata or {}
            
            # Try to get a sample to determine embedding dimension
            sample = self.collection.peek(limit=1)
            embedding_dim = None
            if sample["embeddings"] and sample["embeddings"][0]:
                embedding_dim = len(sample["embeddings"][0])
            
            return {
                "name": self.collection_name,
                "count": count,
                "embedding_dimension": embedding_dim,
                "metadata": metadata,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                "name": self.collection_name,
                "error": str(e)
            }
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            # Delete the collection
            self.client.delete_collection(self.collection_name)
            
            # Recreate empty collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False
    
    def close(self):
        """Close the ChromaDB client."""
        # ChromaDB PersistentClient doesn't have explicit close method
        # But we can clean up references
        self.client = None
        self.collection = None
        logger.info("ChromaService closed")


# Example usage
if __name__ == "__main__":
    import sys
    
    # Initialize service
    service = ChromaService()
    
    # Get collection info
    info = service.get_collection_info()
    print(f"Collection info: {info}")
    
    # Add some test documents
    test_docs = [
        "Artificial intelligence is transforming industries.",
        "Machine learning requires large datasets for training.",
        "Deep learning models use neural networks with many layers.",
        "Natural language processing enables computers to understand human language."
    ]
    
    print(f"\nAdding {len(test_docs)} test documents...")
    ids = service.add_documents(
        texts=test_docs,
        metadatas=[{"source": "test", "index": i} for i in range(len(test_docs))]
    )
    print(f"Added document IDs: {ids}")
    
    # Query
    print("\nQuerying for 'machine learning'...")
    results = service.query(
        query_texts=["machine learning"],
        n_results=2
    )
    
    if results["documents"]:
        for i, doc in enumerate(results["documents"][0]):
            print(f"  Result {i+1}: {doc[:80]}...")
            print(f"    Distance: {results['distances'][0][i]:.4f}")
    
    # Get updated info
    info = service.get_collection_info()
    print(f"\nUpdated collection info: {info}")
    
    # Cleanup
    service.clear_collection()
    print("\nCollection cleared")
