"""
Embedding Service

Handles text embedding generation using various models.
"""

import os
import logging
import numpy as np
from typing import List, Union, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Using fallback embeddings.")


try:
    import chromadb
    from chromadb.api.types import EmbeddingFunction
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed. ChromaDB integration disabled.")


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""
    
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"  # "cpu" or "cuda"
    normalize_embeddings: bool = True
    batch_size: int = 32
    cache_folder: Optional[str] = None


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config or EmbeddingConfig()
        self.model = None
        
        logger.info(f"Initializing EmbeddingService with model: {self.config.model_name}")
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model."""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.info(f"Loading sentence transformer model: {self.config.model_name}")
                
                self.model = SentenceTransformer(
                    self.config.model_name,
                    device=self.config.device,
                    cache_folder=self.config.cache_folder
                )
                
                # Test the model
                test_embedding = self.model.encode(
                    ["test"],
                    normalize_embeddings=self.config.normalize_embeddings,
                    batch_size=self.config.batch_size
                )
                
                self.embedding_dimension = test_embedding.shape[1]
                logger.info(
                    f"Model loaded successfully. Embedding dimension: {self.embedding_dimension}"
                )
                
            else:
                logger.warning("Using fallback embedding method")
                self.model = None
                self.embedding_dimension = 384  # Default for fallback
                
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            self.embedding_dimension = 384
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            List of embedding values
        """
        return self.embed_texts([text])[0]
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embeddings
        """
        if not texts:
            return []
        
        logger.debug(f"Generating embeddings for {len(texts)} texts")
        
        try:
            if self.model is not None and SENTENCE_TRANSFORMERS_AVAILABLE:
                # Use sentence-transformers
                embeddings = self.model.encode(
                    texts,
                    normalize_embeddings=self.config.normalize_embeddings,
                    batch_size=self.config.batch_size,
                    show_progress_bar=False
                )
                
                # Convert to list of lists
                embeddings_list = embeddings.tolist()
                
            else:
                # Fallback: simple TF-IDF like embeddings
                embeddings_list = self._fallback_embed_texts(texts)
            
            logger.debug(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return [[0.0] * self.embedding_dimension for _ in texts]
    
    def _fallback_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Fallback embedding method when sentence-transformers is not available.
        
        This is a simple word frequency based embedding for development/testing.
        """
        import hashlib
        
        embeddings = []
        
        for text in texts:
            # Create a deterministic pseudo-embedding based on text hash
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert hash to embedding vector
            embedding = []
            for i in range(0, len(text_hash), 2):
                if len(embedding) >= self.embedding_dimension:
                    break
                
                hex_pair = text_hash[i:i+2]
                value = int(hex_pair, 16) / 255.0  # Normalize to 0-1
                embedding.append(value)
            
            # Pad or truncate to embedding dimension
            if len(embedding) < self.embedding_dimension:
                embedding.extend([0.0] * (self.embedding_dimension - len(embedding)))
            else:
                embedding = embedding[:self.embedding_dimension]
            
            # Normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = (np.array(embedding) / norm).tolist()
            
            embeddings.append(embedding)
        
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_dimension
    
    def get_embedding_function(self) -> Optional[Callable]:
        """
        Get embedding function compatible with ChromaDB.
        
        Returns:
            Embedding function or None if ChromaDB not available
        """
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, cannot create embedding function")
            return None
        
        class ChromaDBEmbeddingFunction(EmbeddingFunction):
            """Embedding function wrapper for ChromaDB."""
            
            def __init__(self, embedding_service):
                self.embedding_service = embedding_service
            
            def __call__(self, texts: List[str]) -> List[List[float]]:
                return self.embedding_service.embed_texts(texts)
        
        return ChromaDBEmbeddingFunction(self)
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        if len(embedding1) != len(embedding2):
            raise ValueError("Embeddings must have the same dimension")
        
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def find_most_similar(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5
    ) -> List[tuple[int, float]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity) tuples
        """
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            sim = self.similarity(query_embedding, candidate)
            similarities.append((i, sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]


# Pre-configured models
class MiniLMEmbeddingService(EmbeddingService):
    """Pre-configured with all-MiniLM-L6-v2 model."""
    
    def __init__(self):
        config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            normalize_embeddings=True
        )
        super().__init__(config)


class MPNetEmbeddingService(EmbeddingService):
    """Pre-configured with all-mpnet-base-v2 model."""
    
    def __init__(self):
        config = EmbeddingConfig(
            model_name="all-mpnet-base-v2",
            normalize_embeddings=True
        )
        super().__init__(config)


# Example usage
if __name__ == "__main__":
    import sys
    
    # Test the embedding service
    service = EmbeddingService()
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence is transforming the world",
        "Python is a popular programming language for data science",
        "Machine learning models require large amounts of data"
    ]
    
    print(f"Embedding dimension: {service.get_embedding_dimension()}")
    print()
    
    # Generate embeddings
    embeddings = service.embed_texts(test_texts)
    
    for i, (text, embedding) in enumerate(zip(test_texts, embeddings)):
        print(f"Text {i+1}: {text[:50]}...")
        print(f"  Embedding length: {len(embedding)}")
        print(f"  First 5 values: {embedding[:5]}")
        print()
    
    # Test similarity
    print("Similarity between first two texts:")
    sim = service.similarity(embeddings[0], embeddings[1])
    print(f"  Similarity: {sim:.4f}")
    
    # Test most similar
    print("\nMost similar to first text:")
    similarities = service.find_most_similar(embeddings[0], embeddings, top_k=2)
    
    for idx, similarity in similarities:
        print(f"  Text {idx+1}: {similarity:.4f} - {test_texts[idx][:50]}...")
