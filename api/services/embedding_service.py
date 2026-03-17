"""
Embedding Service for generating vector embeddings from text.

This module provides embedding generation using sentence-transformers
or falls back to a simple TF-IDF based approach when transformers
are not available.
"""

import os
import logging
from typing import List, Optional, Dict, Any
import hashlib
import json

# Configure logging
logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings.
    
    Uses sentence-transformers for high-quality embeddings when available,
    falls back to a simple hash-based approach for basic functionality.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        dimension: int = 384
    ):
        """
        Initialize the embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on ("cpu" or "cuda")
            dimension: Dimension of the embedding vectors
        """
        self.model_name = model_name
        self.device = device
        self.dimension = dimension
        self._model = None
        self._use_transformers = False
        
        # Try to load sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name, device=device)
            self._use_transformers = True
            logger.info(f"Loaded embedding model: {model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Using fallback embedding method."
            )
        except Exception as e:
            logger.warning(
                f"Failed to load embedding model: {e}. "
                "Using fallback embedding method."
            )
    
    def embed(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        if self._use_transformers and self._model:
            return self._model.encode(text).tolist()
        else:
            return self._fallback_embed(text)
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        if self._use_transformers and self._model:
            embeddings = self._model.encode(texts)
            return [emb.tolist() for emb in embeddings]
        else:
            return [self._fallback_embed(text) for text in texts]
    
    def _fallback_embed(self, text: str) -> List[float]:
        """
        Generate a simple hash-based embedding as fallback.
        
        This is NOT a proper semantic embedding but provides
        consistent vectors for basic similarity computation.
        
        Args:
            text: Text to embed
            
        Returns:
            Simple embedding vector
        """
        import numpy as np
        
        # Normalize text
        text = text.lower().strip()
        
        # Create multiple hash values for different n-grams
        embedding = np.zeros(self.dimension, dtype=np.float32)
        
        # Use character n-grams
        for n in range(1, 4):
            for i in range(len(text) - n + 1):
                ngram = text[i:i+n]
                hash_val = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
                idx = hash_val % self.dimension
                embedding[idx] += 1.0
        
        # Use word-level hashing
        words = text.split()
        for word in words:
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            idx = hash_val % self.dimension
            embedding[idx] += 2.0  # Weight words more heavily
        
        # Normalize the embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding.tolist()
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        import numpy as np
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "using_transformers": self._use_transformers
        }


class EmbeddingCache:
    """
    Cache for storing pre-computed embeddings.
    
    Reduces computation for frequently accessed texts.
    """
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_count: Dict[str, int] = {}
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash key for text."""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding if available.
        
        Args:
            text: Text to look up
            
        Returns:
            Cached embedding or None
        """
        key = self._hash_text(text)
        if key in self._cache:
            self._access_count[key] = self._access_count.get(key, 0) + 1
            return self._cache[key]
        return None
    
    def set(self, text: str, embedding: List[float]):
        """
        Store embedding in cache.
        
        Args:
            text: Text key
            embedding: Embedding vector to cache
        """
        key = self._hash_text(text)
        
        # Evict least recently used if cache is full
        if len(self._cache) >= self.max_size:
            lru_key = min(self._access_count, key=self._access_count.get)
            del self._cache[lru_key]
            del self._access_count[lru_key]
        
        self._cache[key] = embedding
        self._access_count[key] = 1
    
    def get_or_compute(
        self,
        text: str,
        embedding_service: EmbeddingService
    ) -> List[float]:
        """
        Get cached embedding or compute and cache it.
        
        Args:
            text: Text to embed
            embedding_service: Service to compute embedding if not cached
            
        Returns:
            Embedding vector
        """
        cached = self.get(text)
        if cached is not None:
            return cached
        
        embedding = embedding_service.embed(text)
        self.set(text, embedding)
        return embedding
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._access_count.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "total_accesses": sum(self._access_count.values())
        }


class EmbeddingServiceFactory:
    """Factory for creating embedding service instances."""
    
    _instance: Optional[EmbeddingService] = None
    _cache: Optional[EmbeddingCache] = None
    
    @classmethod
    def get_instance(
        cls,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        dimension: int = 384,
        use_cache: bool = True,
        cache_size: int = 10000
    ) -> EmbeddingService:
        """
        Get or create the embedding service singleton.
        
        Args:
            model_name: Name of the embedding model
            device: Device to run on
            dimension: Embedding dimension
            use_cache: Whether to use caching
            cache_size: Maximum cache size
            
        Returns:
            EmbeddingService instance
        """
        if cls._instance is None:
            cls._instance = EmbeddingService(
                model_name=model_name,
                device=device,
                dimension=dimension
            )
            
            if use_cache:
                cls._cache = EmbeddingCache(max_size=cache_size)
        
        return cls._instance
    
    @classmethod
    def get_cache(cls) -> Optional[EmbeddingCache]:
        """Get the embedding cache."""
        return cls._cache
    
    @classmethod
    def reset(cls):
        """Reset the singleton instance."""
        cls._instance = None
        cls._cache = None
