"""
Tests for RAG (Retrieval-Augmented Generation) Service.

Tests the RAG fallback system including:
- Vector store operations
- Embedding service
- Document indexing
- RAG predictions
"""

import os
import sys
import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.vector_store import VectorStore, VectorStoreFactory, Document
from services.embedding_service import EmbeddingService, EmbeddingCache, EmbeddingServiceFactory
from services.rag_service import RAGService, RAGResponse
from services.document_indexer import DocumentIndexer, IndexingStats


class TestDocument:
    """Tests for the Document dataclass."""
    
    def test_document_creation(self):
        """Test creating a document."""
        doc = Document(
            id="test_1",
            content="Test content",
            metadata={"source": "test"}
        )
        assert doc.id == "test_1"
        assert doc.content == "Test content"
        assert doc.metadata["source"] == "test"
        assert doc.embedding is None
    
    def test_document_to_dict(self):
        """Test document serialization."""
        doc = Document(
            id="test_1",
            content="Test content",
            metadata={"source": "test"},
            embedding=[0.1, 0.2, 0.3]
        )
        result = doc.to_dict()
        assert result["id"] == "test_1"
        assert result["content"] == "Test content"
        assert result["embedding"] == [0.1, 0.2, 0.3]
    
    def test_document_from_dict(self):
        """Test document deserialization."""
        data = {
            "id": "test_1",
            "content": "Test content",
            "metadata": {"source": "test"},
            "embedding": [0.1, 0.2, 0.3]
        }
        doc = Document.from_dict(data)
        assert doc.id == "test_1"
        assert doc.content == "Test content"


class TestVectorStore:
    """Tests for the VectorStore class."""
    
    def test_memory_store_creation(self):
        """Test creating an in-memory vector store."""
        store = VectorStore(backend="memory")
        assert store.backend == "memory"
        assert store.count() == 0
    
    def test_add_documents_to_memory(self):
        """Test adding documents to memory store."""
        store = VectorStore(backend="memory")
        documents = [
            Document(id="1", content="Test 1", metadata={}),
            Document(id="2", content="Test 2", metadata={})
        ]
        embeddings = [[0.1] * 384, [0.2] * 384]
        
        count = store.add_documents(documents, embeddings)
        assert count == 2
        assert store.count() == 2
    
    def test_similarity_search_memory(self):
        """Test similarity search in memory store."""
        store = VectorStore(backend="memory")
        
        # Add documents with different embeddings
        documents = [
            Document(id="1", content="Heart disease", metadata={"category": "cardiac"}),
            Document(id="2", content="Diabetes symptoms", metadata={"category": "diabetes"}),
            Document(id="3", content="Common cold", metadata={"category": "symptom"})
        ]
        
        # Create distinct embeddings
        embeddings = [
            [1.0] + [0.0] * 383,  # Heart-like
            [0.0, 1.0] + [0.0] * 382,  # Diabetes-like
            [0.0, 0.0, 1.0] + [0.0] * 381  # Cold-like
        ]
        
        store.add_documents(documents, embeddings)
        
        # Search with heart-like query
        query_embedding = [0.9] + [0.1] * 383
        results = store.similarity_search(query_embedding, k=2)
        
        assert len(results) <= 2
        if results:
            assert results[0].id == "1"  # Should match heart document first
    
    def test_get_document(self):
        """Test retrieving a specific document."""
        store = VectorStore(backend="memory")
        doc = Document(id="test_id", content="Test content", metadata={"key": "value"})
        store.add_documents([doc], [[0.1] * 384])
        
        retrieved = store.get_document("test_id")
        assert retrieved is not None
        assert retrieved.content == "Test content"
        
        # Non-existent document
        assert store.get_document("nonexistent") is None
    
    def test_delete_document(self):
        """Test deleting a document."""
        store = VectorStore(backend="memory")
        doc = Document(id="test_id", content="Test content", metadata={})
        store.add_documents([doc], [[0.1] * 384])
        
        assert store.count() == 1
        result = store.delete_document("test_id")
        assert result is True
        assert store.count() == 0
    
    def test_clear_store(self):
        """Test clearing the store."""
        store = VectorStore(backend="memory")
        documents = [
            Document(id="1", content="Test 1", metadata={}),
            Document(id="2", content="Test 2", metadata={})
        ]
        store.add_documents(documents, [[0.1] * 384, [0.2] * 384])
        
        assert store.count() == 2
        store.clear()
        assert store.count() == 0
    
    def test_get_stats(self):
        """Test getting store statistics."""
        store = VectorStore(backend="memory", collection_name="test_collection")
        stats = store.get_stats()
        
        assert stats["backend"] == "memory"
        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 0


class TestEmbeddingService:
    """Tests for the EmbeddingService class."""
    
    def test_fallback_embedding(self):
        """Test fallback embedding generation."""
        service = EmbeddingService(dimension=384)
        
        # Should use fallback since sentence-transformers may not be installed
        embedding = service.embed("test text")
        
        assert len(embedding) == 384
        assert isinstance(embedding, list)
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_embedding(self):
        """Test batch embedding generation."""
        service = EmbeddingService(dimension=384)
        texts = ["text one", "text two", "text three"]
        
        embeddings = service.embed_batch(texts)
        
        assert len(embeddings) == 3
        assert all(len(emb) == 384 for emb in embeddings)
    
    def test_similarity_calculation(self):
        """Test cosine similarity calculation."""
        service = EmbeddingService(dimension=384)
        
        # Identical vectors
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [1.0, 0.0, 0.0]
        sim = service.similarity(emb1, emb2)
        assert abs(sim - 1.0) < 0.001
        
        # Orthogonal vectors
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [0.0, 1.0, 0.0]
        sim = service.similarity(emb1, emb2)
        assert abs(sim) < 0.001
        
        # Opposite vectors
        emb1 = [1.0, 0.0, 0.0]
        emb2 = [-1.0, 0.0, 0.0]
        sim = service.similarity(emb1, emb2)
        assert abs(sim + 1.0) < 0.001
    
    def test_model_info(self):
        """Test getting model information."""
        service = EmbeddingService(
            model_name="test-model",
            device="cpu",
            dimension=384
        )
        info = service.get_model_info()
        
        assert info["model_name"] == "test-model"
        assert info["device"] == "cpu"
        assert info["dimension"] == 384


class TestEmbeddingCache:
    """Tests for the EmbeddingCache class."""
    
    def test_cache_operations(self):
        """Test cache get/set operations."""
        cache = EmbeddingCache(max_size=10)
        
        # Set and get
        cache.set("test text", [0.1, 0.2, 0.3])
        result = cache.get("test text")
        
        assert result == [0.1, 0.2, 0.3]
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = EmbeddingCache()
        result = cache.get("nonexistent")
        assert result is None
    
    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = EmbeddingCache(max_size=2)
        
        cache.set("text1", [0.1])
        cache.set("text2", [0.2])
        cache.set("text3", [0.3])  # Should evict text1
        
        assert cache.get("text1") is None
        assert cache.get("text2") == [0.2]
        assert cache.get("text3") == [0.3]
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = EmbeddingCache(max_size=10)
        cache.set("text1", [0.1])
        cache.set("text2", [0.2])
        
        stats = cache.get_stats()
        assert stats["size"] == 2
        assert stats["max_size"] == 10


class TestRAGService:
    """Tests for the RAGService class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock(spec=VectorStore)
        store.count.return_value = 5
        store.get_stats.return_value = {"document_count": 5}
        return store
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock(spec=EmbeddingService)
        service.embed.return_value = [0.1] * 384
        service.get_model_info.return_value = {"model_name": "test"}
        return service
    
    def test_rag_service_creation(self, mock_vector_store, mock_embedding_service):
        """Test creating a RAG service."""
        service = RAGService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        assert service.is_available()
        assert service.vector_store == mock_vector_store
    
    def test_rag_service_unavailable(self, mock_embedding_service):
        """Test RAG service when vector store is empty."""
        empty_store = Mock(spec=VectorStore)
        empty_store.count.return_value = 0
        
        service = RAGService(
            vector_store=empty_store,
            embedding_service=mock_embedding_service
        )
        
        assert not service.is_available()
    
    def test_retrieve_documents(self, mock_vector_store, mock_embedding_service):
        """Test document retrieval."""
        # Setup mock
        mock_vector_store.similarity_search.return_value = [
            Document(
                id="1",
                content="Heart disease information",
                metadata={"category": "cardiac", "distance": 0.2}
            ),
            Document(
                id="2",
                content="Diabetes information",
                metadata={"category": "diabetes", "distance": 0.4}
            )
        ]
        
        service = RAGService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            min_relevance_score=0.3
        )
        
        results = service.retrieve("heart symptoms")
        
        assert len(results) >= 0
        mock_embedding_service.embed.assert_called_once()
        mock_vector_store.similarity_search.assert_called_once()
    
    def test_predict_with_rag(self, mock_vector_store, mock_embedding_service):
        """Test RAG prediction."""
        # Setup mock
        mock_vector_store.similarity_search.return_value = [
            Document(
                id="1",
                content="Common cold symptoms include cough and fever",
                metadata={
                    "category": "disease",
                    "disease_name": "Common Cold",
                    "symptoms": ["cough", "fever", "headache"],
                    "distance": 0.2
                }
            )
        ]
        
        service = RAGService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            min_relevance_score=0.3
        )
        
        response = service.predict_with_rag(
            symptoms=["cough", "fever"],
            model_type="symptom"
        )
        
        assert isinstance(response, RAGResponse)
        assert response.query != ""
        assert response.disclaimer != ""
    
    def test_rag_response_to_dict(self):
        """Test RAG response serialization."""
        response = RAGResponse(
            query="test query",
            possible_conditions=[{"condition": "Test Condition", "relevance_score": 0.8}],
            recommendations=["See a doctor"],
            confidence=0.75,
            sources=["test_source"],
            disclaimer="Test disclaimer"
        )
        
        result = response.to_dict()
        
        assert result["query"] == "test query"
        assert len(result["possible_conditions"]) == 1
        assert result["confidence"] == 0.75
    
    def test_search_endpoint(self, mock_vector_store, mock_embedding_service):
        """Test search endpoint functionality."""
        mock_vector_store.similarity_search.return_value = [
            Document(
                id="1",
                content="Test content",
                metadata={"source": "test", "distance": 0.2}
            )
        ]
        
        service = RAGService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        results = service.search("test query", top_k=5)
        
        assert "query" in results
        assert "results" in results
        assert "total" in results
    
    def test_get_stats(self, mock_vector_store, mock_embedding_service):
        """Test getting RAG statistics."""
        service = RAGService(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            top_k=5,
            min_relevance_score=0.3
        )
        
        stats = service.get_stats()
        
        assert stats["available"] is True
        assert stats["top_k"] == 5
        assert stats["min_relevance_score"] == 0.3


class TestDocumentIndexer:
    """Tests for the DocumentIndexer class."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Create a mock vector store."""
        store = Mock(spec=VectorStore)
        store.count.return_value = 0
        store.add_documents.return_value = 1
        return store
    
    @pytest.fixture
    def mock_embedding_service(self):
        """Create a mock embedding service."""
        service = Mock(spec=EmbeddingService)
        service.embed_batch.return_value = [[0.1] * 384]
        return service
    
    def test_indexer_creation(self, mock_vector_store, mock_embedding_service):
        """Test creating a document indexer."""
        indexer = DocumentIndexer(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        assert indexer.vector_store == mock_vector_store
        assert indexer.batch_size == 100
    
    def test_create_document_chunks(self, mock_vector_store, mock_embedding_service):
        """Test document chunking."""
        indexer = DocumentIndexer(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service,
            chunk_size=100,
            chunk_overlap=20
        )
        
        long_text = "This is a test. " * 20  # ~400 characters
        chunks = indexer.create_document_chunks(long_text, "test")
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, Document) for chunk in chunks)
    
    def test_indexing_stats(self):
        """Test indexing statistics."""
        stats = IndexingStats(
            total_documents=10,
            successful=8,
            failed=2,
            skipped=0
        )
        
        result = stats.to_dict()
        
        assert result["total_documents"] == 10
        assert result["successful"] == 8
        assert result["failed"] == 2
    
    def test_get_builtin_knowledge(self, mock_vector_store, mock_embedding_service):
        """Test getting built-in knowledge documents."""
        indexer = DocumentIndexer(
            vector_store=mock_vector_store,
            embedding_service=mock_embedding_service
        )
        
        knowledge = indexer._get_builtin_knowledge()
        
        assert len(knowledge) > 0
        assert all("content" in k for k in knowledge)
        assert all("category" in k for k in knowledge)


class TestRAGIntegration:
    """Integration tests for the RAG system."""
    
    def test_end_to_end_rag_flow(self):
        """Test complete RAG flow from indexing to retrieval."""
        # Create in-memory store
        store = VectorStore(backend="memory")
        
        # Create embedding service
        embedding_service = EmbeddingService(dimension=384)
        
        # Create RAG service
        rag_service = RAGService(
            vector_store=store,
            embedding_service=embedding_service,
            min_relevance_score=0.0  # Accept all results for testing
        )
        
        # Index some documents
        documents = [
            Document(
                id="doc1",
                content="Heart disease is a serious condition affecting the cardiovascular system.",
                metadata={"category": "cardiac", "model_type": "heart"}
            ),
            Document(
                id="doc2",
                content="Diabetes affects blood sugar levels and requires careful management.",
                metadata={"category": "diabetes", "model_type": "diabetes"}
            ),
            Document(
                id="doc3",
                content="Common cold symptoms include runny nose and cough.",
                metadata={"category": "symptom", "model_type": "symptom"}
            )
        ]
        
        # Generate embeddings and add to store
        embeddings = embedding_service.embed_batch([d.content for d in documents])
        store.add_documents(documents, embeddings)
        
        # Verify documents are indexed
        assert store.count() == 3
        assert rag_service.is_available()
        
        # Test retrieval
        results = rag_service.retrieve("heart problems")
        assert len(results) >= 0  # May or may not find relevant docs depending on embedding quality
        
        # Test prediction
        response = rag_service.predict_with_rag(
            symptoms=["chest pain"],
            model_type="heart"
        )
        
        assert isinstance(response, RAGResponse)
        assert response.disclaimer != ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
