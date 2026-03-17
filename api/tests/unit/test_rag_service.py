"""
Unit Tests for RAGService Component

Test Cases: TC-037, TC-038
Tests RAG service availability and retrieval.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from unittest.mock import Mock
from services.rag_service import RAGService
from services.vector_store import VectorStore
from services.embedding_service import EmbeddingService


def test_tc037_retrieve_documents():
    """TC-037: Retrieve documents from RAG service"""
    print("\n=== TC-037: Retrieve Documents ===")
    
    # Create mock vector store
    mock_store = Mock(spec=VectorStore)
    mock_store.count.return_value = 5
    mock_store.similarity_search.return_value = []
    
    # Create mock embedding service
    mock_embedding = Mock(spec=EmbeddingService)
    mock_embedding.embed.return_value = [0.1] * 384
    
    service = RAGService(
        vector_store=mock_store,
        embedding_service=mock_embedding,
        min_relevance_score=0.3
    )
    
    results = service.retrieve("heart symptoms")
    
    # Assertions
    assert service.is_available(), "RAG service should be available"
    mock_embedding.embed.assert_called_once()
    mock_store.similarity_search.assert_called_once()
    
    print(f"Input: \"heart symptoms\"")
    print(f"Result: Service available, retrieval called")
    print("[PASS] TC-037 passed")
    return True


def test_tc038_empty_store():
    """TC-038: RAG service unavailable with empty store"""
    print("\n=== TC-038: Empty Store Handling ===")
    
    # Create empty mock vector store
    mock_store = Mock(spec=VectorStore)
    mock_store.count.return_value = 0
    
    # Create mock embedding service
    mock_embedding = Mock(spec=EmbeddingService)
    
    service = RAGService(
        vector_store=mock_store,
        embedding_service=mock_embedding
    )
    
    # Assertions
    assert not service.is_available(), "RAG service should not be available with empty store"
    
    print(f"Input: Empty vector store")
    print(f"Result: is_available() = False")
    print("[PASS] TC-038 passed")
    return True


def run_all_tests():
    """Run all RAGService tests."""
    print("=" * 60)
    print("RAG SERVICE UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-037", test_tc037_retrieve_documents),
        ("TC-038", test_tc038_empty_store),
    ]
    
    for test_id, test_func in tests:
        try:
            results.append((test_id, test_func()))
        except AssertionError as e:
            print(f"[FAIL] {test_id}: {e}")
            results.append((test_id, False))
        except Exception as e:
            print(f"[ERROR] {test_id}: {e}")
            results.append((test_id, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_id, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {test_id}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
