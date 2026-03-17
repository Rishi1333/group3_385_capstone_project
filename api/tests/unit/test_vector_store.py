"""
Unit Tests for VectorStore Component

Test Cases: TC-033, TC-034
Tests document storage and retrieval.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.vector_store import VectorStore, Document


def test_tc033_add_documents():
    """TC-033: Add documents to vector store"""
    print("\n=== TC-033: Add Documents ===")
    
    store = VectorStore(backend="memory")
    
    documents = [
        Document(id="1", content="Test 1", metadata={}),
        Document(id="2", content="Test 2", metadata={})
    ]
    embeddings = [[0.1] * 384, [0.2] * 384]
    
    count = store.add_documents(documents, embeddings)
    
    # Assertions
    assert count == 2, f"Expected count 2, got {count}"
    assert store.count() == 2, f"Expected store count 2, got {store.count()}"
    
    print(f"Input: 2 documents with embeddings")
    print(f"Result: Added {count} documents, store count: {store.count()}")
    print("[PASS] TC-033 passed")
    return True


def test_tc034_similarity_search():
    """TC-034: Similarity search returns ranked results"""
    print("\n=== TC-034: Similarity Search ===")
    
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
    
    # Assertions
    assert len(results) <= 2, f"Expected at most 2 results, got {len(results)}"
    if results:
        assert results[0].id == "1", f"Expected first result to be '1', got {results[0].id}"
    
    print(f"Input: Query embedding similar to heart document")
    print(f"Result: Found {len(results)} results, first result ID: {results[0].id if results else 'N/A'}")
    print("[PASS] TC-034 passed")
    return True


def run_all_tests():
    """Run all VectorStore tests."""
    print("=" * 60)
    print("VECTOR STORE UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-033", test_tc033_add_documents),
        ("TC-034", test_tc034_similarity_search),
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
