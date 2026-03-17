"""
Unit Tests for EmbeddingService Component

Test Cases: TC-035, TC-036
Tests embedding generation and similarity calculation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.embedding_service import EmbeddingService


def test_tc035_generate_embedding():
    """TC-035: Generate 384-dimensional embedding"""
    print("\n=== TC-035: Generate Embedding ===")
    
    service = EmbeddingService(dimension=384)
    embedding = service.embed("test text")
    
    # Assertions
    assert len(embedding) == 384, f"Expected 384 dimensions, got {len(embedding)}"
    assert isinstance(embedding, list), f"Expected list, got {type(embedding)}"
    assert all(isinstance(x, float) for x in embedding), "Expected all floats in embedding"
    
    print(f"Input: \"test text\"")
    print(f"Result: {len(embedding)}-dimensional vector")
    print("[PASS] TC-035 passed")
    return True


def test_tc036_identical_vectors():
    """TC-036: Similarity of identical vectors is 1.0"""
    print("\n=== TC-036: Identical Vectors Similarity ===")
    
    service = EmbeddingService(dimension=384)
    
    emb1 = [1.0, 0.0, 0.0]
    emb2 = [1.0, 0.0, 0.0]
    sim = service.similarity(emb1, emb2)
    
    # Assertions
    assert abs(sim - 1.0) < 0.001, f"Expected similarity ~1.0, got {sim}"
    
    print(f"Input: [1,0,0] and [1,0,0]")
    print(f"Result: similarity={sim}")
    print("[PASS] TC-036 passed")
    return True


def run_all_tests():
    """Run all EmbeddingService tests."""
    print("=" * 60)
    print("EMBEDDING SERVICE UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-035", test_tc035_generate_embedding),
        ("TC-036", test_tc036_identical_vectors),
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
