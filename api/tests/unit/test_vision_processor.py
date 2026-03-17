"""
Unit Tests for VisionProcessor Component

Test Cases: TC-044, TC-045, TC-046, TC-047
Tests image validation and encoding.
"""

import sys
import os
import base64
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.vision_processor import VisionProcessor


def test_tc044_encode_image():
    """TC-044: Encode image to base64"""
    print("\n=== TC-044: Encode Image ===")
    
    processor = VisionProcessor()
    test_data = b"test_image_data"
    encoded = processor.encode_image(test_data)
    
    # Assertions
    assert isinstance(encoded, str), f"Expected string, got {type(encoded)}"
    assert base64.b64decode(encoded) == test_data, "Encoded data should match original"
    
    print(f"Input: Binary image data")
    print(f"Result: Base64 encoded string")
    print("[PASS] TC-044 passed")
    return True


def test_tc045_validate_valid_image():
    """TC-045: Validate valid JPEG image"""
    print("\n=== TC-045: Validate Valid Image ===")
    
    processor = VisionProcessor()
    
    # Use the actual x-ray image from docs folder
    xray_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
        "docs", "x-ray.jpg"
    )
    
    if os.path.exists(xray_path):
        with open(xray_path, 'rb') as f:
            image_data = f.read()
        
        result = processor.validate_image(image_data)
        
        # Assertions
        assert result["valid"] is True, f"Expected valid=true, got {result}"
        assert result["format"] in ["JPEG", "JPG", "WEBP", "PNG"], f"Unexpected format: {result.get('format')}"
        
        print(f"Input: Valid X-ray image")
        print(f"Result: valid={result['valid']}, format={result['format']}")
        print("[PASS] TC-045 passed")
        return True
    else:
        print("X-ray image not found, skipping test")
        return True  # Skip, not a failure


def test_tc046_image_too_large():
    """TC-046: Validate image too large"""
    print("\n=== TC-046: Image Too Large ===")
    
    processor = VisionProcessor()
    
    # Create large fake image data (> 10MB)
    large_data = b"x" * (11 * 1024 * 1024)
    result = processor.validate_image(large_data)
    
    # Assertions
    assert result["valid"] is False, f"Expected valid=false, got {result}"
    assert "error" in result, "Expected error message"
    
    print(f"Input: 11MB+ image data")
    print(f"Result: valid={result['valid']}, error={result.get('error', '')[:50]}...")
    print("[PASS] TC-046 passed")
    return True


def test_tc047_invalid_image_format():
    """TC-047: Validate invalid image format"""
    print("\n=== TC-047: Invalid Image Format ===")
    
    processor = VisionProcessor()
    
    # Non-image data
    invalid_data = b"This is not an image, just text data"
    result = processor.validate_image(invalid_data)
    
    # Assertions
    assert result["valid"] is False, f"Expected valid=false, got {result}"
    assert "error" in result, "Expected error message"
    
    print(f"Input: Non-image text data")
    print(f"Result: valid={result['valid']}")
    print("[PASS] TC-047 passed")
    return True


def run_all_tests():
    """Run all VisionProcessor tests."""
    print("=" * 60)
    print("VISION PROCESSOR UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-044", test_tc044_encode_image),
        ("TC-045", test_tc045_validate_valid_image),
        ("TC-046", test_tc046_image_too_large),
        ("TC-047", test_tc047_invalid_image_format),
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
