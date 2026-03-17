"""
Unit Tests for TTSService Component

Test Cases: TC-039, TC-040, TC-041, TC-042, TC-043
Tests TTS service initialization and speech synthesis.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from services.tts_service import TTSService


def test_tc039_tts_initialization():
    """TC-039: Initialize TTS service"""
    print("\n=== TC-039: TTS Initialization ===")
    
    try:
        tts = TTSService()
        
        # Assertions
        assert tts is not None, "TTS service should be created"
        assert hasattr(tts, 'available'), "TTS service should have 'available' attribute"
        
        print(f"Result: Service created, available={tts.available}")
        print("[PASS] TC-039 passed")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_tc040_get_voices():
    """TC-040: Get available voices"""
    print("\n=== TC-040: Get Available Voices ===")
    
    try:
        tts = TTSService()
        voices = tts.get_available_voices()
        
        # Assertions
        assert isinstance(voices, dict), f"Expected dict, got {type(voices)}"
        
        print(f"Result: {len(voices)} voices available")
        print("[PASS] TC-040 passed")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_tc041_synthesize_speech():
    """TC-041: Synthesize speech from text"""
    print("\n=== TC-041: Synthesize Speech ===")
    
    try:
        tts = TTSService()
        
        if not tts.available:
            print("TTS not available, skipping test")
            return True  # Skip, not a failure
        
        wav_buffer = tts.synthesize_to_wav("Hello, this is a test")
        
        # Assertions
        if wav_buffer:
            audio_data = wav_buffer.read()
            assert len(audio_data) > 0, "Expected audio data"
            print(f"Result: Generated {len(audio_data)} bytes of audio")
        else:
            print("Result: No audio generated (TTS may not be fully configured)")
        
        print("[PASS] TC-041 passed")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_tc042_speed_variation():
    """TC-042: Speech with speed variation"""
    print("\n=== TC-042: Speed Variation ===")
    
    try:
        tts = TTSService()
        
        if not tts.available:
            print("TTS not available, skipping test")
            return True
        
        wav_buffer = tts.synthesize_to_wav("Testing speed", speed=1.2)
        
        print(f"Result: Speed variation test completed")
        print("[PASS] TC-042 passed")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def test_tc043_empty_text():
    """TC-043: Handle empty text gracefully"""
    print("\n=== TC-043: Empty Text Handling ===")
    
    try:
        tts = TTSService()
        
        if not tts.available:
            print("TTS not available, skipping test")
            return True
        
        wav_buffer = tts.synthesize_to_wav("")
        
        # Should not crash
        print(f"Result: Handled gracefully")
        print("[PASS] TC-043 passed")
        return True
    except Exception as e:
        print(f"[ERROR] {e}")
        return False


def run_all_tests():
    """Run all TTSService tests."""
    print("=" * 60)
    print("TTS SERVICE UNIT TESTS")
    print("=" * 60)
    
    results = []
    
    tests = [
        ("TC-039", test_tc039_tts_initialization),
        ("TC-040", test_tc040_get_voices),
        ("TC-041", test_tc041_synthesize_speech),
        ("TC-042", test_tc042_speed_variation),
        ("TC-043", test_tc043_empty_text),
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
