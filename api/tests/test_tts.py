"""
Test script for TTS (Text-to-Speech) service using Kokoro-82M.
This script tests the TTS service functionality.
"""

import sys
import os
import tempfile
import wave
import io

# Set UTF-8 encoding for stdout to handle unicode characters
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except:
        pass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.tts_service import TTSService


def test_tts_service_initialization():
    """Test that TTS service initializes correctly."""
    print("=" * 50)
    print("Test 1: TTS Service Initialization")
    print("=" * 50)
    
    try:
        tts = TTSService()
        print(f"[OK] TTS Service initialized successfully")
        print(f"  - Available: {tts.available}")
        return True, tts
    except Exception as e:
        print(f"[FAIL] Failed to initialize TTS Service: {e}")
        return False, None


def test_get_voices(tts):
    """Test getting available voices."""
    print("\n" + "=" * 50)
    print("Test 2: Get Available Voices")
    print("=" * 50)
    
    try:
        voices = tts.get_available_voices()
        print(f"[OK] Retrieved {len(voices)} voices:")
        for voice_id, voice_info in voices.items():
            print(f"  - {voice_id}: {voice_info['name']} ({voice_info['gender']}, {voice_info['locale']})")
        return True, voices
    except Exception as e:
        print(f"[FAIL] Failed to get voices: {e}")
        return False, None


def test_generate_speech(tts):
    """Test generating speech from text."""
    print("\n" + "=" * 50)
    print("Test 3: Generate Speech")
    print("=" * 50)
    
    test_text = "Hello, this is a test of the AI Virtual Clinic text to speech system."
    
    try:
        # Test with default voice
        print(f"  Generating speech for: '{test_text[:50]}...'")
        wav_buffer = tts.synthesize_to_wav(test_text)
        
        if wav_buffer:
            audio_data = wav_buffer.read()
            print(f"[OK] Speech generated successfully")
            print(f"  - Audio data size: {len(audio_data)} bytes")
            
            # Save to temp file and verify it's a valid WAV
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                f.write(audio_data)
                temp_path = f.name
            
            # Verify WAV file
            with wave.open(temp_path, 'rb') as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                frames = wav_file.getnframes()
                duration = frames / framerate
                
                print(f"  - Channels: {channels}")
                print(f"  - Sample width: {sample_width} bytes")
                print(f"  - Frame rate: {framerate} Hz")
                print(f"  - Duration: {duration:.2f} seconds")
            
            # Cleanup
            os.unlink(temp_path)
            return True, audio_data
        else:
            print(f"[FAIL] No audio data generated")
            return False, None
            
    except Exception as e:
        print(f"[FAIL] Failed to generate speech: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_different_voices(tts, voices):
    """Test generating speech with different voices."""
    print("\n" + "=" * 50)
    print("Test 4: Test Different Voices")
    print("=" * 50)
    
    test_text = "Testing voice selection."
    results = {}
    
    # Test first 3 voices only to save time
    for voice_id in list(voices.keys())[:3]:
        try:
            print(f"  Testing voice: {voice_id}...")
            wav_buffer = tts.synthesize_to_wav(test_text, voice=voice_id)
            
            if wav_buffer:
                audio_data = wav_buffer.read()
                results[voice_id] = len(audio_data)
                print(f"    [OK] Generated {len(audio_data)} bytes")
            else:
                results[voice_id] = 0
                print(f"    [FAIL] No audio data")
        except Exception as e:
            results[voice_id] = str(e)
            print(f"    [FAIL] Error: {e}")
    
    success = all(isinstance(v, int) and v > 0 for v in results.values())
    if success:
        print(f"\n[OK] All voices tested successfully")
    else:
        print(f"\n[FAIL] Some voices failed")
    
    return success, results


def test_speed_variation(tts):
    """Test speech generation with different speeds."""
    print("\n" + "=" * 50)
    print("Test 5: Speed Variation")
    print("=" * 50)
    
    test_text = "Testing speech speed."
    speeds = [0.8, 1.0, 1.2]
    results = {}
    
    for speed in speeds:
        try:
            print(f"  Testing speed: {speed}...")
            wav_buffer = tts.synthesize_to_wav(test_text, speed=speed)
            
            if wav_buffer:
                audio_data = wav_buffer.read()
                results[speed] = len(audio_data)
                print(f"    [OK] Generated {len(audio_data)} bytes")
            else:
                results[speed] = 0
                print(f"    [FAIL] No audio data")
        except Exception as e:
            results[speed] = str(e)
            print(f"    [FAIL] Error: {e}")
    
    success = all(isinstance(v, int) and v > 0 for v in results.values())
    if success:
        print(f"\n[OK] Speed variation works correctly")
    else:
        print(f"\n[FAIL] Speed variation has issues")
    
    return success, results


def test_long_text(tts):
    """Test generating speech for longer text."""
    print("\n" + "=" * 50)
    print("Test 6: Long Text Generation")
    print("=" * 50)
    
    long_text = """
    Welcome to the AI Virtual Clinic. This is a comprehensive healthcare 
    assistant that can help you with symptom analysis, disease prediction, 
    and health recommendations. Please describe your symptoms and we will 
    do our best to assist you. Remember, this is not a replacement for 
    professional medical advice.
    """
    
    try:
        print(f"  Generating speech for long text ({len(long_text)} chars)...")
        wav_buffer = tts.synthesize_to_wav(long_text)
        
        if wav_buffer:
            audio_data = wav_buffer.read()
            print(f"[OK] Long text speech generated successfully")
            print(f"  - Audio data size: {len(audio_data)} bytes")
            return True, len(audio_data)
        else:
            print(f"[FAIL] No audio data generated")
            return False, None
    except Exception as e:
        print(f"[FAIL] Failed to generate long text speech: {e}")
        return False, str(e)


def main():
    """Run all TTS tests."""
    print("\n" + "=" * 60)
    print("  AI Virtual Clinic - TTS Service Test Suite")
    print("  Using Kokoro-82M Text-to-Speech")
    print("=" * 60)
    
    results = []
    
    # Test 1: Initialization
    success, tts = test_tts_service_initialization()
    results.append(("Initialization", success))
    
    if not success:
        print("\n" + "!" * 60)
        print("  Cannot continue tests - TTS Service not available")
        print("!" * 60)
        return
    
    # Test 2: Get Voices
    success, voices = test_get_voices(tts)
    results.append(("Get Voices", success))
    
    # Test 3: Generate Speech
    success, _ = test_generate_speech(tts)
    results.append(("Generate Speech", success))
    
    # Test 4: Different Voices
    if voices:
        success, _ = test_different_voices(tts, voices)
        results.append(("Different Voices", success))
    
    # Test 5: Speed Variation
    success, _ = test_speed_variation(tts)
    results.append(("Speed Variation", success))
    
    # Test 6: Long Text
    success, _ = test_long_text(tts)
    results.append(("Long Text", success))
    
    # Summary
    print("\n" + "=" * 60)
    print("  Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {name}: {status}")
    
    print("-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("\n  SUCCESS: All TTS tests passed!")
    else:
        print(f"\n  WARNING: {total - passed} test(s) failed")


if __name__ == "__main__":
    main()
