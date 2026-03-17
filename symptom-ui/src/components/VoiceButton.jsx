/**
 * VoiceButton Component
 * 
 * A microphone button for speech-to-text input.
 * Shows visual feedback during recording.
 */

import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  isSpeechRecognitionSupported,
  getSpeechRecognition
} from '../services/speechService';

// Microphone icon SVG
const MicIcon = ({ size = 24, color = 'currentColor' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    stroke={color}
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" />
    <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
    <line x1="12" y1="19" x2="12" y2="23" />
    <line x1="8" y1="23" x2="16" y2="23" />
  </svg>
);

// Stop icon SVG
const StopIcon = ({ size = 24, color = 'currentColor' }) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill={color}
  >
    <rect x="6" y="6" width="12" height="12" rx="2" />
  </svg>
);

/**
 * VoiceButton props
 * @param {function} onResult - Callback when speech is recognized
 * @param {function} onInterim - Callback for interim results
 * @param {function} onError - Callback for errors
 * @param {boolean} disabled - Disable the button
 * @param {string} size - Button size: 'small', 'medium', 'large'
 * @param {object} style - Additional styles
 */
function VoiceButton({
  onResult,
  onInterim,
  onError,
  disabled = false,
  size = 'medium',
  style = {}
}) {
  const [isListening, setIsListening] = useState(false);

  // Check support once
  const isSupported = useMemo(() => isSpeechRecognitionSupported(), []);

  // Get recognition instance (memoized)
  const recognition = useMemo(() => {
    if (isSupported) {
      try {
        return getSpeechRecognition();
      } catch (error) {
        console.error('Failed to initialize speech recognition:', error);
        return null;
      }
    }
    return null;
  }, [isSupported]);

  // Size configurations
  const sizes = {
    small: { button: 36, icon: 18 },
    medium: { button: 48, icon: 24 },
    large: { button: 60, icon: 30 }
  };
  const currentSize = sizes[size] || sizes.medium;

  // Set up recognition callbacks
  useEffect(() => {
    if (!recognition) return;

    const handleResult = (text) => {
      setIsListening(false);
      if (onResult) onResult(text);
    };

    const handleInterim = (text) => {
      if (onInterim) onInterim(text);
    };

    const handleError = (error) => {
      setIsListening(false);
      if (onError) onError(error);
    };

    const handleEnd = () => {
      setIsListening(false);
    };

    recognition.onResult(handleResult);
    recognition.onInterim(handleInterim);
    recognition.onError(handleError);
    recognition.onEnd(handleEnd);
  }, [recognition, onResult, onInterim, onError]);

  // Toggle listening
  const toggleListening = useCallback(() => {
    if (!recognition) return;

    if (isListening) {
      recognition.stop();
    } else {
      recognition.start();
      setIsListening(true);
    }
  }, [recognition, isListening]);

  // Button styles
  const buttonStyle = {
    width: currentSize.button,
    height: currentSize.button,
    borderRadius: '50%',
    border: 'none',
    cursor: disabled || !isSupported ? 'not-allowed' : 'pointer',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: isListening 
      ? 'linear-gradient(135deg, #ff6b6b, #ee5a5a)'
      : 'linear-gradient(135deg, #4a9eff, #3b8bdb)',
    color: 'white',
    boxShadow: isListening
      ? '0 4px 15px rgba(255, 107, 107, 0.4)'
      : '0 4px 15px rgba(74, 158, 255, 0.3)',
    transition: 'all 0.2s ease',
    opacity: disabled || !isSupported ? 0.5 : 1,
    transform: isListening ? 'scale(1.1)' : 'scale(1)',
    ...style
  };

  // Pulse animation for recording state
  const pulseStyle = isListening ? {
    animation: 'pulse 1.5s infinite'
  } : {};

  if (!isSupported) {
    return null; // Don't render if not supported
  }

  return (
    <>
      <style>
        {`
          @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(255, 107, 107, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); }
          }
        `}
      </style>
      <button
        type="button"
        onClick={toggleListening}
        disabled={disabled || !isSupported}
        style={{ ...buttonStyle, ...pulseStyle }}
        title={isListening ? 'Click to stop' : 'Click to speak'}
        aria-label={isListening ? 'Stop listening' : 'Start listening'}
      >
        {isListening ? (
          <StopIcon size={currentSize.icon} />
        ) : (
          <MicIcon size={currentSize.icon} />
        )}
      </button>
    </>
  );
}

export default VoiceButton;
