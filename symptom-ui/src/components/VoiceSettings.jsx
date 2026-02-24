/**
 * VoiceSettings Component
 * 
 * A settings panel for configuring voice preferences:
 * - Voice selection
 * - Speech speed
 * - Auto-speak toggle
 */

import { useState, useEffect } from 'react';
import {
  getVoiceSettings,
  saveVoiceSettings,
  resetVoiceSettings
} from '../services/voiceStorage';
import { initializeTTS } from '../services/speechService';

// Speaker icon
const SpeakerIcon = ({ size = 20, color = 'currentColor' }) => (
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
    <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
    <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
    <path d="M19.07 4.93a10 10 0 0 1 0 14.14" />
  </svg>
);

// Settings icon
const SettingsIcon = ({ size = 20, color = 'currentColor' }) => (
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
    <circle cx="12" cy="12" r="3" />
    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
  </svg>
);

/**
 * VoiceSettings props
 * @param {boolean} isOpen - Whether the panel is open
 * @param {function} onClose - Callback when panel is closed
 * @param {function} onSettingsChange - Callback when settings change
 */
function VoiceSettings({ isOpen, onClose, onSettingsChange }) {
  const [settings, setSettings] = useState(getVoiceSettings());
  const [voices, setVoices] = useState([]);
  const [ttsAvailable, setTtsAvailable] = useState(false);

  // Initialize TTS and load voices
  useEffect(() => {
    async function init() {
      const status = await initializeTTS();
      setTtsAvailable(status.available);
      setVoices(status.voices || []);
    }
    init();
  }, []);

  // Handle setting change
  const handleChange = (key, value) => {
    const newSettings = { ...settings, [key]: value };
    setSettings(newSettings);
    saveVoiceSettings(newSettings);
    if (onSettingsChange) {
      onSettingsChange(newSettings);
    }
  };

  // Handle reset
  const handleReset = () => {
    const defaultSettings = resetVoiceSettings();
    setSettings(defaultSettings);
    if (onSettingsChange) {
      onSettingsChange(defaultSettings);
    }
  };

  // Panel styles
  const panelStyle = {
    position: 'fixed',
    top: 0,
    right: isOpen ? 0 : '-320px',
    width: '320px',
    height: '100vh',
    background: '#1a1a2e',
    boxShadow: '-4px 0 20px rgba(0, 0, 0, 0.3)',
    transition: 'right 0.3s ease',
    zIndex: 1000,
    overflowY: 'auto',
    color: '#fff'
  };

  const overlayStyle = {
    position: 'fixed',
    top: 0,
    left: 0,
    width: '100vw',
    height: '100vh',
    background: 'rgba(0, 0, 0, 0.5)',
    opacity: isOpen ? 1 : 0,
    visibility: isOpen ? 'visible' : 'hidden',
    transition: 'opacity 0.3s ease',
    zIndex: 999
  };

  const sectionStyle = {
    padding: '16px 20px',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)'
  };

  const labelStyle = {
    display: 'block',
    fontSize: '12px',
    fontWeight: 500,
    color: '#888',
    marginBottom: '8px',
    textTransform: 'uppercase',
    letterSpacing: '0.5px'
  };

  const selectStyle = {
    width: '100%',
    padding: '10px 12px',
    borderRadius: '8px',
    border: '1px solid rgba(255, 255, 255, 0.2)',
    background: 'rgba(255, 255, 255, 0.05)',
    color: '#fff',
    fontSize: '14px',
    cursor: 'pointer'
  };

  const sliderStyle = {
    width: '100%',
    height: '4px',
    borderRadius: '2px',
    background: 'rgba(255, 255, 255, 0.2)',
    outline: 'none',
    cursor: 'pointer'
  };

  const toggleStyle = {
    position: 'relative',
    width: '48px',
    height: '26px',
    borderRadius: '13px',
    background: settings.autoSpeak ? '#4a9eff' : 'rgba(255, 255, 255, 0.2)',
    cursor: 'pointer',
    transition: 'background 0.2s ease'
  };

  const toggleKnobStyle = {
    position: 'absolute',
    top: '3px',
    left: settings.autoSpeak ? '25px' : '3px',
    width: '20px',
    height: '20px',
    borderRadius: '50%',
    background: '#fff',
    transition: 'left 0.2s ease'
  };

  if (!isOpen) {
    return null;
  }

  return (
    <>
      {/* Overlay */}
      <div style={overlayStyle} onClick={onClose} />
      
      {/* Panel */}
      <div style={panelStyle}>
        {/* Header */}
        <div style={{
          ...sectionStyle,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <SpeakerIcon />
            <span style={{ fontSize: '18px', fontWeight: 600 }}>Voice Settings</span>
          </div>
          <button
            onClick={onClose}
            style={{
              background: 'none',
              border: 'none',
              color: '#888',
              fontSize: '24px',
              cursor: 'pointer',
              padding: '4px'
            }}
          >
            ×
          </button>
        </div>

        {/* TTS Status */}
        <div style={sectionStyle}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '10px',
            borderRadius: '8px',
            background: ttsAvailable 
              ? 'rgba(74, 222, 128, 0.1)' 
              : 'rgba(255, 107, 107, 0.1)'
          }}>
            <div style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: ttsAvailable ? '#4ade80' : '#ff6b6b'
            }} />
            <span style={{ fontSize: '13px' }}>
              {ttsAvailable 
                ? 'Kokoro TTS Connected' 
                : 'Using Browser TTS (Fallback)'}
            </span>
          </div>
        </div>

        {/* Voice Selection */}
        <div style={sectionStyle}>
          <label style={labelStyle}>Voice</label>
          <select
            value={settings.voice}
            onChange={(e) => handleChange('voice', e.target.value)}
            style={selectStyle}
            disabled={!ttsAvailable}
          >
            {voices.length > 0 ? (
              voices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name} ({v.gender}, {v.locale})
                </option>
              ))
            ) : (
              <option value="af_bella">Bella (Female, en-US)</option>
            )}
          </select>
        </div>

        {/* Speech Speed */}
        <div style={sectionStyle}>
          <label style={labelStyle}>
            Speech Speed: {settings.speed.toFixed(1)}x
          </label>
          <input
            type="range"
            min="0.5"
            max="1.5"
            step="0.1"
            value={settings.speed}
            onChange={(e) => handleChange('speed', parseFloat(e.target.value))}
            style={sliderStyle}
          />
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '11px',
            color: '#666',
            marginTop: '4px'
          }}>
            <span>Slower</span>
            <span>Faster</span>
          </div>
        </div>

        {/* Auto-speak Toggle */}
        <div style={sectionStyle}>
          <label style={labelStyle}>Auto-speak Responses</label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div
              style={toggleStyle}
              onClick={() => handleChange('autoSpeak', !settings.autoSpeak)}
            >
              <div style={toggleKnobStyle} />
            </div>
            <span style={{ fontSize: '14px', color: '#aaa' }}>
              {settings.autoSpeak ? 'Enabled' : 'Disabled'}
            </span>
          </div>
          <p style={{
            fontSize: '12px',
            color: '#666',
            marginTop: '8px'
          }}>
            When enabled, the assistant will automatically speak its responses.
          </p>
        </div>

        {/* Reset Button */}
        <div style={sectionStyle}>
          <button
            onClick={handleReset}
            style={{
              width: '100%',
              padding: '10px',
              borderRadius: '8px',
              border: '1px solid rgba(255, 255, 255, 0.2)',
              background: 'transparent',
              color: '#888',
              fontSize: '14px',
              cursor: 'pointer'
            }}
          >
            Reset to Defaults
          </button>
        </div>
      </div>
    </>
  );
}

export default VoiceSettings;
