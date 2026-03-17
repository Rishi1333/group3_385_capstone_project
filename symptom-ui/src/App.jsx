import { useState, useEffect, useRef, useCallback } from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import LandingPage from "./components/LandingPage";
import PatientDashboard from "./components/PatientDashboard";
import VoiceButton from "./components/VoiceButton";
import VoiceSettings from "./components/VoiceSettings";
import BookingPage from "./components/BookingPage";
import {
  getTextToSpeech,
  initializeTTS,
  isSpeechRecognitionSupported,
} from "./services/speechService";
import { getVoiceSettings } from "./services/voiceStorage";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:3000";
const AUTH_STORAGE_KEY = "virtual_clinic_auth";

// Triage state constants
const TriageState = {
  IDLE: "IDLE",
  GATHERING: "GATHERING",
  ANALYZING: "ANALYZING",
  REPORTING: "REPORTING",
  COMPLETE: "COMPLETE",
  EMERGENCY: "EMERGENCY",
};

// Intent badges configuration
const INTENT_CONFIG = {
  EMERGENCY: { label: "Emergency", color: "#ff4757", icon: "âš ï¸" },
  SYMPTOM_TRIAGE: { label: "Symptom Triage", color: "#00d4aa", icon: "ðŸ©º" },
  IMAGE_ANALYSIS: { label: "Image Analysis", color: "#3498db", icon: "ðŸ“·" },
  GENERAL_QUERY: { label: "General Query", color: "#9b59b6", icon: "ðŸ’¬" },
};

// ============================================
// Icon Components
// ============================================

function SendIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="22" y1="2" x2="11" y2="13"></line>
      <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
    </svg>
  );
}

function ImageIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
      <circle cx="8.5" cy="8.5" r="1.5"></circle>
      <polyline points="21 15 16 10 5 21"></polyline>
    </svg>
  );
}

function MicIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
      <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
      <line x1="12" y1="19" x2="12" y2="23"></line>
      <line x1="8" y1="23" x2="16" y2="23"></line>
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="3"></circle>
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
      <polyline points="16 17 21 12 16 7"></polyline>
      <line x1="21" y1="12" x2="9" y2="12"></line>
    </svg>
  );
}

function DatabaseIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
      <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
      <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
    </svg>
  );
}

function SpinnerIcon({ size = 20 }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      className="spinner"
    >
      <circle cx="12" cy="12" r="10" strokeOpacity="0.25"></circle>
      <path d="M12 2a10 10 0 0 1 10 10" strokeOpacity="1"></path>
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
  );
}

// ============================================
// Message Bubble Component
// ============================================

function MessageBubble({ message, isLatest, onBookAppointment }) {
  const isBot = message.role === "bot";
  const isEmergency = message.type === "emergency";
  const isReport = message.type === "report";

  return (
    <div
      className={`message-container ${isBot ? "bot" : "user"} ${isEmergency ? "emergency" : ""} ${isLatest ? "latest" : ""}`}
      style={{
        display: "flex",
        justifyContent: isBot ? "flex-start" : "flex-end",
        marginBottom: "16px",
        animation: isLatest ? "slideUp 0.3s ease-out" : "none",
      }}
    >
      <div
        className={`message-bubble ${isReport ? "report-bubble" : ""}`}
        style={{
          maxWidth: isReport ? "100%" : "85%",
          padding: isReport ? "0" : "16px 20px",
          borderRadius: isBot ? "4px 20px 20px 20px" : "20px 4px 20px 20px",
          background: isEmergency
            ? "linear-gradient(135deg, rgba(255,71,87,0.15) 0%, rgba(255,71,87,0.05) 100%)"
            : isBot
              ? "var(--bg-tertiary)"
              : "var(--accent-primary)",
          color: isBot ? "var(--text-primary)" : "var(--bg-primary)",
          border: isEmergency
            ? "1px solid rgba(255,71,87,0.3)"
            : "1px solid var(--border-subtle)",
          boxShadow: isEmergency
            ? "0 0 30px rgba(255,71,87,0.2)"
            : "var(--shadow-sm)",
        }}
      >
        {isBot && (
          <div
            style={{
              fontSize: "11px",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              opacity: 0.6,
              marginBottom: "8px",
              fontWeight: 500,
            }}
          >
            {isEmergency ? "âš ï¸ Emergency Alert" : "Medical Assistant"}
          </div>
        )}

        {isReport ? (
          <div className="clinical-report">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                h1: ({ children }) => (
                  <h1 className="report-title">{children}</h1>
                ),
                h2: ({ children }) => (
                  <h2 className="report-section">{children}</h2>
                ),
                h3: ({ children }) => (
                  <h3 className="report-subsection">{children}</h3>
                ),
                table: ({ children }) => (
                  <table className="report-table">{children}</table>
                ),
                thead: ({ children }) => (
                  <thead className="report-thead">{children}</thead>
                ),
                th: ({ children }) => <th className="report-th">{children}</th>,
                td: ({ children }) => <td className="report-td">{children}</td>,
                tr: ({ children }) => <tr className="report-tr">{children}</tr>,
                blockquote: ({ children }) => (
                  <blockquote className="report-quote">{children}</blockquote>
                ),
                hr: () => <hr className="report-divider" />,
                ul: ({ children }) => (
                  <ul className="report-list">{children}</ul>
                ),
                ol: ({ children }) => (
                  <ol className="report-ordered-list">{children}</ol>
                ),
                li: ({ children }) => (
                  <li className="report-list-item">{children}</li>
                ),
                p: ({ children }) => (
                  <p className="report-paragraph">{children}</p>
                ),
                strong: ({ children }) => (
                  <strong className="report-strong">{children}</strong>
                ),
                em: ({ children }) => (
                  <em className="report-emphasis">{children}</em>
                ),
                code: ({ children }) => (
                  <code className="report-code">{children}</code>
                ),
              }}
            >
              {message.text}
            </ReactMarkdown>

            {/* Book Appointment Button after report */}
            {isReport && message.conditions && onBookAppointment && (
              <div
                className="booking-cta"
                style={{
                  marginTop: "1.5rem",
                  padding: "1rem",
                  background:
                    "linear-gradient(135deg, rgba(0, 212, 170, 0.1) 0%, rgba(0, 212, 170, 0.05) 100%)",
                  borderRadius: "12px",
                  border: "1px solid rgba(0, 212, 170, 0.3)",
                }}
              >
                <p
                  style={{ margin: "0 0 1rem 0", color: "var(--text-primary)" }}
                >
                  Based on your diagnosis, would you like to book an appointment
                  with a specialist?
                </p>
                <button
                  onClick={() =>
                    onBookAppointment(message.conditions, message.reportId)
                  }
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.5rem",
                    padding: "0.75rem 1.5rem",
                    background:
                      "linear-gradient(135deg, #00d4aa 0%, #00b894 100%)",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    fontSize: "1rem",
                    fontWeight: "500",
                    cursor: "pointer",
                    transition: "all 0.2s ease",
                  }}
                >
                  <svg
                    width="20"
                    height="20"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                  >
                    <rect
                      x="3"
                      y="4"
                      width="18"
                      height="18"
                      rx="2"
                      ry="2"
                    ></rect>
                    <line x1="16" y1="2" x2="16" y2="6"></line>
                    <line x1="8" y1="2" x2="8" y2="6"></line>
                    <line x1="3" y1="10" x2="21" y2="10"></line>
                  </svg>
                  Book Appointment
                </button>
              </div>
            )}
          </div>
        ) : (
          <div style={{ lineHeight: 1.6 }}>
            {isBot ? (
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  p: ({ children }) => (
                    <p style={{ margin: "8px 0" }}>{children}</p>
                  ),
                  ul: ({ children }) => (
                    <ul style={{ margin: "8px 0 8px 20px" }}>{children}</ul>
                  ),
                  ol: ({ children }) => (
                    <ol style={{ margin: "8px 0 8px 20px" }}>{children}</ol>
                  ),
                  strong: ({ children }) => (
                    <strong
                      style={{
                        color: isBot ? "var(--accent-primary)" : "inherit",
                      }}
                    >
                      {children}
                    </strong>
                  ),
                }}
              >
                {message.text}
              </ReactMarkdown>
            ) : (
              <span style={{ whiteSpace: "pre-wrap" }}>{message.text}</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================
// State Indicator Component
// ============================================

function StateIndicator({ currentState, intent }) {
  const states = [
    { key: TriageState.IDLE, label: "Ready" },
    { key: TriageState.GATHERING, label: "Gathering" },
    { key: TriageState.ANALYZING, label: "Analyzing" },
    { key: TriageState.REPORTING, label: "Report" },
  ];

  const currentIndex = states.findIndex((s) => s.key === currentState);
  const intentConfig = INTENT_CONFIG[intent] || INTENT_CONFIG.SYMPTOM_TRIAGE;

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "16px",
        padding: "12px 16px",
        background: "var(--bg-tertiary)",
        borderRadius: "var(--radius-lg)",
        border: "1px solid var(--border-subtle)",
      }}
    >
      {/* State Progress */}
      <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        {states.map((state, index) => (
          <div
            key={state.key}
            style={{ display: "flex", alignItems: "center", gap: "8px" }}
          >
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background:
                  index <= currentIndex
                    ? "var(--accent-primary)"
                    : "var(--border-default)",
                transition: "background var(--transition-fast)",
              }}
            />
            {index < states.length - 1 && (
              <div
                style={{
                  width: "24px",
                  height: "2px",
                  background:
                    index < currentIndex
                      ? "var(--accent-primary)"
                      : "var(--border-default)",
                  transition: "background var(--transition-fast)",
                }}
              />
            )}
          </div>
        ))}
      </div>

      {/* Current State Label */}
      <div
        style={{
          fontSize: "13px",
          fontWeight: 500,
          color: "var(--text-secondary)",
        }}
      >
        {states[currentIndex]?.label || "Ready"}
      </div>

      {/* Intent Badge */}
      {intent && currentState !== TriageState.IDLE && (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "4px 12px",
            background: `${intentConfig.color}15`,
            border: `1px solid ${intentConfig.color}40`,
            borderRadius: "var(--radius-md)",
            fontSize: "12px",
            fontWeight: 500,
            color: intentConfig.color,
          }}
        >
          <span>{intentConfig.icon}</span>
          <span>{intentConfig.label}</span>
        </div>
      )}
    </div>
  );
}

// ============================================
// Image Upload Component
// ============================================

function ImageUploader({ onImageSelect, disabled, preview }) {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      onImageSelect(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      onImageSelect(file);
    }
  };

  return (
    <div
      onClick={() => !disabled && fileInputRef.current?.click()}
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: "44px",
        height: "44px",
        borderRadius: "var(--radius-md)",
        border: "1px solid var(--border-default)",
        background: disabled ? "var(--bg-tertiary)" : "transparent",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "all var(--transition-fast)",
        opacity: disabled ? 0.5 : 1,
      }}
      title="Upload medical image"
    >
      {preview ? (
        <img
          src={preview}
          alt="Preview"
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            borderRadius: "var(--radius-md)",
          }}
        />
      ) : (
        <ImageIcon />
      )}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        style={{ display: "none" }}
        disabled={disabled}
      />
    </div>
  );
}

// ============================================
// Main App Component
// ============================================

function App() {
  // Auth state
  const [auth, setAuth] = useState(() => {
    try {
      const raw = localStorage.getItem(AUTH_STORAGE_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch {
      return null;
    }
  });
  const [authMode, setAuthMode] = useState("login");
  const [authBusy, setAuthBusy] = useState(false);
  const [authError, setAuthError] = useState("");
  const [authForm, setAuthForm] = useState({
    role: "patients",
    gender: "male",
    full_name: "",
    email: "",
    password: "",
  });

  // Triage state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [triageState, setTriageState] = useState(TriageState.IDLE);
  const [sessionId, setSessionId] = useState(null);
  const [intent, setIntent] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);

  // UI state
  const [voiceSettingsOpen, setVoiceSettingsOpen] = useState(false);
  const [voiceSettings, setVoiceSettings] = useState(() => getVoiceSettings());
  const [dbModalOpen, setDbModalOpen] = useState(false);
  const [dbBusy, setDbBusy] = useState(false);
  const [dbError, setDbError] = useState("");
  const [dbData, setDbData] = useState(null);

  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const ttsRef = useRef(null);

  const isAuthenticated = Boolean(auth?.token);
  const isPatientRole = auth?.role === "patients";

  // Initialize TTS
  useEffect(() => {
    initializeTTS().then(() => {
      ttsRef.current = getTextToSpeech({
        voice: voiceSettings.voice,
        speed: voiceSettings.speed,
      });
    });
  }, [voiceSettings.speed, voiceSettings.voice]);

  // Auto-scroll messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  // Helper functions
  const pushMessage = useCallback(
    (role, text, type = null, conditions = null, reportId = null) => {
      setMessages((prev) => [
        ...prev,
        { role, text, type, timestamp: Date.now(), conditions, reportId },
      ]);

      // Auto-speak bot messages
      if (role === "bot" && voiceSettings.autoSpeak && ttsRef.current) {
        const plainText = text
          .replace(/\*\*/g, "")
          .replace(/\*/g, "")
          .replace(/#{1,6}\s/g, "");
        setTimeout(() => ttsRef.current?.speak(plainText), 100);
      }
    },
    [voiceSettings],
  );

  const resetTriage = useCallback(() => {
    setMessages([]);
    setInput("");
    setBusy(false);
    setTriageState(TriageState.IDLE);
    setSessionId(null);
    setIntent(null);
    setImagePreview(null);
    setSelectedImage(null);
  }, []);

  const focusChatInput = useCallback(() => {
    inputRef.current?.focus();
    inputRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, []);

  const saveAuth = useCallback((payload, selectedGender = null) => {
    const next = {
      token: payload.token || payload.access_token,
      role: payload.role,
      email: payload.email || payload.user?.email,
      gender:
        payload.gender ||
        payload.user?.gender ||
        payload.profile?.gender ||
        selectedGender ||
        "male",
    };
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(next));
    setAuth(next);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setAuth(null);
    setDbModalOpen(false);
    setDbData(null);
    setDbError("");
    resetTriage();
  }, [resetTriage]);

  // Start triage session
  const startTriage = useCallback(
    async (text, imageFile = null) => {
      setBusy(true);
      setTriageState(TriageState.GATHERING);

      try {
        const formData = new FormData();
        formData.append("text", text);
        if (imageFile) {
          formData.append("image", imageFile);
        }

        const authHeaders = auth?.token
          ? { Authorization: `Bearer ${auth.token}` }
          : {};

        const response = await fetch(`${API_BASE}/api/triage/start`, {
          method: "POST",
          headers: authHeaders,
          body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data?.error || "Failed to start triage");
        }

        // Handle emergency
        if (data.status === "emergency") {
          setTriageState(TriageState.EMERGENCY);
          pushMessage("bot", data.emergency_response, "emergency");
          return;
        }

        setSessionId(data.session_id);
        setIntent(data.intent);

        // Handle different states
        if (data.state === "GATHERING") {
          pushMessage("bot", data.question);
        } else if (data.state === "COMPLETE" && data.report) {
          setTriageState(TriageState.COMPLETE);
          pushMessage("bot", data.report_display, "report");
        }
      } catch (error) {
        pushMessage("bot", `Error: ${error.message}`);
        setTriageState(TriageState.IDLE);
      } finally {
        setBusy(false);
      }
    },
    [pushMessage, auth?.token],
  );

  // Send message
  const sendMessage = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;

    setInput("");
    pushMessage("user", text);

    if (!sessionId) {
      await startTriage(text, selectedImage);
      return;
    }

    setBusy(true);

    try {
      const authHeaders = auth?.token
        ? { Authorization: `Bearer ${auth.token}` }
        : {};

      const response = await fetch(`${API_BASE}/api/triage/message`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...authHeaders,
        },
        body: JSON.stringify({ session_id: sessionId, message: text }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.error || "Failed to process message");
      }

      if (data.state === "GATHERING") {
        pushMessage("bot", data.question);
      } else if (data.state === "COMPLETE" && data.report) {
        setTriageState(TriageState.COMPLETE);
        pushMessage(
          "bot",
          data.report_display,
          "report",
          data.conditions,
          data.report?.report_id,
        );
      }
    } catch (error) {
      pushMessage("bot", `Error: ${error.message}`);
    } finally {
      setBusy(false);
    }
  }, [
    input,
    busy,
    sessionId,
    selectedImage,
    pushMessage,
    startTriage,
    auth?.token,
  ]);

  // Handle booking navigation
  const handleBookAppointment = useCallback((conditions, reportId) => {
    // Navigate to booking page - this will be handled by the Router
    window.location.href = `/booking?conditions=${encodeURIComponent(JSON.stringify(conditions))}&reportId=${reportId || ""}`;
  }, []);

  // Handle image selection
  const handleImageSelect = useCallback((file) => {
    setSelectedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => {
      setImagePreview(reader.result);
    };
    reader.readAsDataURL(file);
  }, []);

  // Auth submit
  const handleAuthSubmit = useCallback(
    async (e) => {
      e.preventDefault();
      if (authBusy) return;

      const isRegister = authMode === "register";
      const { role, gender, email, password, full_name } = authForm;

      if (
        !role ||
        !gender ||
        !email ||
        !password ||
        (isRegister && !full_name)
      ) {
        setAuthError("Please fill all required fields.");
        return;
      }

      setAuthBusy(true);
      setAuthError("");

      try {
        const endpoint = isRegister ? "/auth/register" : "/auth/login";
        const payload = isRegister
          ? { role, email, password, full_name }
          : { role, email, password };

        const response = await fetch(`${API_BASE}${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const data = await response.json();

        if (!response.ok) {
          throw new Error(data?.error || "Authentication failed");
        }

        saveAuth(data, gender);
        resetTriage();
        setAuthForm((prev) => ({
          ...prev,
          gender: prev.gender || "male",
          full_name: "",
          email: "",
          password: "",
        }));
      } catch (error) {
        setAuthError(error.message);
      } finally {
        setAuthBusy(false);
      }
    },
    [authBusy, authMode, authForm, saveAuth, resetTriage],
  );

  // Fetch database data
  const fetchDatabaseData = useCallback(async () => {
    if (!auth?.token) return;

    setDbBusy(true);
    setDbError("");

    try {
      const endpoint =
        auth.role === "administrator" ? "/auth/admin/users" : "/auth/profile";
      const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: { Authorization: `Bearer ${auth.token}` },
      });

      const data = await response.json();

      if (!response.ok) {
        if (response.status === 401) {
          logout();
          throw new Error("Session expired");
        }
        throw new Error(data?.error || "Failed to load data");
      }

      setDbData(data);
    } catch (error) {
      setDbError(error.message);
      setDbData(null);
    } finally {
      setDbBusy(false);
    }
  }, [auth, logout]);

  // Render auth screen if not authenticated
  if (!isAuthenticated) {
    return (
      <LandingPage
        mode={authMode}
        setMode={setAuthMode}
        form={authForm}
        setForm={setAuthForm}
        busy={authBusy}
        error={authError}
        onSubmit={handleAuthSubmit}
      />
    );
  }

  // Main app render
  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="logo">
            <svg width="32" height="32" viewBox="0 0 48 48" fill="none">
              <circle
                cx="24"
                cy="24"
                r="22"
                stroke="var(--accent-primary)"
                strokeWidth="2"
              />
              <path
                d="M24 14v10l6 6"
                stroke="var(--accent-primary)"
                strokeWidth="2"
                strokeLinecap="round"
              />
              <circle cx="24" cy="24" r="4" fill="var(--accent-primary)" />
            </svg>
          </div>
          <div>
            <h1 className="app-title">AI Virtual Clinic</h1>
            <p className="app-subtitle">
              {auth.email} â€¢ {auth.role}
            </p>
          </div>
        </div>

        <div className="header-right">
          <button
            onClick={() => {
              setDbModalOpen(true);
              fetchDatabaseData();
            }}
            className="header-btn"
          >
            <DatabaseIcon />
            <span>Database</span>
          </button>

          <button
            onClick={() => setVoiceSettingsOpen(true)}
            className="header-btn icon-only"
          >
            <SettingsIcon />
          </button>

          <button onClick={logout} className="header-btn">
            <LogoutIcon />
            <span>Logout</span>
          </button>
        </div>
      </header>

      {/* State Indicator */}
      <div className="state-bar">
        <StateIndicator currentState={triageState} intent={intent} />

        {triageState !== TriageState.IDLE && (
          <button onClick={resetTriage} className="new-session-btn">
            New Session
          </button>
        )}
      </div>

      {/* Messages Area */}
      <div ref={scrollRef} className="messages-area">
        {isPatientRole && (
          <PatientDashboard
            patientEmail={auth.email}
            patientGender={auth.gender}
            onFocusChatInput={focusChatInput}
          />
        )}

        {messages.map((msg, i) => (
          <MessageBubble
            key={i}
            message={msg}
            isLatest={i === messages.length - 1}
            onBookAppointment={handleBookAppointment}
          />
        ))}

        {busy && (
          <div className="typing-indicator">
            <div className="typing-dots">
              <span></span>
              <span></span>
              <span></span>
            </div>
            <span>Analyzing...</span>
          </div>
        )}
      </div>

      {/* Input Area */}
      <div className="input-area">
        <ImageUploader
          onImageSelect={handleImageSelect}
          disabled={busy}
          preview={imagePreview}
        />

        {selectedImage && (
          <button
            onClick={() => {
              setSelectedImage(null);
              setImagePreview(null);
            }}
            className="remove-image-btn"
          >
            <CloseIcon />
          </button>
        )}

        <input
          ref={inputRef}
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !e.shiftKey && sendMessage()}
          placeholder={
            selectedImage
              ? "Add context for the image..."
              : "Describe your symptoms..."
          }
          disabled={busy}
          className="message-input"
        />

        {isSpeechRecognitionSupported() && (
          <VoiceButton
            onResult={setInput}
            onInterim={setInput}
            onError={(err) => console.error("Voice error:", err)}
            disabled={busy}
            size="medium"
          />
        )}

        <button
          onClick={sendMessage}
          disabled={busy || (!input.trim() && !selectedImage)}
          className="send-btn"
        >
          {busy ? <SpinnerIcon /> : <SendIcon />}
        </button>
      </div>
      {/* Voice Settings Modal */}
      <VoiceSettings
        open={voiceSettingsOpen}
        onClose={() => setVoiceSettingsOpen(false)}
        settings={voiceSettings}
        onSettingsChange={setVoiceSettings}
      />

      {/* Database Modal */}
      {dbModalOpen && (
        <div className="modal-overlay" onClick={() => setDbModalOpen(false)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Database Data</h3>
              <div className="modal-actions">
                <button onClick={fetchDatabaseData} disabled={dbBusy}>
                  {dbBusy ? <SpinnerIcon size={16} /> : "Refresh"}
                </button>
                <button onClick={() => setDbModalOpen(false)}>
                  <CloseIcon />
                </button>
              </div>
            </div>
            <div className="modal-body">
              {dbError && <div className="error-message">{dbError}</div>}
              {dbBusy ? (
                <div className="loading-state">
                  <SpinnerIcon size={24} />
                  <span>Loading...</span>
                </div>
              ) : dbData ? (
                <pre className="json-display">
                  {JSON.stringify(dbData, null, 2)}
                </pre>
              ) : (
                <p className="empty-state">No data loaded</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Wrap App with Router for routing support
function AppWithRouter() {
  return (
    <Router>
      <Routes>
        <Route path="/booking" element={<BookingPage />} />
        <Route path="*" element={<App />} />
      </Routes>
    </Router>
  );
}

export { AppWithRouter as default };
