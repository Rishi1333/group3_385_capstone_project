import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import "./App.css";
import VoiceButton from "./components/VoiceButton";
import VoiceSettings from "./components/VoiceSettings";
import {
  getTextToSpeech,
  initializeTTS,
  isSpeechRecognitionSupported
} from "./services/speechService";
import { getVoiceSettings } from "./services/voiceStorage";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:3000";
const AUTH_STORAGE_KEY = "virtual_clinic_auth";
const INITIAL_MESSAGE =
  "Hi! Tell me what you are feeling. I can help with general symptoms, heart concerns, diabetes screening, or mental health questions.";

const MODEL_CONFIG = {
  symptom: {
    label: "Symptom Analysis",
    badgeColor: "#64c864",
    description: "general symptom-based disease prediction",
  },
  heart: {
    label: "Heart Health",
    badgeColor: "#ff6464",
    description: "heart disease risk assessment",
  },
  diabetes: {
    label: "Diabetes Screening",
    badgeColor: "#6495ed",
    description: "diabetes risk prediction",
  },
  mental_health: {
    label: "Mental Health",
    badgeColor: "#9b59b6",
    description: "mental health screening",
  },
};

function Bubble({ role, text }) {
  const isBot = role === "bot";

  return (
    <div
      style={{
        display: "flex",
        justifyContent: isBot ? "flex-start" : "flex-end",
        margin: "10px 0",
      }}
    >
      <div
        style={{
          width: "fit-content",
          maxWidth: "78%",
          padding: "12px 14px",
          borderRadius: 16,
          border: "1px solid #2f2f2f",
          background: isBot
            ? "rgba(255,255,255,0.03)"
            : "rgba(255,255,255,0.06)",
          lineHeight: 1.35,
          textAlign: "left",
        }}
      >
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>
          {isBot ? "Assistant" : "You"}
        </div>

        {isBot ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={{
              table: ({ children }) => (
                <div style={{ overflowX: "auto", marginTop: 10 }}>
                  <table
                    style={{
                      width: "100%",
                      borderCollapse: "collapse",
                      fontSize: 14,
                    }}
                  >
                    {children}
                  </table>
                </div>
              ),
              th: ({ children }) => (
                <th
                  style={{
                    border: "1px solid #3a3a3a",
                    padding: "8px",
                    textAlign: "left",
                    background: "rgba(255,255,255,0.05)",
                  }}
                >
                  {children}
                </th>
              ),
              td: ({ children }) => (
                <td
                  style={{
                    border: "1px solid #3a3a3a",
                    padding: "8px",
                    verticalAlign: "top",
                  }}
                >
                  {children}
                </td>
              ),
              p: ({ children }) => (
                <p style={{ margin: "6px 0" }}>{children}</p>
              ),
              ul: ({ children }) => (
                <ul style={{ margin: "6px 0 6px 18px" }}>{children}</ul>
              ),
              ol: ({ children }) => (
                <ol style={{ margin: "6px 0 6px 18px" }}>{children}</ol>
              ),
            }}
          >
            {text}
          </ReactMarkdown>
        ) : (
          <div style={{ whiteSpace: "pre-wrap" }}>{text}</div>
        )}
      </div>
    </div>
  );
}

function AuthScreen({ mode, setMode, form, setForm, busy, error, onSubmit }) {
  const isRegister = mode === "register";

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: 18,
      }}
    >
      <div
        style={{
          width: "min(460px, 94vw)",
          border: "1px solid #2f2f2f",
          borderRadius: 18,
          padding: 20,
          background: "rgba(11,13,18,0.95)",
        }}
      >
        <h2 style={{ margin: "0 0 8px", textAlign: "left" }}>
          AI Virtual Clinic
        </h2>
        <p style={{ margin: "0 0 16px", opacity: 0.8, textAlign: "left" }}>
          {isRegister ? "Create account" : "Sign in"}
        </p>

        <div style={{ display: "flex", gap: 8, marginBottom: 14 }}>
          <button
            type="button"
            onClick={() => setMode("login")}
            style={{
              flex: 1,
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid #2f2f2f",
              background:
                mode === "login" ? "rgba(255,255,255,0.12)" : "transparent",
              color: "#ffffff",
              cursor: "pointer",
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            onClick={() => setMode("register")}
            style={{
              flex: 1,
              padding: "10px 12px",
              borderRadius: 10,
              border: "1px solid #2f2f2f",
              background:
                mode === "register" ? "rgba(255,255,255,0.12)" : "transparent",
              color: "#ffffff",
              cursor: "pointer",
            }}
          >
            Create Account
          </button>
        </div>

        <form
          onSubmit={onSubmit}
          style={{ display: "flex", flexDirection: "column", gap: 10 }}
        >
          <select
            value={form.role}
            onChange={(event) =>
              setForm((prev) => ({ ...prev, role: event.target.value }))
            }
            style={{
              padding: "11px 12px",
              borderRadius: 10,
              border: "1px solid #2f2f2f",
              background: "#11141b",
              color: "#ffffff",
            }}
            disabled={busy}
          >
            <option value="patients">Patient</option>
            <option value="doctors">Doctor</option>
            <option value="administrator">Administrator</option>
          </select>

          {isRegister && (
            <input
              type="text"
              value={form.full_name}
              onChange={(event) =>
                setForm((prev) => ({ ...prev, full_name: event.target.value }))
              }
              placeholder="Full name"
              style={{
                padding: "11px 12px",
                borderRadius: 10,
                border: "1px solid #2f2f2f",
                background: "#11141b",
                color: "#ffffff",
              }}
              disabled={busy}
            />
          )}

          <input
            type="email"
            value={form.email}
            onChange={(event) =>
              setForm((prev) => ({ ...prev, email: event.target.value }))
            }
            placeholder="Email"
            style={{
              padding: "11px 12px",
              borderRadius: 10,
              border: "1px solid #2f2f2f",
              background: "#11141b",
              color: "#ffffff",
            }}
            disabled={busy}
          />

          <input
            type="password"
            value={form.password}
            onChange={(event) =>
              setForm((prev) => ({ ...prev, password: event.target.value }))
            }
            placeholder="Password"
            style={{
              padding: "11px 12px",
              borderRadius: 10,
              border: "1px solid #2f2f2f",
              background: "#11141b",
              color: "#ffffff",
            }}
            disabled={busy}
          />

          {error && (
            <div
              style={{
                border: "1px solid #7d2a2a",
                background: "rgba(173,61,61,0.2)",
                borderRadius: 10,
                padding: "9px 10px",
                textAlign: "left",
                fontSize: 13,
              }}
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={busy}
            style={{
              marginTop: 6,
              padding: "11px 14px",
              borderRadius: 10,
              border: "1px solid #2f2f2f",
              background: "rgba(255,255,255,0.1)",
              color: "#ffffff",
              cursor: busy ? "not-allowed" : "pointer",
            }}
          >
            {busy
              ? "Please wait..."
              : isRegister
                ? "Create Account"
                : "Sign In"}
          </button>
        </form>
      </div>
    </div>
  );
}

function DatabaseModal({ open, onClose, onRefresh, busy, error, data, role }) {
  if (!open) {
    return null;
  }

  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.62)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        padding: 16,
        zIndex: 50,
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: "min(900px, 95vw)",
          maxHeight: "86vh",
          overflow: "hidden",
          border: "1px solid #2f2f2f",
          borderRadius: 14,
          background: "#0f1117",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <div
          style={{
            padding: "12px 14px",
            borderBottom: "1px solid #2f2f2f",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            gap: 10,
          }}
        >
          <div style={{ fontWeight: 600 }}>Database Data</div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              type="button"
              onClick={onRefresh}
              disabled={busy}
              style={{
                padding: "8px 10px",
                borderRadius: 9,
                border: "1px solid #2f2f2f",
                background: "rgba(255,255,255,0.08)",
                color: "#ffffff",
                cursor: busy ? "not-allowed" : "pointer",
              }}
            >
              Refresh
            </button>
            <button
              type="button"
              onClick={onClose}
              style={{
                padding: "8px 10px",
                borderRadius: 9,
                border: "1px solid #2f2f2f",
                background: "transparent",
                color: "#ffffff",
                cursor: "pointer",
              }}
            >
              Close
            </button>
          </div>
        </div>

        <div style={{ padding: "14px", overflow: "auto" }}>
          <p style={{ marginTop: 0, opacity: 0.7 }}>
            {role === "administrator"
              ? "Administrator view: doctors and patients collections."
              : "Your stored profile data."}
          </p>
          {busy && <p>Loading...</p>}
          {error && (
            <div
              style={{
                border: "1px solid #7d2a2a",
                background: "rgba(173,61,61,0.2)",
                borderRadius: 10,
                padding: "10px 12px",
                marginBottom: 12,
              }}
            >
              {error}
            </div>
          )}
          {!busy && !error && (
            <pre
              style={{
                margin: 0,
                fontSize: 13,
                lineHeight: 1.45,
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                border: "1px solid #2f2f2f",
                borderRadius: 10,
                padding: 12,
                background: "rgba(255,255,255,0.03)",
              }}
            >
              {JSON.stringify(data, null, 2)}
            </pre>
          )}
        </div>
      </div>
    </div>
  );
}

export default function App() {
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
    full_name: "",
    email: "",
    password: "",
  });

  const [messages, setMessages] = useState([
    { role: "bot", text: INITIAL_MESSAGE },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const [sessionId, setSessionId] = useState(null);
  const [modelType, setModelType] = useState(null);
  const [questions, setQuestions] = useState([]);
  const [qIndex, setQIndex] = useState(0);
  const [answers, setAnswers] = useState({});

  const [dbModalOpen, setDbModalOpen] = useState(false);
  const [dbBusy, setDbBusy] = useState(false);
  const [dbError, setDbError] = useState("");
  const [dbData, setDbData] = useState(null);

  // Voice feature state
  const [voiceSettingsOpen, setVoiceSettingsOpen] = useState(false);
  const [voiceSettings, setVoiceSettings] = useState(() => getVoiceSettings());
  const ttsRef = useRef(null);

  const scrollRef = useRef(null);

  const isAuthenticated = Boolean(auth?.token);

  const currentQuestion = useMemo(() => {
    if (!questions.length) return null;
    if (qIndex >= questions.length) return null;
    return questions[qIndex];
  }, [questions, qIndex]);

  const modelConfig = useMemo(
    () => MODEL_CONFIG[modelType] || MODEL_CONFIG.symptom,
    [modelType],
  );

  // Initialize TTS on mount
  useEffect(() => {
    initializeTTS().then(() => {
      ttsRef.current = getTextToSpeech({
        voice: voiceSettings.voice,
        speed: voiceSettings.speed
      });
    });
  }, []);

  // Speak bot messages when auto-speak is enabled
  const speakMessage = useCallback((text) => {
    if (!voiceSettings.autoSpeak || !ttsRef.current) return;
    // Strip markdown for speech
    const plainText = text.replace(/\*\*/g, '').replace(/\*/g, '').replace(/#{1,6}\s/g, '');
    ttsRef.current.speak(plainText, {
      voice: voiceSettings.voice,
      speed: voiceSettings.speed
    });
  }, [voiceSettings]);

  function push(role, text) {
    setMessages((previous) => [...previous, { role, text }]);
    // Auto-speak bot messages
    if (role === 'bot') {
      setTimeout(() => speakMessage(text), 100);
    }
  }

  function resetConversation() {
    setMessages([{ role: "bot", text: INITIAL_MESSAGE }]);
    setInput("");
    setBusy(false);
    setSessionId(null);
    setModelType(null);
    setQuestions([]);
    setQIndex(0);
    setAnswers({});
  }

  function saveAuth(payload) {
    const next = {
      token: payload.token || payload.access_token,
      role: payload.role,
      email: payload.email || payload.user?.email,
    };
    localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(next));
    setAuth(next);
  }

  function logout() {
    localStorage.removeItem(AUTH_STORAGE_KEY);
    setAuth(null);
    setDbModalOpen(false);
    setDbData(null);
    setDbError("");
    resetConversation();
  }

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (currentQuestion) {
      push("bot", currentQuestion.question);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [qIndex, questions]);

  function getModelLabel(modelTypeKey) {
    const config = MODEL_CONFIG[modelTypeKey] || MODEL_CONFIG.symptom;
    return config.label;
  }

  function normalizeYesNo(text) {
    const value = (text || "").trim().toLowerCase();
    if (["yes", "y", "yeah", "yep", "true", "1"].includes(value)) return 1;
    if (["no", "n", "nope", "false", "0"].includes(value)) return 0;
    return null;
  }

  async function startConversation(userText) {
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/predict/symptoms/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || "Could not start prediction.");
      }

      const detectedModelType = data.model_type || "symptom";
      setModelType(detectedModelType);

      if (data.status === "complete") {
        setSessionId(null);
        setQuestions([]);
        setQIndex(0);
        setAnswers({});

        const prediction =
          data?.prediction?.prediction ||
          data?.prediction?.predictions ||
          "N/A";
        const modelLabel = getModelLabel(detectedModelType);
        push("bot", `**${modelLabel}:** ${prediction}`);

        if (data?.prediction?.confidence) {
          push(
            "bot",
            `**Confidence:** ${(data.prediction.confidence * 100).toFixed(1)}%`,
          );
        }

        if (
          data?.prediction?.probabilities &&
          Array.isArray(data.prediction.probabilities)
        ) {
          const topThree = data.prediction.probabilities.slice(0, 3);
          const text = topThree
            .map(
              (item, index) =>
                `${index + 1}. ${item[0]}: ${(item[1] * 100).toFixed(1)}%`,
            )
            .join("\n");
          push("bot", `**Top Predictions:**\n${text}`);
        }

        if (data.explanation) push("bot", data.explanation);
        if (data.disclaimer) push("bot", `*${data.disclaimer}*`);
        return;
      }

      setSessionId(data.session_id);
      const newQuestions = Array.isArray(data.questions) ? data.questions : [];
      setQuestions(newQuestions);
      setQIndex(0);

      const initialAnswers = {};
      for (const question of newQuestions) {
        initialAnswers[question.feature] = null;
      }
      setAnswers(initialAnswers);

      const config = MODEL_CONFIG[detectedModelType] || MODEL_CONFIG.symptom;
      push(
        "bot",
        `I will help with ${config.description}. Let me ask a few questions.`,
      );
    } finally {
      setBusy(false);
    }
  }

  async function submitAll(finalAnswers) {
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/predict/symptoms/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          answers: finalAnswers,
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.error || "Could not submit answers.");
      }

      const prediction =
        data?.prediction?.prediction || data?.prediction?.predictions || "N/A";
      const currentModelType = data.model_type || modelType || "symptom";
      const modelLabel = getModelLabel(currentModelType);

      push("bot", `**${modelLabel}:** ${prediction}`);

      if (data?.prediction?.probability) {
        push(
          "bot",
          `**Probability:** ${(data.prediction.probability * 100).toFixed(1)}%`,
        );
      }
      if (data.explanation) push("bot", data.explanation);
      if (data.disclaimer) push("bot", `*${data.disclaimer}*`);

      setSessionId(null);
      setQuestions([]);
      setQIndex(0);
      setAnswers({});
      setModelType(null);
    } finally {
      setBusy(false);
    }
  }

  async function onSend() {
    const userText = input.trim();
    if (!userText || busy) return;
    setInput("");
    push("user", userText);

    try {
      if (!sessionId) {
        await startConversation(userText);
        return;
      }

      const yesNo = normalizeYesNo(userText);
      const answerValue = yesNo === null ? userText : yesNo;

      if (!currentQuestion) {
        push("bot", "No more questions left. Submitting now.");
        await submitAll(answers);
        return;
      }

      const feature = currentQuestion.feature;
      const updatedAnswers = { ...answers, [feature]: answerValue };
      setAnswers(updatedAnswers);

      const next = qIndex + 1;
      if (next >= questions.length) {
        push("bot", "Thanks. Analyzing now.");
        await submitAll(updatedAnswers);
      } else {
        setQIndex(next);
      }
    } catch (error) {
      push("bot", `Error: ${error.message}`);
    }
  }

  async function fetchDatabaseData() {
    if (!auth?.token) return;

    setDbBusy(true);
    setDbError("");
    try {
      const endpoint =
        auth.role === "administrator" ? "/auth/admin/users" : "/auth/profile";
      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "GET",
        headers: {
          Authorization: `Bearer ${auth.token}`,
        },
      });

      const data = await response.json();
      if (!response.ok) {
        if (response.status === 401) {
          logout();
          throw new Error("Session expired. Please sign in again.");
        }
        throw new Error(data?.error || "Failed to load database data.");
      }

      setDbData(data);
    } catch (error) {
      setDbError(error.message);
      setDbData(null);
    } finally {
      setDbBusy(false);
    }
  }

  async function handleAuthSubmit(event) {
    event.preventDefault();
    if (authBusy) return;

    const isRegister = authMode === "register";
    const role = authForm.role;
    const email = authForm.email.trim().toLowerCase();
    const password = authForm.password;
    const fullName = authForm.full_name.trim();

    if (!role || !email || !password || (isRegister && !fullName)) {
      setAuthError("Please fill all required fields.");
      return;
    }

    setAuthBusy(true);
    setAuthError("");

    try {
      const endpoint = isRegister ? "/auth/register" : "/auth/login";
      const payload = isRegister
        ? { role, email, password, full_name: fullName }
        : { role, email, password };

      const response = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.error || "Authentication failed.");
      }

      saveAuth(data);
      resetConversation();
      setAuthForm((prev) => ({
        role: prev.role,
        full_name: "",
        email: "",
        password: "",
      }));
    } catch (error) {
      setAuthError(error.message);
    } finally {
      setAuthBusy(false);
    }
  }

  if (!isAuthenticated) {
    return (
      <AuthScreen
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

  return (
    <>
      <div
        style={{
          position: "fixed",
          inset: 0,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          padding: 18,
        }}
      >
        <div
          style={{
            width: "min(980px, 95vw)",
            height: "min(760px, 90vh)",
            border: "1px solid #2f2f2f",
            borderRadius: 18,
            padding: 16,
            display: "flex",
            flexDirection: "column",
            gap: 12,
            background: "rgba(11,13,18,0.96)",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: 10,
            }}
          >
            <h3 style={{ margin: 0 }}>AI Virtual Clinic</h3>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                flexWrap: "wrap",
                justifyContent: "end",
              }}
            >
              <div style={{ fontSize: 12, opacity: 0.8 }}>
                {auth.email} ({auth.role})
              </div>
              <button
                type="button"
                onClick={async () => {
                  setDbModalOpen(true);
                  await fetchDatabaseData();
                }}
                style={{
                  padding: "8px 10px",
                  borderRadius: 9,
                  border: "1px solid #2f2f2f",
                  background: "rgba(255,255,255,0.08)",
                  color: "#ffffff",
                  cursor: "pointer",
                }}
              >
                Database
              </button>
              <button
                type="button"
                onClick={() => setVoiceSettingsOpen(true)}
                style={{
                  padding: "8px 10px",
                  borderRadius: 9,
                  border: "1px solid #2f2f2f",
                  background: "transparent",
                  color: "#ffffff",
                  cursor: "pointer",
                }}
                title="Voice Settings"
              >
                🎤
              </button>
              <button
                type="button"
                onClick={logout}
                style={{
                  padding: "8px 10px",
                  borderRadius: 9,
                  border: "1px solid #2f2f2f",
                  background: "transparent",
                  color: "#ffffff",
                  cursor: "pointer",
                }}
              >
                Logout
              </button>
            </div>
          </div>

          <div style={{ fontSize: 12, opacity: 0.75 }}>
            {sessionId
              ? `${modelConfig.label} | Q ${Math.min(qIndex + 1, questions.length)} / ${questions.length}`
              : "No active session"}
          </div>

          {modelType && sessionId && (
            <div
              style={{
                fontSize: 11,
                padding: "4px 8px",
                borderRadius: 8,
                background: `${modelConfig.badgeColor}20`,
                border: `1px solid ${modelConfig.badgeColor}`,
                width: "fit-content",
              }}
            >
              {modelConfig.label} mode
            </div>
          )}

          <style>{`.no-scrollbar::-webkit-scrollbar{display:none} .no-scrollbar{-ms-overflow-style:none; scrollbar-width:none;}`}</style>
          <div
            ref={scrollRef}
            className="no-scrollbar"
            style={{
              flex: 1,
              overflowY: "auto",
              WebkitOverflowScrolling: "touch",
              border: "1px solid #2f2f2f",
              borderRadius: 16,
              padding: 14,
              msOverflowStyle: "none",
              scrollbarWidth: "none",
            }}
          >
            {messages.map((message, index) => (
              <Bubble key={index} role={message.role} text={message.text} />
            ))}
          </div>

          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            {isSpeechRecognitionSupported() && (
              <VoiceButton
                onResult={(text) => {
                  setInput(text);
                }}
                onInterim={(text) => setInput(text)}
                onError={(error) => console.error("Voice error:", error)}
                disabled={busy}
                size="medium"
              />
            )}
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => (event.key === "Enter" ? onSend() : null)}
              placeholder={
                sessionId
                  ? "Answer the question..."
                  : "Describe your symptoms... (for example: fever, chest pain, high blood sugar, anxiety)"
              }
              style={{
                flex: 1,
                padding: "12px 12px",
                borderRadius: 14,
                border: "1px solid #2f2f2f",
                outline: "none",
                background: "#11141b",
                color: "#ffffff",
              }}
              disabled={busy}
            />
            <button
              onClick={onSend}
              disabled={busy || !input.trim()}
              style={{
                padding: "12px 16px",
                borderRadius: 14,
                border: "1px solid #2f2f2f",
                cursor: busy ? "not-allowed" : "pointer",
                opacity: busy ? 0.7 : 1,
                background: "rgba(255,255,255,0.1)",
                color: "#ffffff",
              }}
            >
              {busy ? "..." : "Send"}
            </button>
          </div>
        </div>
      </div>

      <DatabaseModal
        open={dbModalOpen}
        onClose={() => setDbModalOpen(false)}
        onRefresh={fetchDatabaseData}
        busy={dbBusy}
        error={dbError}
        data={dbData}
        role={auth.role}
      />

      <VoiceSettings
        isOpen={voiceSettingsOpen}
        onClose={() => setVoiceSettingsOpen(false)}
        onSettingsChange={(newSettings) => {
          setVoiceSettings(newSettings);
          if (ttsRef.current) {
            ttsRef.current.setVoice(newSettings.voice);
            ttsRef.current.setSpeed(newSettings.speed);
          }
        }}
      />
    </>
  );
}
