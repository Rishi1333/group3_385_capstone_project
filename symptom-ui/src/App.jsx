import { useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_BASE = "http://127.0.0.1:3000";

// Model configuration for display
const MODEL_CONFIG = {
  symptom: {
    label: "Symptom Analysis",
    icon: "🩺",
    color: "#64c864",
    description: "General symptom-based disease prediction"
  },
  heart: {
    label: "Heart Health",
    icon: "❤️",
    color: "#ff6464",
    description: "Heart disease risk assessment"
  },
  diabetes: {
    label: "Diabetes Screening",
    icon: "🩸",
    color: "#6495ed",
    description: "Diabetes risk prediction"
  },
  mental_health: {
    label: "Mental Health",
    icon: "🧠",
    color: "#9b59b6",
    description: "Mental health screening"
  },
  symptom_disease_extended: {
    label: "Disease Prediction",
    icon: "🔬",
    color: "#e67e22",
    description: "NLP-based disease prediction"
  }
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
        }}
      >
        <div style={{ fontSize: 12, opacity: 0.7, marginBottom: 6 }}>
          {isBot ? "Assistant" : "You"}
        </div>

        {/* Render markdown for bot (tables, bullets, bold, etc.) */}
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

export default function App() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hi! Tell me what you're feeling. I can help with general symptoms, heart concerns, diabetes screening, mental health questions, or detailed disease prediction." },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const [sessionId, setSessionId] = useState(null);
  const [modelType, setModelType] = useState(null);
  const [questions, setQuestions] = useState([]); // [{feature, question}]
  const [qIndex, setQIndex] = useState(0);
  const [answers, setAnswers] = useState({}); // feature -> value

  const scrollRef = useRef(null);

  const currentQuestion = useMemo(() => {
    if (!questions.length) return null;
    if (qIndex >= questions.length) return null;
    return questions[qIndex];
  }, [questions, qIndex]);

  const modelConfig = useMemo(() => {
    return MODEL_CONFIG[modelType] || MODEL_CONFIG.symptom;
  }, [modelType]);

  function push(role, text) {
    setMessages((m) => [...m, { role, text }]);
  }

  // Auto-scroll on new messages
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages]);

  // Ask the next question when index changes
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

  async function startConversation(userText) {
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/predict/symptoms/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: userText }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "start failed");

      // Set model type
      const detectedModelType = data.model_type || "symptom";
      setModelType(detectedModelType);

      // If backend immediately completes
      if (data.status === "complete") {
        setSessionId(null);
        setQuestions([]);
        setQIndex(0);
        setAnswers({});
        
        const predName = data?.prediction?.prediction || data?.prediction?.predictions || "N/A";
        const modelLabel = getModelLabel(detectedModelType);
        
        push("bot", `**${modelLabel}:** ${predName}`);
        
        // Show confidence if available
        if (data?.prediction?.confidence) {
          push("bot", `**Confidence:** ${(data.prediction.confidence * 100).toFixed(1)}%`);
        }
        
        // Show probabilities for NLP model
        if (data?.prediction?.probabilities && Array.isArray(data.prediction.probabilities)) {
          const topProbs = data.prediction.probabilities.slice(0, 3);
          const probText = topProbs.map((p, i) => `${i + 1}. ${p[0]}: ${(p[1] * 100).toFixed(1)}%`).join("\n");
          push("bot", `**Top Predictions:**\n${probText}`);
        }
        
        if (data.explanation) push("bot", data.explanation);
        if (data.disclaimer) push("bot", `*${data.disclaimer}*`);
        return;
      }

      setSessionId(data.session_id);

      const qs = Array.isArray(data.questions) ? data.questions : [];
      setQuestions(qs);
      setQIndex(0);

      const init = {};
      for (const q of qs) init[q.feature] = null;
      setAnswers(init);

      // Show which model was selected
      const config = MODEL_CONFIG[detectedModelType] || MODEL_CONFIG.symptom;
      push("bot", `${config.icon} I'll help with ${config.description.toLowerCase()}. Let me ask you a few questions.`);
    } finally {
      setBusy(false);
    }
  }

  async function submitAll(finalAnswers) {
    setBusy(true);
    try {
      const res = await fetch(`${API_BASE}/predict/symptoms/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          answers: finalAnswers,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data?.error || "submit failed");

      const predName = data?.prediction?.prediction || data?.prediction?.predictions || "N/A";
      const currentModelType = data.model_type || modelType || "symptom";
      const modelLabel = getModelLabel(currentModelType);

      push("bot", `**${modelLabel}:** ${predName}`);
      
      // Show probability if available
      if (data?.prediction?.probability) {
        push("bot", `**Probability:** ${(data.prediction.probability * 100).toFixed(1)}%`);
      }
      
      if (data.explanation) push("bot", data.explanation);
      if (data.disclaimer) push("bot", `*${data.disclaimer}*`);

      // Reset
      setSessionId(null);
      setQuestions([]);
      setQIndex(0);
      setAnswers({});
      setModelType(null);
    } finally {
      setBusy(false);
    }
  }

  function normalizeYesNo(text) {
    const t = (text || "").trim().toLowerCase();
    if (["yes", "y", "yeah", "yep", "true", "1"].includes(t)) return 1;
    if (["no", "n", "nope", "false", "0"].includes(t)) return 0;
    return null;
  }

  async function onSend() {
    const userText = input.trim();
    if (!userText || busy) return;
    setInput("");

    push("user", userText);

    // If no session: start
    if (!sessionId) {
      await startConversation(userText);
      return;
    }

    // Otherwise: answer current question
    const yn = normalizeYesNo(userText);
    
    // For heart/diabetes/mental_health models, we may need to accept numeric/text answers too
    let answerValue = yn;
    if (yn === null) {
      // Not a yes/no answer - could be a number or text
      answerValue = userText;
    }

    if (!currentQuestion) {
      push("bot", "No more questions left — submitting now.");
      await submitAll(answers);
      return;
    }

    const feat = currentQuestion.feature;
    const updated = { ...answers, [feat]: answerValue };
    setAnswers(updated);

    const nextIndex = qIndex + 1;
    if (nextIndex >= questions.length) {
      push("bot", "Thanks — analyzing now.");
      await submitAll(updated);
    } else {
      setQIndex(nextIndex);
    }
  }

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
      {/* Centered chat card */}
      <div
        style={{
          width: "min(900px, 95vw)",
          height: "min(760px, 90vh)",
          border: "1px solid #2f2f2f",
          borderRadius: 18,
          padding: 16,
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        <div
          style={{ display: "flex", justifyContent: "space-between", gap: 10 }}
        >
          <h3 style={{ margin: 0 }}>AI Virtual Clinic</h3>
          <div style={{ fontSize: 12, opacity: 0.75, alignSelf: "center" }}>
            {sessionId
              ? `${modelConfig.label} • Q ${Math.min(qIndex + 1, questions.length)} / ${questions.length}`
              : "No active session"}
          </div>
        </div>

        {/* Model type indicator */}
        {modelType && sessionId && (
          <div
            style={{
              fontSize: 11,
              padding: "4px 8px",
              borderRadius: 8,
              background: `${modelConfig.color}20`,
              border: `1px solid ${modelConfig.color}`,
              width: "fit-content",
            }}
          >
            {modelConfig.icon} {modelConfig.label} Mode
          </div>
        )}

        {/* Chat messages */}
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
          {messages.map((m, i) => (
            <Bubble key={i} role={m.role} text={m.text} />
          ))}
        </div>

        {/* Input */}
        <div style={{ display: "flex", gap: 10 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => (e.key === "Enter" ? onSend() : null)}
            placeholder={
              sessionId
                ? "Answer the question..."
                : "Describe your symptoms… (e.g., I have fever and headache, chest pain, high blood sugar, feeling anxious…)"
            }
            style={{
              flex: 1,
              padding: "12px 12px",
              borderRadius: 14,
              border: "1px solid #2f2f2f",
              outline: "none",
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
            }}
          >
            {busy ? "..." : "Send"}
          </button>
        </div>
      </div>
    </div>
  );
}
