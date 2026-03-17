import { useEffect, useRef, useState } from "react";
import "@google/model-viewer";
import maleBodyModel from "../assets/male_body.glb";
import femaleBodyModel from "../assets/study_human_female_sculpt.glb";
import "./PatientDashboard.css";

export default function PatientDashboard({
  patientEmail,
  patientGender,
  onFocusChatInput,
}) {
  const modelRef = useRef(null);
  const [autoRotate, setAutoRotate] = useState(true);
  const normalizedGender = String(patientGender || "")
    .trim()
    .toLowerCase();
  const useFemaleModel = ["female", "f", "woman", "girl"].includes(
    normalizedGender,
  );
  const bodyModelSrc = useFemaleModel ? femaleBodyModel : maleBodyModel;

  useEffect(() => {
    if (modelRef.current) {
      modelRef.current.autoRotate = autoRotate;
    }
  }, [autoRotate]);

  return (
    <main className="patient-dashboard" aria-label="Patient dashboard">
      <section className="patient-hero-card" aria-labelledby="patient-dashboard-title">
        <div className="patient-hero-copy">
          <p className="patient-kicker">Patient Dashboard</p>
          <h2 id="patient-dashboard-title">Interactive body model for guided symptom intake</h2>
          <p className="patient-description">
            Rotate and inspect the anatomical model before starting your symptom
            conversation.
          </p>

          <dl className="patient-meta-list">
            <div>
              <dt>Account</dt>
              <dd>{patientEmail || "Patient user"}</dd>
            </div>
            <div>
              <dt>Mode</dt>
              <dd>Clinical Assistant</dd>
            </div>
            <div>
              <dt>Availability</dt>
              <dd>24/7 Triage</dd>
            </div>
          </dl>

          <div className="patient-action-row">
            <button
              type="button"
              className="patient-primary-btn"
              onClick={onFocusChatInput}
            >
              Start Symptom Chat
            </button>
            <button
              type="button"
              className="patient-secondary-btn"
              onClick={() => setAutoRotate((prev) => !prev)}
              aria-pressed={autoRotate}
            >
              {autoRotate ? "Pause Rotation" : "Resume Rotation"}
            </button>
          </div>
        </div>

        <figure className="patient-model-panel">
          <model-viewer
            ref={modelRef}
            src={bodyModelSrc}
            alt="Interactive 3D anatomical human body model for symptom localization."
            loading="eager"
            camera-controls
            shadow-intensity="0.45"
            exposure="1.05"
            interaction-prompt="auto"
            environment-image="neutral"
            touch-action="pan-y"
            tabIndex="0"
            aria-label="3D anatomy viewer"
            className="patient-model-viewer"
          />
          <figcaption className="patient-model-help">
            Drag to rotate, pinch or scroll to zoom, and use your keyboard focus
            here for accessible navigation.
          </figcaption>
        </figure>
      </section>
    </main>
  );
}
