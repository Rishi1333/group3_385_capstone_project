import "./PatientDashboard.css";

export default function PatientDashboard({
  patientEmail,
  patientReports,
  onFocusChatInput,
}) {
  const reports = Array.isArray(patientReports) ? patientReports : [];

  const formatDateTime = (value) => {
    if (!value) return "Not available";
    const parsed = new Date(value);
    if (Number.isNaN(parsed.getTime())) return value;
    return parsed.toLocaleString();
  };

  const extractRiskLabel = (report) => {
    const risk = report?.risk_assessment;
    if (!risk) return "Unknown";
    if (typeof risk === "string") return risk;
    return (
      risk?.overall_risk_level ||
      risk?.level ||
      risk?.summary ||
      "Unknown"
    );
  };

  const extractComplaint = (report) =>
    report?.chief_complaint ||
    report?.clinical_snapshot?.chief_complaint ||
    "No chief complaint captured.";

  const extractSymptoms = (report) => {
    const symptoms = report?.clinical_snapshot?.symptoms;
    if (!Array.isArray(symptoms) || symptoms.length === 0) {
      return "No symptoms recorded";
    }
    return symptoms.slice(0, 4).join(", ");
  };

  return (
    <main className="patient-dashboard" aria-label="Patient dashboard">
      <section className="patient-hero-card" aria-labelledby="patient-dashboard-title">
        <div className="patient-hero-copy">
          <p className="patient-kicker">Patient Dashboard</p>
          <h2 id="patient-dashboard-title">Guided symptom intake</h2>
          <p className="patient-description">
            Start your conversation with the clinical assistant to share your
            symptoms and receive next-step guidance.
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
          </div>
        </div>

        <section className="patient-reports" aria-labelledby="patient-reports-title">
          <div className="patient-reports-header">
            <h3 id="patient-reports-title">Saved Clinical Reports</h3>
            <span className="patient-reports-count">{reports.length}</span>
          </div>

          {reports.length === 0 ? (
            <p className="patient-reports-empty">
              No saved reports yet. Complete a triage chat to generate one.
            </p>
          ) : (
            <div className="patient-reports-grid">
              {reports.slice(0, 6).map((report, index) => {
                const reportId = report?.report_id || `REPORT-${index + 1}`;
                const generatedAt = report?.generated_at || report?.stored_at;
                const riskLabel = extractRiskLabel(report);
                const complaint = extractComplaint(report);
                const symptoms = extractSymptoms(report);

                return (
                  <article
                    key={`${reportId}-${generatedAt || index}`}
                    className="patient-report-card"
                  >
                    <p className="patient-report-id">{reportId}</p>
                    <p className="patient-report-date">
                      Generated: {formatDateTime(generatedAt)}
                    </p>
                    <p className="patient-report-risk">Risk: {riskLabel}</p>
                    <p className="patient-report-complaint">{complaint}</p>
                    <p className="patient-report-symptoms">Symptoms: {symptoms}</p>
                  </article>
                );
              })}
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
