import backgroundHero from "../assets/backgroundHeropage.jpeg";
import "./LandingPage.css";

const stats = [
  { value: "24/7", label: "Always-on triage support" },
  { value: "Fast", label: "Instant symptom guidance" },
  { value: "Secure", label: "Built for patient workflows" },
];

const features = [
  {
    eyebrow: "AI triage",
    title: "Symptom analysis in seconds",
    description:
      "Patients can describe symptoms naturally and receive structured follow-up questions without waiting for a live intake call.",
  },
  {
    eyebrow: "Image support",
    title: "Add visual context when needed",
    description:
      "Support for image upload makes it easier to pair symptom descriptions with visible conditions during assessment.",
  },
  {
    eyebrow: "Clinical flow",
    title: "Built for patient, doctor, and admin access",
    description:
      "A single entry point supports different user roles while keeping the intake experience consistent.",
  },
];

const steps = [
  {
    number: "01",
    title: "Sign in or register",
    description: "Choose your role and access the clinic workspace.",
  },
  {
    number: "02",
    title: "Describe symptoms",
    description: "Share what you are feeling in plain language or attach an image.",
  },
  {
    number: "03",
    title: "Receive guided analysis",
    description: "The assistant asks follow-up questions and prepares a clear response.",
  },
];

function BrandMark() {
  return (
    <svg
      viewBox="0 0 48 48"
      width="28"
      height="28"
      aria-hidden="true"
      className="landing-brand-mark"
    >
      <circle cx="24" cy="24" r="22" />
      <path d="M14 25h6l3-7 4 14 3-7h4" />
    </svg>
  );
}

export default function LandingPage({
  mode,
  setMode,
  form,
  setForm,
  busy,
  error,
  onSubmit,
}) {
  const isRegister = mode === "register";

  const scrollToSection = (sectionId, nextMode) => {
    if (nextMode) {
      setMode(nextMode);
    }

    document
      .getElementById(sectionId)
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div
      className="landing-page-shell"
      style={{ "--landing-hero-image": `url(${backgroundHero})` }}
    >
      <section className="landing-hero-section" id="home">
        <div className="landing-overlay" />
        <div className="landing-container">
          <nav className="landing-nav" aria-label="Primary navigation">
            <button
              type="button"
              className="landing-brand"
              onClick={() => scrollToSection("home")}
            >
              <span className="landing-brand-icon">
                <BrandMark />
              </span>
              <span className="landing-brand-copy">
                <strong>AI Virtual Clinic</strong>
                <span>Smart intake for modern care</span>
              </span>
            </button>

            <div className="landing-nav-links">
              <button type="button" onClick={() => scrollToSection("home")}>
                Home
              </button>
              <button type="button" onClick={() => scrollToSection("about")}>
                About
              </button>
              <button type="button" onClick={() => scrollToSection("features")}>
                Features
              </button>
              <button type="button" onClick={() => scrollToSection("contact")}>
                Contact
              </button>
            </div>

            <div className="landing-nav-actions">
              <button
                type="button"
                className="landing-nav-ghost"
                onClick={() => scrollToSection("auth-panel", "login")}
              >
                Sign In
              </button>
              <button
                type="button"
                className="landing-nav-cta"
                onClick={() => scrollToSection("auth-panel", "register")}
              >
                Get Started
              </button>
            </div>
          </nav>

          <div className="landing-hero-grid">
            <div className="landing-hero-copy">
              <span className="landing-badge">Modern AI Healthcare Platform</span>
              <h1>Smart health insights, powered by AI.</h1>
              <p>
                Describe symptoms, upload context when needed, and move through
                a guided intake experience designed for faster decisions.
              </p>

              <div className="landing-hero-actions">
                <button
                  type="button"
                  className="landing-primary-btn"
                  onClick={() => scrollToSection("auth-panel", "register")}
                >
                  Start Now
                </button>
                <button
                  type="button"
                  className="landing-secondary-btn"
                  onClick={() => scrollToSection("features")}
                >
                  Learn More
                </button>
              </div>

              <div className="landing-stat-grid">
                {stats.map((stat) => (
                  <article className="landing-stat-card" key={stat.label}>
                    <h2>{stat.value}</h2>
                    <p>{stat.label}</p>
                  </article>
                ))}
              </div>
            </div>

            <aside className="landing-auth-column" id="auth-panel">
              <div className="landing-auth-card">
                <div className="landing-auth-header">
                  <span className="landing-auth-icon">
                    <BrandMark />
                  </span>
                  <h2>AI Virtual Clinic</h2>
                  <p>{isRegister ? "Create your patient account" : "Welcome back"}</p>
                </div>

                <div className="landing-auth-tabs" role="tablist" aria-label="Account access">
                  <button
                    type="button"
                    role="tab"
                    aria-selected={!isRegister}
                    className={`landing-auth-tab ${!isRegister ? "is-active" : ""}`}
                    onClick={() => setMode("login")}
                  >
                    Sign In
                  </button>
                  <button
                    type="button"
                    role="tab"
                    aria-selected={isRegister}
                    className={`landing-auth-tab ${isRegister ? "is-active" : ""}`}
                    onClick={() => setMode("register")}
                  >
                    Register
                  </button>
                </div>

                <form onSubmit={onSubmit} className="landing-auth-form">
                  {isRegister && (
                    <>
                      <div className="landing-form-group">
                        <label htmlFor="landing-full-name">Full name</label>
                        <input
                          id="landing-full-name"
                          type="text"
                          value={form.full_name}
                          onChange={(event) =>
                            setForm((prev) => ({
                              ...prev,
                              full_name: event.target.value,
                            }))
                          }
                          placeholder="Enter your full name"
                          disabled={busy}
                          autoComplete="name"
                        />
                      </div>
                    </>
                  )}

                  <div className="landing-form-group">
                    <label htmlFor="landing-email">Email</label>
                    <input
                      id="landing-email"
                      type="email"
                      value={form.email}
                      onChange={(event) =>
                        setForm((prev) => ({ ...prev, email: event.target.value }))
                      }
                      placeholder="Enter your email"
                      disabled={busy}
                      autoComplete="email"
                    />
                  </div>

                  <div className="landing-form-group">
                    <label htmlFor="landing-password">Password</label>
                    <input
                      id="landing-password"
                      type="password"
                      value={form.password}
                      onChange={(event) =>
                        setForm((prev) => ({
                          ...prev,
                          password: event.target.value,
                        }))
                      }
                      placeholder="Enter your password"
                      disabled={busy}
                      autoComplete={isRegister ? "new-password" : "current-password"}
                    />
                  </div>

                  {error && <div className="landing-auth-error">{error}</div>}

                  <button type="submit" disabled={busy} className="landing-submit-btn">
                    {busy ? "Please wait..." : isRegister ? "Create Account" : "Sign In"}
                  </button>
                </form>
              </div>
            </aside>
          </div>
        </div>
      </section>

      <section className="landing-content-section" id="about">
        <div className="landing-container landing-section-grid">
          <div className="landing-section-copy">
            <span className="landing-section-kicker">About the platform</span>
            <h2>Designed to make intake feel immediate, calm, and structured.</h2>
            <p>
              This interface gives your symptom checker a clear starting point
              instead of dropping users directly into the application. It frames
              the experience, explains the workflow, and keeps account access
              visible from the first screen.
            </p>
          </div>

          <div className="landing-process-card">
            {steps.map((step) => (
              <div className="landing-process-row" key={step.number}>
                <span>{step.number}</span>
                <div>
                  <h3>{step.title}</h3>
                  <p>{step.description}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-content-section landing-content-section-alt" id="features">
        <div className="landing-container">
          <div className="landing-section-heading">
            <span className="landing-section-kicker">Features</span>
            <h2>Navigation now points to real landing-page sections.</h2>
          </div>

          <div className="landing-feature-grid">
            {features.map((feature) => (
              <article className="landing-feature-card" key={feature.title}>
                <span>{feature.eyebrow}</span>
                <h3>{feature.title}</h3>
                <p>{feature.description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="landing-content-section" id="contact">
        <div className="landing-container landing-contact-card">
          <div>
            <span className="landing-section-kicker">Contact</span>
            <h2>Ready to continue building the clinic experience.</h2>
            <p>
              Use the landing page as the public-facing entry point, then send
              authenticated users into the triage workspace you already have.
            </p>
          </div>

          <div className="landing-contact-actions">
            <button
              type="button"
              className="landing-primary-btn"
              onClick={() => scrollToSection("auth-panel", "register")}
            >
              Open Registration
            </button>
            <button
              type="button"
              className="landing-secondary-btn"
              onClick={() => scrollToSection("auth-panel", "login")}
            >
              Back to Sign In
            </button>
          </div>
        </div>
      </section>
    </div>
  );
}
