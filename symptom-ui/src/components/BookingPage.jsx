/**
 * BookingPage Component
 *
 * Allows patients to book appointments with recommended doctors.
 */

import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import "./BookingPage.css";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:5000";

// Icons
function LocationIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
      <circle cx="12" cy="10" r="3"></circle>
    </svg>
  );
}

function CalendarIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
      <line x1="16" y1="2" x2="16" y2="6"></line>
      <line x1="8" y1="2" x2="8" y2="6"></line>
      <line x1="3" y1="10" x2="21" y2="10"></line>
    </svg>
  );
}

function ClockIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <circle cx="12" cy="12" r="10"></circle>
      <polyline points="12 6 12 12 16 14"></polyline>
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  );
}

function ArrowLeftIcon() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
    >
      <line x1="19" y1="12" x2="5" y2="12"></line>
      <polyline points="12 19 5 12 12 5"></polyline>
    </svg>
  );
}

function BookingPage() {
  const navigate = useNavigate();
  const location = useLocation();

  // Get diagnosis data from navigation state
  const { conditions, reportId } = location.state || {};

  // State
  const [step, setStep] = useState(1); // 1: Location, 2: Clinic, 3: DateTime, 4: Confirm, 5: Success
  const [userLocation, setUserLocation] = useState(null);
  // Location input for manual entry (future use)
  const [clinics, setClinics] = useState([]);
  const [selectedClinic, setSelectedClinic] = useState(null);
  const [selectedDoctor, setSelectedDoctor] = useState(null);
  const [selectedDate, setSelectedDate] = useState("");
  const [selectedTime, setSelectedTime] = useState("");
  const [availableSlots, setAvailableSlots] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [bookingConfirmed, setBookingConfirmed] = useState(null);
  const [recommendationMessage, setRecommendationMessage] = useState("");

  // Get auth token
  const getAuthHeaders = () => {
    const auth = localStorage.getItem("virtual_clinic_auth");
    if (auth) {
      const { token } = JSON.parse(auth);
      return { Authorization: `Bearer ${token}` };
    }
    return {};
  };

  // Get user's geolocation
  useEffect(() => {
    if (navigator.geolocation) {
      navigator.geolocation.getCurrentPosition(
        (position) => {
          setUserLocation({
            lat: position.coords.latitude,
            lng: position.coords.longitude,
          });
        },
        (err) => {
          console.log("Geolocation error:", err);
        },
      );
    }
  }, []);

  // Fetch recommended clinics when component mounts
  useEffect(() => {
    if (!conditions || conditions.length === 0) {
      // If no conditions, just fetch all clinics
      fetchAllClinics();
      return;
    }

    fetchRecommendedClinics();
  }, [conditions, userLocation]);

  const fetchRecommendedClinics = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/api/booking/recommend`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conditions,
          location: userLocation,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setClinics(data.clinics || []);
        setRecommendationMessage(data.message || "");
      } else {
        setError(data.error || "Failed to get recommendations");
      }
    } catch {
      setError("Failed to load clinic recommendations");
    } finally {
      setLoading(false);
    }
  };

  const fetchAllClinics = async () => {
    setLoading(true);
    setError("");
    try {
      const params = new URLSearchParams();
      if (userLocation) {
        params.append("lat", userLocation.lat);
        params.append("lng", userLocation.lng);
      }

      const response = await fetch(`${API_BASE}/api/booking/clinics?${params}`);
      const data = await response.json();

      if (response.ok) {
        setClinics(data.clinics || []);
      } else {
        setError(data.error || "Failed to load clinics");
      }
    } catch {
      setError("Failed to load clinics");
    } finally {
      setLoading(false);
    }
  };

  // Fetch available slots when date is selected
  useEffect(() => {
    if (!selectedClinic || !selectedDoctor || !selectedDate) return;

    const fetchSlots = async () => {
      setLoading(true);
      try {
        const params = new URLSearchParams({
          clinic_id: selectedClinic.id,
          doctor_id: selectedDoctor.id,
          date: selectedDate,
        });

        const response = await fetch(`${API_BASE}/api/booking/slots?${params}`);
        const data = await response.json();

        if (response.ok) {
          setAvailableSlots(data.available_slots || []);
        } else {
          setAvailableSlots([]);
        }
      } catch {
        setAvailableSlots([]);
      } finally {
        setLoading(false);
      }
    };

    fetchSlots();
  }, [selectedClinic, selectedDoctor, selectedDate]);

  // Handle booking submission
  const handleConfirmBooking = async () => {
    if (!selectedClinic || !selectedDoctor || !selectedDate || !selectedTime) {
      setError("Please select all required fields");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE}/api/booking/book`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...getAuthHeaders(),
        },
        body: JSON.stringify({
          clinic_id: selectedClinic.id,
          doctor_id: selectedDoctor.id,
          date: selectedDate,
          time: selectedTime,
          reason: conditions?.join(", ") || "General consultation",
          report_id: reportId,
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setBookingConfirmed(data.booking);
        setStep(5); // Success step
      } else {
        setError(data.error || "Booking failed");
      }
    } catch {
      setError("Failed to create booking. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  // Generate next 14 days for date selection
  const getAvailableDates = () => {
    const dates = [];
    const today = new Date();

    for (let i = 1; i <= 14; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dates.push({
        value: date.toISOString().split("T")[0],
        label: date.toLocaleDateString("en-US", {
          weekday: "short",
          month: "short",
          day: "numeric",
        }),
      });
    }

    return dates;
  };

  // Format time for display
  const formatTime = (time) => {
    const [hours, minutes] = time.split(":");
    const hour = parseInt(hours);
    const ampm = hour >= 12 ? "PM" : "AM";
    const displayHour = hour % 12 || 12;
    return `${displayHour}:${minutes} ${ampm}`;
  };

  // Render step content
  const renderStepContent = () => {
    switch (step) {
      case 1:
        return (
          <div className="step-content">
            <h2>Find a Clinic</h2>
            {recommendationMessage && (
              <div className="recommendation-message">
                <p>{recommendationMessage}</p>
              </div>
            )}

            <div className="location-section">
              <h3>Your Location</h3>
              {userLocation ? (
                <p className="location-detected">
                  <LocationIcon /> Location detected automatically
                </p>
              ) : (
                <p className="location-manual">
                  Enter your location or use automatic detection
                </p>
              )}

              <div className="location-actions">
                <button
                  className="btn-secondary"
                  onClick={() => {
                    if (navigator.geolocation) {
                      navigator.geolocation.getCurrentPosition((position) => {
                        setUserLocation({
                          lat: position.coords.latitude,
                          lng: position.coords.longitude,
                        });
                      });
                    }
                  }}
                >
                  <LocationIcon /> Detect Location
                </button>
              </div>
            </div>

            {loading ? (
              <div className="loading">Loading clinics...</div>
            ) : clinics.length > 0 ? (
              <div className="clinics-list">
                <h3>Recommended Clinics</h3>
                {clinics.map((clinic) => (
                  <div key={clinic.id} className="clinic-card">
                    <div className="clinic-info">
                      <h4>{clinic.name}</h4>
                      <p className="clinic-address">{clinic.address}</p>
                      <p className="clinic-phone">{clinic.phone}</p>
                      {clinic.distance && (
                        <p className="clinic-distance">
                          {clinic.distance.toFixed(1)} km away
                        </p>
                      )}
                      <div className="clinic-specialties">
                        {clinic.matching_specialties?.map((s) => (
                          <span key={s} className="specialty-tag">
                            {s}
                          </span>
                        ))}
                      </div>
                      <div className="clinic-rating">
                        ⭐ {clinic.rating || "N/A"}
                      </div>
                    </div>
                    <button
                      className="btn-primary"
                      onClick={() => {
                        setSelectedClinic(clinic);
                        setStep(2);
                      }}
                    >
                      Select
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <p className="no-results">
                No clinics found. Try adjusting your location.
              </p>
            )}
          </div>
        );

      case 2:
        return (
          <div className="step-content">
            <button className="btn-back" onClick={() => setStep(1)}>
              <ArrowLeftIcon /> Back to Clinics
            </button>

            <h2>Select a Doctor</h2>
            <div className="selected-clinic">
              <h3>{selectedClinic?.name}</h3>
              <p>{selectedClinic?.address}</p>
            </div>

            <div className="doctors-list">
              {selectedClinic?.matching_doctors?.length > 0
                ? selectedClinic.matching_doctors.map((doctor) => (
                    <div key={doctor.id} className="doctor-card">
                      <div className="doctor-info">
                        <h4>{doctor.name}</h4>
                        <p className="doctor-specialty">{doctor.specialty}</p>
                        <p className="doctor-availability">
                          Available: {doctor.available_days?.join(", ")}
                        </p>
                      </div>
                      <button
                        className="btn-primary"
                        onClick={() => {
                          setSelectedDoctor(doctor);
                          setStep(3);
                        }}
                      >
                        Select
                      </button>
                    </div>
                  ))
                : selectedClinic?.doctors?.map((doctor) => (
                    <div key={doctor.id} className="doctor-card">
                      <div className="doctor-info">
                        <h4>{doctor.name}</h4>
                        <p className="doctor-specialty">{doctor.specialty}</p>
                        <p className="doctor-availability">
                          Available: {doctor.available_days?.join(", ")}
                        </p>
                      </div>
                      <button
                        className="btn-primary"
                        onClick={() => {
                          setSelectedDoctor(doctor);
                          setStep(3);
                        }}
                      >
                        Select
                      </button>
                    </div>
                  ))}
            </div>
          </div>
        );

      case 3:
        return (
          <div className="step-content">
            <button className="btn-back" onClick={() => setStep(2)}>
              <ArrowLeftIcon /> Back to Doctors
            </button>

            <h2>Select Date & Time</h2>

            <div className="selection-summary">
              <div className="summary-item">
                <strong>Clinic:</strong> {selectedClinic?.name}
              </div>
              <div className="summary-item">
                <strong>Doctor:</strong> {selectedDoctor?.name}
              </div>
            </div>

            <div className="datetime-section">
              <div className="date-section">
                <h3>
                  <CalendarIcon /> Select Date
                </h3>
                <div className="date-grid">
                  {getAvailableDates().map((date) => (
                    <button
                      key={date.value}
                      className={`date-btn ${selectedDate === date.value ? "selected" : ""}`}
                      onClick={() => setSelectedDate(date.value)}
                    >
                      {date.label}
                    </button>
                  ))}
                </div>
              </div>

              {selectedDate && (
                <div className="time-section">
                  <h3>
                    <ClockIcon /> Select Time
                  </h3>
                  {loading ? (
                    <p>Loading available times...</p>
                  ) : availableSlots.length > 0 ? (
                    <div className="time-grid">
                      {availableSlots.map((slot) => (
                        <button
                          key={slot}
                          className={`time-btn ${selectedTime === slot ? "selected" : ""}`}
                          onClick={() => setSelectedTime(slot)}
                        >
                          {formatTime(slot)}
                        </button>
                      ))}
                    </div>
                  ) : (
                    <p className="no-slots">
                      No available slots for this date. Please select another
                      date.
                    </p>
                  )}
                </div>
              )}
            </div>

            {selectedDate && selectedTime && (
              <button
                className="btn-primary btn-continue"
                onClick={() => setStep(4)}
              >
                Continue to Confirmation
              </button>
            )}
          </div>
        );

      case 4:
        return (
          <div className="step-content confirmation-step">
            <button className="btn-back" onClick={() => setStep(3)}>
              <ArrowLeftIcon /> Back
            </button>

            <h2>Confirm Your Booking</h2>

            <div className="confirmation-details">
              <div className="detail-card">
                <h3>Appointment Details</h3>
                <div className="detail-row">
                  <span className="label">Clinic:</span>
                  <span className="value">{selectedClinic?.name}</span>
                </div>
                <div className="detail-row">
                  <span className="label">Address:</span>
                  <span className="value">{selectedClinic?.address}</span>
                </div>
                <div className="detail-row">
                  <span className="label">Doctor:</span>
                  <span className="value">{selectedDoctor?.name}</span>
                </div>
                <div className="detail-row">
                  <span className="label">Specialty:</span>
                  <span className="value">{selectedDoctor?.specialty}</span>
                </div>
                <div className="detail-row">
                  <span className="label">Date:</span>
                  <span className="value">
                    {new Date(selectedDate).toLocaleDateString("en-US", {
                      weekday: "long",
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </span>
                </div>
                <div className="detail-row">
                  <span className="label">Time:</span>
                  <span className="value">{formatTime(selectedTime)}</span>
                </div>
                {conditions && conditions.length > 0 && (
                  <div className="detail-row">
                    <span className="label">Reason:</span>
                    <span className="value">{conditions.join(", ")}</span>
                  </div>
                )}
              </div>

              <div className="important-notes">
                <h4>Important Notes</h4>
                <ul>
                  <li>Please arrive 10 minutes before your appointment</li>
                  <li>Bring your ID and insurance card (if applicable)</li>
                  <li>You will receive a confirmation email</li>
                </ul>
              </div>
            </div>

            {error && <div className="error-message">{error}</div>}

            <button
              className="btn-primary btn-confirm"
              onClick={handleConfirmBooking}
              disabled={loading}
            >
              {loading ? "Booking..." : "Confirm Booking"}
            </button>
          </div>
        );

      case 5:
        return (
          <div className="step-content success-step">
            <div className="success-icon">
              <CheckIcon />
            </div>
            <h2>Booking Confirmed!</h2>

            <div className="booking-details">
              <p className="booking-id">
                Booking ID: <strong>{bookingConfirmed?.booking_id}</strong>
              </p>

              <div className="confirmed-info">
                <p>
                  <strong>{bookingConfirmed?.doctor_name}</strong>
                </p>
                <p>{bookingConfirmed?.doctor_specialty}</p>
                <p>{bookingConfirmed?.clinic_name}</p>
                <p>
                  {new Date(bookingConfirmed?.date).toLocaleDateString(
                    "en-US",
                    {
                      weekday: "long",
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    },
                  )}
                </p>
                <p>at {formatTime(bookingConfirmed?.time)}</p>
              </div>
            </div>

            <div className="success-actions">
              <button className="btn-primary" onClick={() => navigate("/")}>
                Return to Home
              </button>
              <button
                className="btn-secondary"
                onClick={() => navigate("/bookings")}
              >
                View My Bookings
              </button>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  return (
    <div className="booking-page">
      <div className="booking-container">
        {/* Progress indicator */}
        {step < 5 && (
          <div className="progress-bar">
            <div className={`progress-step ${step >= 1 ? "active" : ""}`}>
              <span className="step-number">1</span>
              <span className="step-label">Clinic</span>
            </div>
            <div className="progress-line"></div>
            <div className={`progress-step ${step >= 2 ? "active" : ""}`}>
              <span className="step-number">2</span>
              <span className="step-label">Doctor</span>
            </div>
            <div className="progress-line"></div>
            <div className={`progress-step ${step >= 3 ? "active" : ""}`}>
              <span className="step-number">3</span>
              <span className="step-label">Date & Time</span>
            </div>
            <div className="progress-line"></div>
            <div className={`progress-step ${step >= 4 ? "active" : ""}`}>
              <span className="step-number">4</span>
              <span className="step-label">Confirm</span>
            </div>
          </div>
        )}

        {error && step !== 4 && <div className="error-message">{error}</div>}

        {renderStepContent()}
      </div>
    </div>
  );
}

export default BookingPage;
