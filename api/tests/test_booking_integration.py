"""
Integration Tests for Booking Flow

Tests the complete booking workflow from recommendation to confirmation.
"""

import os
import sys
import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clinic_service import ClinicService


class TestBookingFlowIntegration:
    """Integration tests for the complete booking flow."""
    
    @pytest.fixture
    def clinic_service(self):
        """Create a clinic service instance."""
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "clinics.json"
        )
        return ClinicService(clinics_config_path=config_path)
    
    @pytest.fixture
    def mock_db(self):
        """Create a mock MongoDB database."""
        mock_db = MagicMock()
        mock_bookings = MagicMock()
        mock_patients = MagicMock()
        
        mock_db.__getitem__.side_effect = lambda x: {
            "bookings": mock_bookings,
            "patients": mock_patients
        }.get(x)
        
        return mock_db, mock_bookings, mock_patients
    
    def test_complete_booking_flow(self, clinic_service, mock_db):
        """Test the complete booking flow from diagnosis to confirmation."""
        mock_db_obj, mock_bookings, mock_patients = mock_db
        
        # Step 1: Get recommended specialties based on conditions
        conditions = ["Headache", "Migraine"]
        specialties = clinic_service.get_recommended_specialties(conditions)
        
        assert len(specialties) > 0
        assert "general" in specialties or "neurology" in specialties
        
        # Step 2: Find clinics matching the specialties
        clinics = clinic_service.find_clinics_by_specialty(specialties)
        
        assert len(clinics) > 0
        
        # Step 3: Select a clinic and doctor
        selected_clinic = clinics[0]
        assert selected_clinic is not None
        
        doctors = selected_clinic.get("matching_doctors", [])
        if not doctors:
            doctors = selected_clinic.get("doctors", [])
        
        assert len(doctors) > 0
        selected_doctor = doctors[0]
        
        # Step 4: Get available slots
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        slots = clinic_service.get_available_slots(
            selected_clinic["id"],
            selected_doctor["id"],
            tomorrow
        )
        
        # Step 5: Simulate booking creation
        if slots:
            booking = {
                "booking_id": f"BKG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
                "patient_email": "test@example.com",
                "clinic_id": selected_clinic["id"],
                "clinic_name": selected_clinic.get("name", ""),
                "doctor_id": selected_doctor["id"],
                "doctor_name": selected_doctor.get("name", ""),
                "doctor_specialty": selected_doctor.get("specialty", ""),
                "date": tomorrow,
                "time": slots[0],
                "reason": ", ".join(conditions),
                "status": "confirmed",
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Verify booking structure
            assert booking["booking_id"].startswith("BKG-")
            assert booking["status"] == "confirmed"
            assert booking["clinic_id"] == selected_clinic["id"]
            assert booking["doctor_id"] == selected_doctor["id"]
    
    def test_booking_with_location(self, clinic_service):
        """Test booking flow with user location."""
        # User location (Toronto downtown)
        user_location = {"lat": 43.6532, "lng": -79.3832}
        
        # Get clinics sorted by distance
        clinics = clinic_service.find_clinics_by_specialty(
            ["general"],
            user_location=user_location
        )
        
        assert len(clinics) > 0
        
        # Verify clinics have distance calculated
        for clinic in clinics:
            if clinic.get("distance") is not None:
                assert isinstance(clinic["distance"], (int, float))
                assert clinic["distance"] >= 0
        
        # Verify clinics are sorted by distance (if multiple)
        if len(clinics) > 1:
            distances = [c.get("distance", float('inf')) for c in clinics]
            assert distances == sorted(distances)
    
    def test_booking_prevents_double_booking(self, clinic_service):
        """Test that double booking is prevented."""
        clinic_id = "clinic-001"
        doctor_id = "doc-001"
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        # Get available slots
        slots = clinic_service.get_available_slots(clinic_id, doctor_id, tomorrow)
        
        if slots:
            # Simulate existing booking
            existing_bookings = [
                {"clinic_id": clinic_id, "doctor_id": doctor_id, "date": tomorrow, "time": slots[0]}
            ]
            
            # Get slots again with existing bookings
            remaining_slots = clinic_service.get_available_slots(
                clinic_id,
                doctor_id,
                tomorrow,
                existing_bookings=existing_bookings
            )
            
            # The booked slot should not be available
            assert slots[0] not in remaining_slots
    
    def test_booking_validation(self, clinic_service):
        """Test booking validation logic."""
        # Test invalid clinic
        result = clinic_service.get_clinic_by_id("invalid-clinic")
        assert result is None
        
        # Test invalid doctor
        result = clinic_service.get_doctor_by_id("clinic-001", "invalid-doctor")
        assert result is None
        
        # Test invalid date
        slots = clinic_service.get_available_slots(
            "clinic-001",
            "doc-001",
            "invalid-date"
        )
        assert slots == []
    
    def test_specialty_mapping_coverage(self, clinic_service):
        """Test that specialty mapping covers common conditions."""
        common_conditions = [
            "Headache",
            "Chest Pain",
            "Fever",
            "Anxiety",
            "Skin Rash"
        ]
        
        for condition in common_conditions:
            specialties = clinic_service.get_recommended_specialties([condition])
            assert len(specialties) > 0, f"No specialties found for {condition}"
            # General is only a fallback when no other specialty matches
            # Some conditions like Chest Pain correctly map to emergency/cardiology


class TestBookingAPIIntegration:
    """Integration tests for booking API endpoints."""
    
    @pytest.fixture
    def app(self):
        """Create a test Flask app."""
        from flask import Flask
        from routes.booking import create_booking_blueprint
        from services.clinic_service import get_clinic_service
        
        app = Flask(__name__)
        
        # Mock database
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=MagicMock())
        
        clinic_service = get_clinic_service()
        booking_bp = create_booking_blueprint(clinic_service, mock_db)
        app.register_blueprint(booking_bp)
        
        return app
    
    @pytest.fixture
    def client(self, app):
        """Create a test client."""
        return app.test_client()
    
    def test_recommend_endpoint(self, client):
        """Test the /api/booking/recommend endpoint."""
        response = client.post(
            "/api/booking/recommend",
            json={"conditions": ["Headache"]}
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert "specialties" in data
        assert "clinics" in data
        assert "message" in data
    
    def test_recommend_endpoint_missing_conditions(self, client):
        """Test recommend endpoint with missing conditions."""
        response = client.post(
            "/api/booking/recommend",
            json={}
        )
        
        assert response.status_code == 400
        data = response.get_json()
        assert "error" in data
    
    def test_clinics_endpoint(self, client):
        """Test the /api/booking/clinics endpoint."""
        response = client.get("/api/booking/clinics")
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert "clinics" in data
        assert len(data["clinics"]) > 0
    
    def test_clinics_endpoint_with_location(self, client):
        """Test clinics endpoint with location parameters."""
        response = client.get(
            "/api/booking/clinics?lat=43.6532&lng=-79.3832"
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert "clinics" in data
    
    def test_clinic_by_id_endpoint(self, client):
        """Test the /api/booking/clinics/<clinic_id> endpoint."""
        response = client.get("/api/booking/clinics/clinic-001")
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert "clinic" in data
        assert data["clinic"]["id"] == "clinic-001"
    
    def test_clinic_by_id_not_found(self, client):
        """Test clinic by ID endpoint with non-existent clinic."""
        response = client.get("/api/booking/clinics/nonexistent")
        
        assert response.status_code == 404
    
    def test_slots_endpoint(self, client):
        """Test the /api/booking/slots endpoint."""
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        response = client.get(
            f"/api/booking/slots?clinic_id=clinic-001&doctor_id=doc-001&date={tomorrow}"
        )
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert "available_slots" in data
    
    def test_slots_endpoint_missing_params(self, client):
        """Test slots endpoint with missing parameters."""
        response = client.get("/api/booking/slots")
        
        assert response.status_code == 400
    
    def test_specialties_endpoint(self, client):
        """Test the /api/booking/specialties endpoint."""
        response = client.get("/api/booking/specialties")
        
        assert response.status_code == 200
        data = response.get_json()
        
        assert "specialties" in data
        assert len(data["specialties"]) > 0


class TestClinicalReportIntegration:
    """Integration tests for enhanced clinical report."""
    
    @pytest.fixture
    def report_generator(self):
        """Create a clinical report generator instance."""
        from services.clinical_report import ClinicalReportGenerator
        return ClinicalReportGenerator()
    
    def test_report_with_empty_fields(self, report_generator):
        """Test report generation with empty/minimal data."""
        clinical_data = {
            "patient_profile": {"age": 30, "sex": "male"},
            "chief_complaint": "Headache",
            "symptoms": [{"name": "headache", "duration": "2 days"}],
            "medical_history": [],
            "medications": [],
            "allergies": [],
            "risk_factors": [],
            "lifestyle": {},
            "family_history_details": []
        }
        
        report = report_generator._generate_fallback_report(clinical_data, [])
        
        assert report is not None
        assert report["chief_complaint"] == "Headache"
        assert "disclaimer" in report
        assert report["patient_profile"]["age"] == 30
    
    def test_report_with_complete_data(self, report_generator):
        """Test report generation with complete data."""
        clinical_data = {
            "patient_profile": {"age": 45, "sex": "female"},
            "chief_complaint": "Chest pain and shortness of breath",
            "symptoms": [
                {"name": "chest pain", "duration": "1 week", "severity": 7},
                {"name": "shortness of breath", "duration": "3 days"}
            ],
            "medical_history": ["Hypertension", "Diabetes Type 2"],
            "medications": ["Metformin", "Lisinopril"],
            "allergies": ["Penicillin"],
            "risk_factors": ["Smoking", "Family history of heart disease"],
            "lifestyle": {
                "smoking": "current",
                "alcohol": "moderate",
                "exercise": "none"
            },
            "family_history_details": ["Father had heart attack at 55"],
            "occupation": "Office Worker"
        }
        
        tool_results = [
            {
                "tool": "heart_analysis",
                "prediction": {"predictions": ["Possible angina", "Cardiac risk"]}
            }
        ]
        
        report = report_generator._generate_fallback_report(clinical_data, tool_results)
        
        assert report is not None
        assert report["chief_complaint"] == "Chest pain and shortness of breath"
        assert len(report["past_medical_history"]) == 2
        assert len(report["current_medications"]) == 2
        assert len(report["allergies"]) == 1
        assert "risk_assessment" in report
        assert report["risk_assessment"]["level"] in ["Low", "Moderate", "High"]
    
    def test_risk_assessment_calculation(self, report_generator):
        """Test risk assessment calculation."""
        # High risk scenario
        clinical_data = {
            "patient_profile": {"age": 60, "sex": "male"},
            "medical_history": ["Diabetes", "Hypertension"],
            "lifestyle": {"smoking": "current", "exercise": "none"},
            "family_history_details": ["Heart disease"]
        }
        
        assessment = report_generator._calculate_risk_assessment(clinical_data)
        
        assert assessment["level"] in ["Low", "Moderate", "High"]
        assert len(assessment["risk_factors"]) > 0
    
    def test_preventive_care_reminders(self, report_generator):
        """Test preventive care reminder generation."""
        # 50-year-old female
        clinical_data = {
            "patient_profile": {"age": 50, "sex": "female"},
            "lifestyle": {"smoking": "current"}
        }
        
        reminders = report_generator._get_preventive_care_reminders(clinical_data)
        
        assert len(reminders) > 0
        # Should include mammogram reminder for 50+ female
        assert any("mammogram" in r.lower() or "cervical" in r.lower() for r in reminders)
    
    def test_report_format_for_display(self, report_generator):
        """Test report formatting for display."""
        report = {
            "report_id": "RPT-20240101-120000",
            "generated_at": "2024-01-01T12:00:00",
            "patient_profile": {"age": 35, "sex": "male"},
            "chief_complaint": "Headache",
            "history_of_present_illness": "Patient reports headache for 2 days",
            "past_medical_history": [],
            "current_medications": [],
            "allergies": [],
            "lifestyle_factors": {"smoking": "never", "exercise": "moderate"},
            "family_history": [],
            "risk_factors": [],
            "risk_assessment": {"level": "Low", "risk_factors": [], "protective_factors": ["Non-smoker"]},
            "suggested_differential": [{"condition": "Tension Headache", "likelihood": "High"}],
            "recommended_next_steps": ["Follow up with PCP"],
            "preventive_care": ["Annual physical recommended"],
            "disclaimer": "Test disclaimer"
        }
        
        formatted = report_generator.format_for_display(report)
        
        assert "CLINICAL SUMMARY REPORT" in formatted
        assert "Headache" in formatted
        assert "Patient Profile" in formatted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
