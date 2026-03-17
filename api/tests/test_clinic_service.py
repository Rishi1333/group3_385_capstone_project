"""
Tests for Clinic Service

Tests the clinic recommendation and booking functionality.
"""

import os
import sys
import pytest
import json
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.clinic_service import ClinicService, get_clinic_service


class TestClinicService:
    """Tests for the ClinicService class."""
    
    @pytest.fixture
    def clinic_service(self):
        """Create a ClinicService instance for testing."""
        # Use the actual clinics.json config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "clinics.json"
        )
        return ClinicService(clinics_config_path=config_path)
    
    def test_clinic_service_creation(self, clinic_service):
        """Test creating a clinic service instance."""
        assert clinic_service is not None
        assert len(clinic_service.clinics) > 0
        assert len(clinic_service.specialty_mapping) > 0
    
    def test_get_recommended_specialties_headache(self, clinic_service):
        """Test specialty recommendation for headache."""
        specialties = clinic_service.get_recommended_specialties(["Headache"])
        assert "general" in specialties or "neurology" in specialties
    
    def test_get_recommended_specialties_chest_pain(self, clinic_service):
        """Test specialty recommendation for chest pain."""
        specialties = clinic_service.get_recommended_specialties(["Chest Pain"])
        assert "cardiology" in specialties or "emergency" in specialties
    
    def test_get_recommended_specialties_multiple_conditions(self, clinic_service):
        """Test specialty recommendation for multiple conditions."""
        specialties = clinic_service.get_recommended_specialties([
            "Headache",
            "Anxiety"
        ])
        # Should include specialties for both conditions
        assert len(specialties) >= 1
    
    def test_get_recommended_specialties_empty(self, clinic_service):
        """Test specialty recommendation with empty conditions."""
        specialties = clinic_service.get_recommended_specialties([])
        assert "general" in specialties  # Should default to general
    
    def test_find_clinics_by_specialty(self, clinic_service):
        """Test finding clinics by specialty."""
        clinics = clinic_service.find_clinics_by_specialty(["cardiology"])
        assert len(clinics) > 0
        # All returned clinics should have cardiology
        for clinic in clinics:
            assert "cardiology" in clinic.get("specialties", [])
    
    def test_find_clinics_by_specialty_with_location(self, clinic_service):
        """Test finding clinics by specialty with user location."""
        user_location = {"lat": 43.6532, "lng": -79.3832}  # Toronto
        
        clinics = clinic_service.find_clinics_by_specialty(
            ["general"],
            user_location=user_location
        )
        
        assert len(clinics) > 0
        # Check that distance is calculated
        for clinic in clinics:
            if clinic.get("distance") is not None:
                assert clinic["distance"] >= 0
    
    def test_find_clinics_limit(self, clinic_service):
        """Test that clinic results are limited."""
        clinics = clinic_service.find_clinics_by_specialty(
            ["general"],
            limit=2
        )
        assert len(clinics) <= 2
    
    def test_get_clinic_by_id(self, clinic_service):
        """Test getting a clinic by ID."""
        clinic = clinic_service.get_clinic_by_id("clinic-001")
        assert clinic is not None
        assert clinic.get("id") == "clinic-001"
    
    def test_get_clinic_by_id_not_found(self, clinic_service):
        """Test getting a non-existent clinic."""
        clinic = clinic_service.get_clinic_by_id("nonexistent")
        assert clinic is None
    
    def test_get_doctor_by_id(self, clinic_service):
        """Test getting a doctor by ID."""
        doctor = clinic_service.get_doctor_by_id("clinic-001", "doc-001")
        assert doctor is not None
        assert doctor.get("id") == "doc-001"
    
    def test_get_doctor_by_id_not_found(self, clinic_service):
        """Test getting a non-existent doctor."""
        doctor = clinic_service.get_doctor_by_id("clinic-001", "nonexistent")
        assert doctor is None
    
    def test_get_available_slots(self, clinic_service):
        """Test getting available time slots."""
        # Get a valid date (tomorrow)
        from datetime import datetime, timedelta
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        slots = clinic_service.get_available_slots(
            "clinic-001",
            "doc-001",
            tomorrow
        )
        
        # Should have some slots available
        assert isinstance(slots, list)
    
    def test_get_available_slots_invalid_clinic(self, clinic_service):
        """Test getting slots for invalid clinic."""
        from datetime import datetime, timedelta
        tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        
        slots = clinic_service.get_available_slots(
            "nonexistent",
            "doc-001",
            tomorrow
        )
        
        assert slots == []
    
    def test_get_available_slots_invalid_date(self, clinic_service):
        """Test getting slots with invalid date format."""
        slots = clinic_service.get_available_slots(
            "clinic-001",
            "doc-001",
            "invalid-date"
        )
        
        assert slots == []
    
    def test_calculate_distance(self, clinic_service):
        """Test distance calculation."""
        # Toronto to North York (approximately 15 km)
        loc1 = {"lat": 43.6532, "lng": -79.3832}
        loc2 = {"coordinates": [-79.4163, 43.7543]}
        
        distance = clinic_service._calculate_distance(loc1, loc2)
        
        assert distance > 0
        assert distance < 50  # Should be less than 50 km
    
    def test_calculate_distance_same_location(self, clinic_service):
        """Test distance calculation for same location."""
        loc1 = {"lat": 43.6532, "lng": -79.3832}
        loc2 = {"coordinates": [-79.3832, 43.6532]}
        
        distance = clinic_service._calculate_distance(loc1, loc2)
        
        assert distance < 1  # Should be very close to 0
    
    def test_calculate_distance_missing_location(self, clinic_service):
        """Test distance calculation with missing location."""
        distance = clinic_service._calculate_distance(None, {"coordinates": [-79.3832, 43.6532]})
        assert distance == float('inf')
    
    def test_get_all_clinics(self, clinic_service):
        """Test getting all clinics."""
        clinics = clinic_service.get_all_clinics()
        assert len(clinics) > 0
    
    def test_get_all_specialties(self, clinic_service):
        """Test getting all specialties."""
        specialties = clinic_service.get_all_specialties()
        assert len(specialties) > 0
        assert "general" in specialties


class TestClinicServiceSingleton:
    """Tests for the singleton pattern."""
    
    def test_get_clinic_service_singleton(self):
        """Test that get_clinic_service returns a singleton."""
        service1 = get_clinic_service()
        service2 = get_clinic_service()
        
        assert service1 is service2


class TestClinicServiceWithMockConfig:
    """Tests with mock clinic configuration."""
    
    @pytest.fixture
    def mock_clinic_service(self, tmp_path):
        """Create a clinic service with mock config."""
        mock_config = {
            "clinics": [
                {
                    "id": "test-clinic-1",
                    "name": "Test Clinic",
                    "address": "123 Test St",
                    "location": {
                        "type": "Point",
                        "coordinates": [-79.3832, 43.6532]
                    },
                    "phone": "555-1234",
                    "specialties": ["general", "cardiology"],
                    "doctors": [
                        {
                            "id": "test-doc-1",
                            "name": "Dr. Test",
                            "specialty": "General Practice",
                            "specialty_codes": ["general"],
                            "available_days": ["monday", "tuesday"],
                            "consultation_duration_minutes": 30
                        }
                    ],
                    "operating_hours": {
                        "weekday": {"open": "09:00", "close": "17:00"},
                        "weekend": {"open": None, "close": None}
                    }
                }
            ],
            "specialty_mapping": {
                "headache": ["general", "neurology"],
                "chest_pain": ["cardiology", "emergency"]
            },
            "condition_to_symptom_mapping": {
                "Tension Headache": ["headache"]
            }
        }
        
        config_file = tmp_path / "clinics.json"
        config_file.write_text(json.dumps(mock_config))
        
        return ClinicService(clinics_config_path=str(config_file))
    
    def test_mock_clinic_loading(self, mock_clinic_service):
        """Test that mock clinics are loaded correctly."""
        assert len(mock_clinic_service.clinics) == 1
        assert mock_clinic_service.clinics[0]["id"] == "test-clinic-1"
    
    def test_mock_specialty_mapping(self, mock_clinic_service):
        """Test specialty mapping with mock config."""
        specialties = mock_clinic_service.get_recommended_specialties(["Tension Headache"])
        assert "general" in specialties
    
    def test_mock_find_clinics(self, mock_clinic_service):
        """Test finding clinics with mock config."""
        clinics = mock_clinic_service.find_clinics_by_specialty(["general"])
        assert len(clinics) == 1
        assert clinics[0]["id"] == "test-clinic-1"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
