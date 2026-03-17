"""
Clinic Recommendation Service

Provides clinic and doctor recommendations based on diagnosis.
"""

import json
import math
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)


class ClinicService:
    """
    Manages clinic data and provides recommendations.
    """
    
    def __init__(self, clinics_config_path: str = None):
        """
        Initialize the clinic service.
        
        Args:
            clinics_config_path: Path to clinics configuration JSON
        """
        if clinics_config_path is None:
            clinics_config_path = Path(__file__).parent.parent / "config" / "clinics.json"
        
        self.clinics = []
        self.specialty_mapping = {}
        self.condition_to_symptom_mapping = {}
        self._load_clinics(clinics_config_path)
    
    def _load_clinics(self, path) -> None:
        """Load clinic data from JSON config."""
        try:
            if isinstance(path, str):
                path = Path(path)
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.clinics = data.get("clinics", [])
            self.specialty_mapping = data.get("specialty_mapping", {})
            self.condition_to_symptom_mapping = data.get("condition_to_symptom_mapping", {})
            
            logger.info(f"Loaded {len(self.clinics)} clinics from configuration")
        except Exception as e:
            logger.error(f"Error loading clinics configuration: {e}")
            self.clinics = []
            self.specialty_mapping = {}
            self.condition_to_symptom_mapping = {}
    
    def get_recommended_specialties(self, conditions: List[str]) -> List[str]:
        """
        Get recommended specialties based on conditions.
        
        Args:
            conditions: List of condition names from diagnosis
            
        Returns:
            List of recommended specialty codes
        """
        specialties = set()
        
        for condition in conditions:
            # Normalize condition name
            condition_normalized = condition.lower().replace(" ", "_").replace("-", "_")
            
            # Try direct mapping from condition
            if condition_normalized in self.condition_to_symptom_mapping:
                symptoms = self.condition_to_symptom_mapping[condition_normalized]
                for symptom in symptoms:
                    if symptom in self.specialty_mapping:
                        specialties.update(self.specialty_mapping[symptom])
            
            # Try symptom mapping
            if condition_normalized in self.specialty_mapping:
                specialties.update(self.specialty_mapping[condition_normalized])
            
            # Try partial matching for common conditions
            condition_lower = condition.lower()
            for key, values in self.specialty_mapping.items():
                if key in condition_lower or condition_lower in key:
                    specialties.update(values)
        
        # Always include general practice as fallback
        if not specialties:
            specialties.add("general")
        
        return list(specialties)
    
    def find_clinics_by_specialty(
        self, 
        specialties: List[str],
        user_location: Optional[Dict] = None,
        limit: int = 5
    ) -> List[Dict]:
        """
        Find clinics matching the required specialties.
        
        Args:
            specialties: List of required specialty codes
            user_location: Optional user location with lat/lng
            limit: Maximum number of clinics to return
            
        Returns:
            List of matching clinics with distance info
        """
        matching_clinics = []
        
        for clinic in self.clinics:
            clinic_specialties = set(clinic.get("specialties", []))
            matching = clinic_specialties.intersection(set(specialties))
            
            if matching:
                # Find matching doctors
                matching_doctors = []
                for doctor in clinic.get("doctors", []):
                    doc_specialties = set(doctor.get("specialty_codes", []))
                    if doc_specialties.intersection(set(specialties)):
                        matching_doctors.append(doctor)
                
                clinic_data = {
                    **clinic,
                    "matching_specialties": list(matching),
                    "matching_doctors": matching_doctors,
                    "distance": self._calculate_distance(
                        user_location, 
                        clinic.get("location", {})
                    ) if user_location else None
                }
                matching_clinics.append(clinic_data)
        
        # Sort by distance if available, otherwise by rating
        if user_location:
            matching_clinics.sort(key=lambda x: x.get("distance", float('inf')))
        else:
            matching_clinics.sort(key=lambda x: x.get("rating", 0), reverse=True)
        
        return matching_clinics[:limit]
    
    def _calculate_distance(self, loc1: Optional[Dict], loc2: Dict) -> float:
        """
        Calculate distance between two points in km using Haversine formula.
        
        Args:
            loc1: User location with coordinates [lng, lat]
            loc2: Clinic location with coordinates [lng, lat]
            
        Returns:
            Distance in kilometers
        """
        if not loc1 or not loc2:
            return float('inf')
        
        # Handle different location formats
        if "lat" in loc1 and "lng" in loc1:
            lat1, lon1 = loc1["lat"], loc1["lng"]
        elif "coordinates" in loc1 and len(loc1["coordinates"]) >= 2:
            lon1, lat1 = loc1["coordinates"][0], loc1["coordinates"][1]
        else:
            return float('inf')
        
        coords2 = loc2.get("coordinates", [])
        if len(coords2) < 2:
            return float('inf')
        
        lon2, lat2 = coords2[0], coords2[1]
        
        # Haversine formula
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371 * c  # Earth radius in km
    
    def get_clinic_by_id(self, clinic_id: str) -> Optional[Dict]:
        """
        Get a clinic by its ID.
        
        Args:
            clinic_id: Clinic identifier
            
        Returns:
            Clinic data or None if not found
        """
        for clinic in self.clinics:
            if clinic.get("id") == clinic_id:
                return clinic
        return None
    
    def get_doctor_by_id(self, clinic_id: str, doctor_id: str) -> Optional[Dict]:
        """
        Get a doctor by clinic and doctor ID.
        
        Args:
            clinic_id: Clinic identifier
            doctor_id: Doctor identifier
            
        Returns:
            Doctor data or None if not found
        """
        clinic = self.get_clinic_by_id(clinic_id)
        if not clinic:
            return None
        
        for doctor in clinic.get("doctors", []):
            if doctor.get("id") == doctor_id:
                return doctor
        return None
    
    def get_available_slots(
        self, 
        clinic_id: str, 
        doctor_id: str, 
        date: str,
        existing_bookings: List[Dict] = None
    ) -> List[str]:
        """
        Get available time slots for a doctor on a specific date.
        
        Args:
            clinic_id: Clinic identifier
            doctor_id: Doctor identifier
            date: Date in YYYY-MM-DD format
            existing_bookings: List of existing bookings to exclude
            
        Returns:
            List of available time slots in HH:MM format
        """
        clinic = self.get_clinic_by_id(clinic_id)
        doctor = self.get_doctor_by_id(clinic_id, doctor_id)
        
        if not clinic or not doctor:
            return []
        
        # Parse date to get day of week
        try:
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            day_name = date_obj.strftime("%A").lower()
        except ValueError:
            return []
        
        # Check if doctor works on this day
        if day_name not in doctor.get("available_days", []):
            return []
        
        # Get operating hours
        is_weekend = date_obj.weekday() >= 5
        hours_key = "weekend" if is_weekend else "weekday"
        operating_hours = clinic.get("operating_hours", {}).get(hours_key, {})
        
        open_time = operating_hours.get("open")
        close_time = operating_hours.get("close")
        
        if not open_time or not close_time:
            return []
        
        # Generate slots
        consultation_duration = doctor.get("consultation_duration_minutes", 30)
        slots = self._generate_time_slots(open_time, close_time, consultation_duration)
        
        # Remove already booked slots
        if existing_bookings:
            booked_times = set()
            for booking in existing_bookings:
                if (booking.get("clinic_id") == clinic_id and 
                    booking.get("doctor_id") == doctor_id and
                    booking.get("date") == date):
                    booked_times.add(booking.get("time"))
            
            slots = [s for s in slots if s not in booked_times]
        
        return slots
    
    def _generate_time_slots(
        self, 
        start_time: str, 
        end_time: str, 
        duration_minutes: int
    ) -> List[str]:
        """
        Generate time slots between start and end times.
        
        Args:
            start_time: Start time in HH:MM format
            end_time: End time in HH:MM format
            duration_minutes: Duration of each slot in minutes
            
        Returns:
            List of time slots in HH:MM format
        """
        slots = []
        
        try:
            start = datetime.strptime(start_time, "%H:%M")
            end = datetime.strptime(end_time, "%H:%M")
            
            # Add buffer for lunch break (12:00 - 13:00)
            lunch_start = datetime.strptime("12:00", "%H:%M")
            lunch_end = datetime.strptime("13:00", "%H:%M")
            
            current = start
            while current + timedelta(minutes=duration_minutes) <= end:
                slot_time = current.strftime("%H:%M")
                
                # Skip lunch break
                if current >= lunch_start and current < lunch_end:
                    current += timedelta(minutes=15)
                    continue
                
                slots.append(slot_time)
                current += timedelta(minutes=duration_minutes)
                
        except ValueError as e:
            logger.error(f"Error generating time slots: {e}")
            return []
        
        return slots
    
    def get_all_clinics(self) -> List[Dict]:
        """Get all clinics."""
        return self.clinics
    
    def get_all_specialties(self) -> List[str]:
        """Get all available specialties."""
        specialties = set()
        for clinic in self.clinics:
            specialties.update(clinic.get("specialties", []))
        return list(specialties)


# Singleton instance
_clinic_service = None


def get_clinic_service() -> ClinicService:
    """Get or create the singleton ClinicService instance."""
    global _clinic_service
    if _clinic_service is None:
        _clinic_service = ClinicService()
    return _clinic_service
