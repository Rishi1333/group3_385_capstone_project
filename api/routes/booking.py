"""
Booking Routes

API endpoints for clinic booking functionality.
"""

import logging
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt, verify_jwt_in_request
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)


def create_booking_blueprint(clinic_service, db):
    """
    Create the booking blueprint.
    
    Args:
        clinic_service: ClinicService instance
        db: MongoDB database instance
        
    Returns:
        Flask blueprint
    """
    bp = Blueprint("booking", __name__, url_prefix="/api/booking")
    
    @bp.route("/recommend", methods=["POST"])
    def get_recommendations():
        """
        Get clinic recommendations based on diagnosis.
        
        Request body:
        {
            "conditions": ["Tension Headache", "Migraine"],
            "location": {"lat": 43.6532, "lng": -79.3832}  // optional
        }
        
        Returns:
        {
            "specialties": ["general", "neurology"],
            "clinics": [...],
            "message": "Based on your symptoms..."
        }
        """
        data = request.get_json() or {}
        conditions = data.get("conditions", [])
        user_location = data.get("location")
        
        if not conditions:
            return jsonify({"error": "conditions field is required"}), 400
        
        # Get recommended specialties
        specialties = clinic_service.get_recommended_specialties(conditions)
        
        # Find matching clinics
        clinics = clinic_service.find_clinics_by_specialty(
            specialties, 
            user_location
        )
        
        # Generate recommendation message
        if len(specialties) == 1:
            specialty_text = specialties[0].replace("_", " ").title()
            message = f"Based on your symptoms, we recommend consulting with a {specialty_text} specialist."
        else:
            specialty_list = [s.replace("_", " ").title() for s in specialties[:3]]
            message = f"Based on your symptoms, we recommend consulting with specialists in: {', '.join(specialty_list)}."
        
        return jsonify({
            "specialties": specialties,
            "clinics": clinics,
            "message": message
        }), 200
    
    @bp.route("/clinics", methods=["GET"])
    def search_clinics():
        """
        Search for clinics by location or specialty.
        
        Query params:
        - lat, lng: User coordinates
        - specialty: Filter by specialty
        - radius: Search radius in km (default: 25)
        """
        lat = request.args.get("lat", type=float)
        lng = request.args.get("lng", type=float)
        specialty = request.args.get("specialty")
        radius = request.args.get("radius", default=25, type=float)
        
        user_location = {"lat": lat, "lng": lng} if lat and lng else None
        
        if specialty:
            clinics = clinic_service.find_clinics_by_specialty([specialty], user_location)
        else:
            clinics = clinic_service.get_all_clinics()
            # Add distance if location provided
            if user_location:
                for clinic in clinics:
                    clinic["distance"] = clinic_service._calculate_distance(
                        user_location, 
                        clinic.get("location", {})
                    )
                clinics.sort(key=lambda x: x.get("distance", float('inf')))
        
        # Filter by radius
        if user_location and radius:
            clinics = [c for c in clinics 
                      if c.get("distance", float('inf')) <= radius]
        
        return jsonify({"clinics": clinics}), 200
    
    @bp.route("/clinics/<clinic_id>", methods=["GET"])
    def get_clinic(clinic_id):
        """Get details for a specific clinic."""
        clinic = clinic_service.get_clinic_by_id(clinic_id)
        
        if not clinic:
            return jsonify({"error": "Clinic not found"}), 404
        
        return jsonify({"clinic": clinic}), 200
    
    @bp.route("/slots", methods=["GET"])
    def get_available_slots():
        """
        Get available appointment slots.
        
        Query params:
        - clinic_id: Clinic identifier
        - doctor_id: Doctor identifier
        - date: Date in YYYY-MM-DD format
        """
        clinic_id = request.args.get("clinic_id")
        doctor_id = request.args.get("doctor_id")
        date = request.args.get("date")
        
        if not all([clinic_id, doctor_id, date]):
            return jsonify({"error": "Missing required parameters: clinic_id, doctor_id, date"}), 400
        
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        # Get existing bookings for this doctor on this date
        existing_bookings = list(db["bookings"].find(
            {
                "clinic_id": clinic_id,
                "doctor_id": doctor_id,
                "date": date,
                "status": {"$ne": "cancelled"}
            },
            {"_id": 0, "time": 1}
        ))
        
        slots = clinic_service.get_available_slots(
            clinic_id, doctor_id, date, existing_bookings
        )
        
        return jsonify({
            "clinic_id": clinic_id,
            "doctor_id": doctor_id,
            "date": date,
            "available_slots": slots
        }), 200
    
    @bp.route("/book", methods=["POST"])
    def create_booking():
        """
        Create a new booking.
        
        Request body:
        {
            "clinic_id": "clinic-001",
            "doctor_id": "doc-001",
            "date": "2024-01-15",
            "time": "10:00",
            "reason": "Follow-up for headache",
            "report_id": "RPT-20240101-123456"
        }
        
        Returns:
        {
            "message": "Booking confirmed successfully",
            "booking": {...}
        }
        """
        # Check authentication
        try:
            verify_jwt_in_request()
            claims = get_jwt()
        except Exception as e:
            logger.error(f"JWT verification failed: {e}")
            return jsonify({"error": "Authentication required"}), 401
        
        if claims.get("role") != "patients":
            return jsonify({"error": "Only patients can book appointments"}), 403
        
        email = claims.get("email")
        if not email:
            return jsonify({"error": "Invalid authentication token"}), 401
        
        data = request.get_json() or {}
        
        # Validate required fields
        required = ["clinic_id", "doctor_id", "date", "time"]
        missing = [f for f in required if not data.get(f)]
        if missing:
            return jsonify({"error": f"Missing required fields: {', '.join(missing)}"}), 400
        
        clinic_id = data["clinic_id"]
        doctor_id = data["doctor_id"]
        date = data["date"]
        time = data["time"]
        
        # Validate clinic and doctor exist
        clinic = clinic_service.get_clinic_by_id(clinic_id)
        if not clinic:
            return jsonify({"error": "Clinic not found"}), 404
        
        doctor = clinic_service.get_doctor_by_id(clinic_id, doctor_id)
        if not doctor:
            return jsonify({"error": "Doctor not found"}), 404
        
        # Validate date format
        try:
            appointment_date = datetime.strptime(date, "%Y-%m-%d")
            if appointment_date.date() < datetime.now().date():
                return jsonify({"error": "Cannot book appointments in the past"}), 400
        except ValueError:
            return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400
        
        # Check for existing booking at same time
        existing = db["bookings"].find_one({
            "clinic_id": clinic_id,
            "doctor_id": doctor_id,
            "date": date,
            "time": time,
            "status": {"$ne": "cancelled"}
        })
        
        if existing:
            return jsonify({"error": "This time slot is already booked"}), 409
        
        # Create booking record
        booking = {
            "booking_id": f"BKG-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
            "patient_email": email,
            "clinic_id": clinic_id,
            "clinic_name": clinic.get("name", ""),
            "doctor_id": doctor_id,
            "doctor_name": doctor.get("name", ""),
            "doctor_specialty": doctor.get("specialty", ""),
            "date": date,
            "time": time,
            "reason": data.get("reason", ""),
            "report_id": data.get("report_id"),
            "status": "confirmed",
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat()
        }
        
        try:
            # Save to bookings collection
            db["bookings"].insert_one(booking)
            
            # Also add to patient's record
            db["patients"].update_one(
                {"email": email},
                {
                    "$push": {"bookings": booking},
                    "$set": {"updated_at": datetime.utcnow().isoformat()}
                }
            )
            
            logger.info(f"Created booking {booking['booking_id']} for patient {email}")
            
        except Exception as e:
            logger.error(f"Error creating booking: {e}")
            return jsonify({"error": "Failed to create booking"}), 500
        
        # Remove MongoDB _id for response
        booking.pop("_id", None)
        
        return jsonify({
            "message": "Booking confirmed successfully",
            "booking": booking
        }), 201
    
    @bp.route("/my-bookings", methods=["GET"])
    def get_my_bookings():
        """Get all bookings for the current patient."""
        try:
            verify_jwt_in_request()
            claims = get_jwt()
        except Exception:
            return jsonify({"error": "Authentication required"}), 401
        
        email = claims.get("email")
        if not email:
            return jsonify({"error": "Invalid authentication token"}), 401
        
        # Get bookings from database
        bookings = list(db["bookings"].find(
            {"patient_email": email},
            {"_id": 0}
        ).sort("created_at", -1))
        
        return jsonify({"bookings": bookings}), 200
    
    @bp.route("/<booking_id>", methods=["GET"])
    def get_booking(booking_id):
        """Get details for a specific booking."""
        try:
            verify_jwt_in_request()
            claims = get_jwt()
        except Exception:
            return jsonify({"error": "Authentication required"}), 401
        
        email = claims.get("email")
        
        booking = db["bookings"].find_one(
            {"booking_id": booking_id},
            {"_id": 0}
        )
        
        if not booking:
            return jsonify({"error": "Booking not found"}), 404
        
        # Check ownership (patients can only see their own bookings)
        if claims.get("role") == "patients" and booking.get("patient_email") != email:
            return jsonify({"error": "Access denied"}), 403
        
        return jsonify({"booking": booking}), 200
    
    @bp.route("/<booking_id>/cancel", methods=["POST"])
    def cancel_booking(booking_id):
        """Cancel a booking."""
        try:
            verify_jwt_in_request()
            claims = get_jwt()
        except Exception:
            return jsonify({"error": "Authentication required"}), 401
        
        email = claims.get("email")
        
        booking = db["bookings"].find_one({"booking_id": booking_id})
        
        if not booking:
            return jsonify({"error": "Booking not found"}), 404
        
        # Check ownership
        if claims.get("role") == "patients" and booking.get("patient_email") != email:
            return jsonify({"error": "Access denied"}), 403
        
        # Check if already cancelled
        if booking.get("status") == "cancelled":
            return jsonify({"error": "Booking is already cancelled"}), 400
        
        # Update status
        db["bookings"].update_one(
            {"booking_id": booking_id},
            {
                "$set": {
                    "status": "cancelled",
                    "cancelled_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )
        
        return jsonify({
            "message": "Booking cancelled successfully",
            "booking_id": booking_id
        }), 200
    
    @bp.route("/specialties", methods=["GET"])
    def get_specialties():
        """Get all available specialties."""
        specialties = clinic_service.get_all_specialties()
        return jsonify({"specialties": specialties}), 200
    
    return bp
