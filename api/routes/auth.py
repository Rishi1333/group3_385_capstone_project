from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt
import bcrypt
from bson import ObjectId
from pymongo.errors import DuplicateKeyError

from mongo_db import get_db

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


def _collection_for_role(role: str):
    db = get_db()
    r = (role or "").lower().strip()

    if r in ("admin", "administrator"):
        return db["administrator"], "administrator"
    if r in ("doctor", "doctors"):
        return db["doctors"], "doctors"
    if r in ("patient", "patients"):
        return db["patients"], "patients"

    return None, None


def _hash_password(password: str) -> bytes:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())


def _check_password(password: str, hashed: bytes) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed)


def _find_users_by_email(email: str) -> list[tuple[dict, str]]:
    """Search all role collections for a matching email."""
    normalized = (email or "").lower().strip()
    if not normalized:
        return []

    db = get_db()
    matches = []
    role_map = (
        ("administrator", "administrator"),
        ("doctors", "doctors"),
        ("patients", "patients"),
    )

    for collection_name, role_name in role_map:
        user = db[collection_name].find_one({"email": normalized})
        if user:
            matches.append((user, role_name))

    return matches


def _build_auth_response(user: dict, role_name: str):
    token = create_access_token(
        identity=str(user["_id"]),
        additional_claims={
            "role": role_name,
            "email": user["email"],
            "gender": user.get("gender"),
        }
    )

    return jsonify({
        "message": "Authenticated",
        "role": role_name,
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "gender": user.get("gender"),
            "specialty": user.get("specialty", "")
        }
    })


def _json_safe(value):
    """Recursively convert Mongo-specific values into JSON-safe types."""
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _safe_user(doc: dict) -> dict:
    doc = _json_safe(dict(doc))
    doc.pop("password", None)
    reports = doc.get("clinical_reports")
    if isinstance(reports, list):
        sanitized_reports = []
        for report in reports:
            if isinstance(report, dict):
                clean_report = dict(report)
                clean_report.pop("disclaimer", None)
                sanitized_reports.append(clean_report)
            else:
                sanitized_reports.append(report)
        doc["clinical_reports"] = sanitized_reports
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


@auth_bp.post("/register")
def register():
    """
    POST /auth/register
    body:
    {
      "role": "administrator" | "doctors" | "patients",
      "email": "...",
      "password": "...",
      "full_name": "..."
    }
    """
    data = request.get_json(silent=True) or {}

    role = "patients"
    email = (data.get("email") or "").lower().strip()
    password = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    gender = (data.get("gender") or "").strip().lower() or None

    col, role_name = _collection_for_role(role)
    if col is None:
        return jsonify({"error": "Patient registration is currently the only public sign-up flow."}), 400
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    if _find_users_by_email(email):
        return jsonify({"error": "Email already exists. Please login."}), 409

    user_doc = {
        "email": email,
        "password": _hash_password(password),
        "full_name": full_name,
        "role": role_name,
        "gender": gender
    }

    try:
        res = col.insert_one(user_doc)
    except DuplicateKeyError:
        return jsonify({"error": "Email already exists. Please login."}), 409

    created_user = dict(user_doc)
    created_user["_id"] = res.inserted_id
    return _build_auth_response(created_user, role_name), 201


@auth_bp.post("/login")
def login():
    """
    POST /auth/login
    body:
    {
      "role": "administrator" | "doctors" | "patients",
      "email": "...",
      "password": "..."
    }
    """
    data = request.get_json(silent=True) or {}

    email = (data.get("email") or "").lower().strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    matches = _find_users_by_email(email)
    if not matches:
        return jsonify({"error": "No account found. Please register."}), 404
    if len(matches) > 1:
        return jsonify({
            "error": "This email is attached to multiple account types. Ask an administrator to resolve the duplicate records."
        }), 409

    user, role_name = matches[0]

    stored_hash = user.get("password", b"")
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")

    if not _check_password(password, stored_hash):
        return jsonify({"error": "Invalid email/password"}), 401

    return _build_auth_response(user, role_name), 200


@auth_bp.get("/admin/users")
@jwt_required()
def admin_users():
    """
    GET /auth/admin/users
    Admin can view both doctors + patients
    Header: Authorization: Bearer <ADMIN_TOKEN>
    """
    claims = get_jwt()
    if claims.get("role") != "administrator":
        return jsonify({"error": "Admin only"}), 403

    db = get_db()
    admin = db["administrator"].find_one({"email": claims.get("email")})
    doctors = [_safe_user(d) for d in db["doctors"].find().limit(500)]
    patients = [_safe_user(p) for p in db["patients"].find().limit(500)]

    return jsonify({
        "admin": _safe_user(admin) if admin else None,
        "doctors": doctors,
        "patients": patients
    }), 200


@auth_bp.post("/admin/doctors")
@jwt_required()
def create_doctor():
    """Admin-only doctor account creation."""
    claims = get_jwt()
    if claims.get("role") != "administrator":
        return jsonify({"error": "Admin only"}), 403

    data = request.get_json(silent=True) or {}
    email = (data.get("email") or "").lower().strip()
    password = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    specialty = (data.get("specialty") or "").strip()

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400
    if _find_users_by_email(email):
        return jsonify({"error": "Email already exists. Use a different email."}), 409

    if not full_name:
        full_name = email.split("@", 1)[0]

    db = get_db()
    user_doc = {
        "email": email,
        "password": _hash_password(password),
        "full_name": full_name,
        "role": "doctors",
        "specialty": specialty,
    }

    try:
        res = db["doctors"].insert_one(user_doc)
    except DuplicateKeyError:
        return jsonify({"error": "Email already exists. Use a different email."}), 409

    created_user = dict(user_doc)
    created_user["_id"] = res.inserted_id
    return jsonify({
        "message": "Doctor account created",
        "doctor": _safe_user(created_user)
    }), 201


@auth_bp.get("/doctor/patients")
@jwt_required()
def doctor_patients():
    """Doctors can review patient records; admins can also use this endpoint."""
    claims = get_jwt()
    if claims.get("role") not in ("doctors", "administrator"):
        return jsonify({"error": "Doctor or admin only"}), 403

    db = get_db()
    patients = [_safe_user(p) for p in db["patients"].find().limit(500)]
    return jsonify({"patients": patients}), 200


@auth_bp.get("/profile")
@jwt_required()
def get_profile():
    """
    GET /auth/profile
    Get the current user's profile based on their role.
    Header: Authorization: Bearer <TOKEN>
    """
    claims = get_jwt()
    user_id = claims.get("sub")
    role = claims.get("role")
    email = claims.get("email")
    
    col, role_name = _collection_for_role(role)
    if col is None:
        return jsonify({"error": "Invalid role"}), 400
    
    user = col.find_one({"_id": ObjectId(user_id)})
    
    if user is None:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify(_safe_user(user)), 200
