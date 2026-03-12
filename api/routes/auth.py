from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt
import bcrypt
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


def _safe_user(doc: dict) -> dict:
    doc = dict(doc)
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

    role = data.get("role")
    email = (data.get("email") or "").lower().strip()
    password = data.get("password") or ""
    full_name = (data.get("full_name") or "").strip()
    gender = (data.get("gender") or "").strip().lower() or None

    col, role_name = _collection_for_role(role)
    if col is None:
        return jsonify({"error": "role must be administrator/doctors/patients"}), 400
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

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

    token = create_access_token(
        identity=str(res.inserted_id),
        additional_claims={"role": role_name, "email": email, "gender": gender}
    )

    return jsonify({
        "message": "Registered",
        "role": role_name,
        "token": token,
        "user": {
            "id": str(res.inserted_id),
            "email": email,
            "full_name": full_name,
            "gender": gender
        }
    }), 201


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

    role = data.get("role")
    email = (data.get("email") or "").lower().strip()
    password = data.get("password") or ""

    col, role_name = _collection_for_role(role)
    if col is None:
        return jsonify({"error": "role must be administrator/doctors/patients"}), 400
    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = col.find_one({"email": email})
    if not user:
        return jsonify({"error": "No account found. Please register."}), 404

    stored_hash = user.get("password", b"")
    if isinstance(stored_hash, str):
        stored_hash = stored_hash.encode("utf-8")

    if not _check_password(password, stored_hash):
        return jsonify({"error": "Invalid email/password"}), 401

    token = create_access_token(
        identity=str(user["_id"]),
        additional_claims={
            "role": role_name,
            "email": user["email"],
            "gender": user.get("gender")
        }
    )

    return jsonify({
        "message": "Logged in",
        "role": role_name,
        "token": token,
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "full_name": user.get("full_name", ""),
            "gender": user.get("gender")
        }
    }), 200


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
    doctors = [_safe_user(d) for d in db["doctors"].find().limit(500)]
    patients = [_safe_user(p) for p in db["patients"].find().limit(500)]

    return jsonify({"doctors": doctors, "patients": patients}), 200


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
    
    from bson import ObjectId
    user = col.find_one({"_id": ObjectId(user_id)})
    
    if user is None:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify(_safe_user(user)), 200
