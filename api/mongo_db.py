import os

from dotenv import load_dotenv
from pymongo import MongoClient

DB_NAME = "VirtualClinicDatabase"
ROLE_COLLECTIONS = ("administrator", "doctors", "patients")

_client = None
_db = None


def init_mongo():
    """
    Initialize a single MongoDB connection and ensure required indexes exist.
    """
    global _client, _db

    if _db is not None:
        return _db

    load_dotenv()
    mongo_uri = os.environ.get("MONGO_URI")
    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable is not set")

    _client = MongoClient(mongo_uri)
    _client.admin.command("ping")
    _db = _client[DB_NAME]

    for collection_name in ROLE_COLLECTIONS:
        _db[collection_name].create_index("email", unique=True, name="unique_email")

    return _db


def get_db():
    """
    Return the initialized MongoDB database handle.
    """
    if _db is None:
        raise RuntimeError("MongoDB is not initialized. Call init_mongo() at app startup.")
    return _db
