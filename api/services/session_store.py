"""
Session Store Service

Manages triage session state with support for:
- State machine (IDLE, GATHERING, ANALYZING, REPORTING, COMPLETE)
- Clinical data tracking
- Message history
- Data retention policies
"""

import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List


class SessionStore:
    """
    Session store for triage agent with new schema support.
    """
    
    # Default retention period (30 days)
    DEFAULT_RETENTION_DAYS = 30
    
    def __init__(self, ttl_seconds: int = 900, retention_days: int = DEFAULT_RETENTION_DAYS):
        """
        Initialize the session store.
        
        Args:
            ttl_seconds: Time-to-live for session inactivity (15 min default)
            retention_days: Data retention period in days
        """
        self.ttl = ttl_seconds
        self.retention_days = retention_days
        self.store = {}
    
    def create(self, data: dict = None) -> str:
        """
        Create a new session.
        
        Args:
            data: Initial session data
            
        Returns:
            Session ID
        """
        sid = str(uuid.uuid4())
        
        # Build initial session structure
        session_data = {
            "session_id": sid,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "current_state": "IDLE",
            "intent": None,
            "patient_profile": {
                "age": None,
                "sex": None
            },
            "clinical_data": {
                "chief_complaint": "",
                "symptoms": [],
                "medical_history": [],
                "medications": [],
                "risk_factors": [],
                "image_findings": None
            },
            "message_history": [],
            "tools_invoked": [],
            "final_report": None,
            "data_retention_policy": f"auto_delete_after_{self.retention_days}_days",
            "pii_encrypted": False
        }
        
        # Merge any provided data
        if data:
            session_data.update(data)
        
        self.store[sid] = {
            "data": session_data,
            "ts": time.time()
        }
        
        return sid
    
    def get(self, sid: str) -> Optional[Dict]:
        """
        Get session data.
        
        Args:
            sid: Session ID
            
        Returns:
            Session data or None if not found/expired
        """
        item = self.store.get(sid)
        if not item:
            return None
        
        # Check inactivity TTL
        if time.time() - item["ts"] > self.ttl:
            self.store.pop(sid, None)
            return None
        
        # Check retention policy
        session_data = item["data"]
        created_at = datetime.fromisoformat(session_data.get("created_at", datetime.utcnow().isoformat()))
        if datetime.utcnow() - created_at > timedelta(days=self.retention_days):
            self.store.pop(sid, None)
            return None
        
        return item["data"]
    
    def update(self, sid: str, data: dict) -> bool:
        """
        Update session data.
        
        Args:
            sid: Session ID
            data: Data to update
            
        Returns:
            True if updated, False if session not found
        """
        if sid not in self.store:
            return False
        
        # Update timestamp
        self.store[sid]["ts"] = time.time()
        
        # Merge data
        current = self.store[sid]["data"]
        current.update(data)
        current["updated_at"] = datetime.utcnow().isoformat()
        
        return True
    
    def update_state(self, sid: str, state: str) -> bool:
        """
        Update session state.
        
        Args:
            sid: Session ID
            state: New state
            
        Returns:
            True if updated
        """
        return self.update(sid, {"current_state": state})
    
    def update_clinical_data(self, sid: str, clinical_data: dict) -> bool:
        """
        Update clinical data section.
        
        Args:
            sid: Session ID
            clinical_data: Clinical data to merge
            
        Returns:
            True if updated
        """
        session = self.get(sid)
        if not session:
            return False
        
        current_clinical = session.get("clinical_data", {})
        current_clinical.update(clinical_data)
        
        return self.update(sid, {"clinical_data": current_clinical})
    
    def add_message(self, sid: str, role: str, content: str) -> bool:
        """
        Add a message to history.
        
        Args:
            sid: Session ID
            role: Message role (user/assistant/system)
            content: Message content
            
        Returns:
            True if added
        """
        session = self.get(sid)
        if not session:
            return False
        
        messages = session.get("message_history", [])
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return self.update(sid, {"message_history": messages})
    
    def get_state(self, sid: str) -> Optional[str]:
        """
        Get current session state.
        
        Args:
            sid: Session ID
            
        Returns:
            Current state or None
        """
        session = self.get(sid)
        return session.get("current_state") if session else None
    
    def get_clinical_data(self, sid: str) -> Optional[Dict]:
        """
        Get clinical data for session.
        
        Args:
            sid: Session ID
            
        Returns:
            Clinical data or None
        """
        session = self.get(sid)
        return session.get("clinical_data") if session else None
    
    def set_report(self, sid: str, report: Dict) -> bool:
        """
        Set final report for session.
        
        Args:
            sid: Session ID
            report: Clinical report
            
        Returns:
            True if set
        """
        return self.update(sid, {
            "final_report": report,
            "current_state": "COMPLETE"
        })
    
    def delete(self, sid: str) -> bool:
        """
        Delete a session.
        
        Args:
            sid: Session ID
            
        Returns:
            True if deleted
        """
        if sid in self.store:
            del self.store[sid]
            return True
        return False
    
    def list_sessions(self) -> List[str]:
        """
        List all session IDs.
        
        Returns:
            List of session IDs
        """
        return list(self.store.keys())
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.
        
        Returns:
            Number of sessions cleaned up
        """
        cleaned = 0
        now = time.time()
        
        for sid in list(self.store.keys()):
            item = self.store[sid]
            
            # Check TTL
            if now - item["ts"] > self.ttl:
                del self.store[sid]
                cleaned += 1
                continue
            
            # Check retention
            session_data = item["data"]
            created_at = datetime.fromisoformat(session_data.get("created_at", datetime.utcnow().isoformat()))
            if datetime.utcnow() - created_at > timedelta(days=self.retention_days):
                del self.store[sid]
                cleaned += 1
        
        return cleaned


# Singleton instance
_session_store = None

def get_session_store() -> SessionStore:
    """Get or create the singleton SessionStore instance."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore()
    return _session_store
