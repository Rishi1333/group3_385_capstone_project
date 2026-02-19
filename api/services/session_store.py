import uuid
import time

class SessionStore:
    def __init__(self, ttl_seconds=900):
        self.ttl = ttl_seconds
        self.store = {}

    def create(self, data: dict):
        sid = str(uuid.uuid4())
        self.store[sid] = {"data": data, "ts": time.time()}
        return sid

    def get(self, sid: str):
        item = self.store.get(sid)
        if not item:
            return None
        if time.time() - item["ts"] > self.ttl:
            self.store.pop(sid, None)
            return None
        return item["data"]

    def update(self, sid: str, data: dict):
        if sid in self.store:
            self.store[sid] = {"data": data, "ts": time.time()}
            return True
        return False

    def delete(self, sid: str):
        self.store.pop(sid, None)