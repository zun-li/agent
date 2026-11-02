import json
import os
import threading
from pathlib import Path
from uuid import uuid4


class JsonlSessionStore:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()

    def create(self):
        session = uuid4().hex
        (self.root / f"{session}.jsonl").touch(mode=0o600, exist_ok=False)
        return session

    def list(self):
        return sorted(p.stem for p in self.root.glob("*.jsonl"))

    def path(self, session):
        if len(session) != 32 or any(c not in "0123456789abcdef" for c in session):
            raise KeyError("unknown session")
        path = self.root / f"{session}.jsonl"
        if not path.is_file():
            raise KeyError("unknown session")
        return path

    def read(self, session):
        with self.lock, self.path(session).open("rb+") as f:
            events = []
            for line in f:
                if not line.endswith(b"\n"):  # interrupted final append
                    f.seek(-len(line), os.SEEK_CUR)
                    f.truncate()
                    break
                events.append(json.loads(line))
            return events

    def append(self, session, event):
        with self.lock, self.path(session).open("ab") as f:
            f.write(json.dumps(event, ensure_ascii=False).encode() + b"\n")
            f.flush()
            os.fsync(f.fileno())
