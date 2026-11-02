"""Trusted, registered workflow scripts with per-call checkpoints."""

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from functools import partial
from uuid import uuid4

from domain import Tool, check_shape


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


class Steps:
    def __init__(self, registry, scope, run_id, path=()):
        self.registry, self.scope, self.run_id, self.path = registry, scope, run_id, path
        self.work = registry.work

    def log(self, message):
        with self.work.db() as db:
            db.execute("INSERT INTO flow_events(run,body) VALUES(?,?)", (self.run_id, message))

    def phase(self, name):
        self.log("phase: " + name)

    def agent(self, label, prompt, schema=None):
        key = hashlib.sha256(canonical([self.path, label, prompt, schema]).encode()).hexdigest()
        with self.work.db() as db:
            row = db.execute("SELECT value FROM steps WHERE run=? AND key=?", (self.run_id, key)).fetchone()
        if row:
            self.log("cached: " + label)
            return json.loads(row[0])
        self.log("agent: " + label)
        query = prompt
        if schema is not None:
            query += "\nRespond with JSON matching: " + canonical(schema)
        for attempt in range(2):
            child = replace(self.scope, actor="workflow-" + uuid4().hex, child=True)
            output = self.work.run_child(child, query)
            if schema is None:
                value = output
                break
            try:
                value = json.loads(output)
                check_shape(value, schema)
                break
            except (TypeError, ValueError) as error:
                if attempt:
                    raise ValueError(f"Invalid structured result from {label}: {error}") from error
                query += f"\nThe previous output was invalid ({error}). Return valid JSON only."
        with self.work.db() as db:
            db.execute("INSERT OR IGNORE INTO steps VALUES(?,?,?)", (self.run_id, key, canonical(value)))
        return value

    @staticmethod
    def parallel(thunks):
        with ThreadPoolExecutor(max_workers=4) as workers:
            futures = [workers.submit(fn) for fn in thunks]
            return [f.result() for f in futures]

    def pipeline(self, items, *stages):
        def run(item):
            value = item
            for stage in stages:
                value = stage(value, item)
            return value

        return self.parallel([lambda item=item: run(item) for item in items])

    def workflow(self, name, args):
        if self.path:
            raise ValueError("Only one level of nested workflows is supported")
        return self.registry.script(name)(Steps(self.registry, self.scope, self.run_id, (name,)), args)


def review_changes(ctx, args):
    changes = args.get("changes", "")
    if not isinstance(changes, str) or not changes.strip():
        raise ValueError("review-changes requires a nonempty changes string")
    result_schema = {
        "type": "object",
        "required": ["findings"],
        "additionalProperties": False,
        "properties": {"findings": {"type": "array", "items": {"type": "string"}}},
    }
    ctx.phase("Review")

    def audit(_, dimension):
        return ctx.agent(
            dimension, f"Inspect these changes for {dimension} issues:\n{changes}", result_schema
        )

    def verify(audited, dimension):
        if not audited["findings"]:
            return []
        prompt = f"Verify these {dimension} findings against the changes. Retain only real issues.\n"
        return ctx.agent("verify " + dimension, prompt + changes + "\n" + canonical(audited), result_schema)[
            "findings"
        ]

    results = ctx.pipeline(["correctness", "security", "regressions"], audit, verify)
    ctx.phase("Synthesis")
    return {"findings": list(dict.fromkeys(item for group in results for item in group))}


class Workflows:
    def __init__(self, work):
        self.work = work
        self.scripts = {}
        self.active = set()
        self.register(
            "review-changes", "Review a supplied diff in parallel and verify findings.", review_changes
        )
        with work.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS flows(
                    id TEXT PRIMARY KEY, session TEXT, name TEXT, args TEXT, status TEXT, output TEXT);
                CREATE TABLE IF NOT EXISTS steps(
                    run TEXT, key TEXT, value TEXT, PRIMARY KEY(run,key));
                CREATE TABLE IF NOT EXISTS flow_events(id INTEGER PRIMARY KEY, run TEXT, body TEXT);
            """)

    def register(self, name, description, script):
        if (
            not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,63}", name)
            or not description
            or not callable(script)
        ):
            raise ValueError("Invalid workflow registration")
        if name in self.scripts:
            raise ValueError("Workflow already registered")
        self.scripts[name] = (description, script)

    def script(self, name):
        if name not in self.scripts:
            raise ValueError("Unknown saved workflow")
        return self.scripts[name][1]

    @contextmanager
    def _lock_run(self, run_id):
        # Work.start gives this process exclusive ownership of the runtime.
        with self.work.lock:
            if run_id in self.active:
                raise ValueError("Workflow run is already active")
            self.active.add(run_id)
        try:
            yield
        finally:
            with self.work.lock:
                self.active.remove(run_id)

    def run(self, scope, name: str, args: dict | None = None, resume_from_run_id: str | None = None):
        script = self.script(name)
        if resume_from_run_id is not None and not re.fullmatch(r"[a-f0-9]{32}", resume_from_run_id):
            raise ValueError("Invalid run ID")
        run_id = resume_from_run_id or uuid4().hex
        with self._lock_run(run_id):
            with self.work.db() as db:
                if resume_from_run_id:
                    row = db.execute(
                        "SELECT * FROM flows WHERE session=? AND id=?", (scope.session, run_id)
                    ).fetchone()
                    if (
                        not row
                        or row["name"] != name
                        or (args is not None and canonical(args) != row["args"])
                    ):
                        raise ValueError(
                            "Resume must match the original session, workflow name and arguments"
                        )
                    args = json.loads(row["args"])
                    db.execute("UPDATE flows SET status='running',output=NULL WHERE id=?", (run_id,))
                else:
                    args = args or {}
                    db.execute(
                        "INSERT INTO flows VALUES(?,?,?,?,'running',NULL)",
                        (run_id, scope.session, name, canonical(args)),
                    )
            ctx = Steps(self, scope, run_id)
            ctx.log("started")
            try:
                output, status = canonical(script(ctx, args)), "completed"
            except Exception as error:  # noqa: BLE001 - persist the failed run for resumption
                output, status = canonical({"error": str(error)}), "failed"
            with self.work.db() as db:
                db.execute("UPDATE flows SET status=?,output=? WHERE id=?", (status, output, run_id))
            ctx.log(status)
            return {"run_id": run_id, "status": status, "result": json.loads(output)}

    def status(self, scope, run_id: str | None = None):
        with self.work.db() as db:
            if not run_id:
                return [
                    dict(row)
                    for row in db.execute(
                        "SELECT id,name,status FROM flows WHERE session=? ORDER BY rowid DESC",
                        (scope.session,),
                    )
                ]
            row = db.execute(
                "SELECT id,name,status,output FROM flows WHERE session=? AND id=?",
                (scope.session, run_id),
            ).fetchone()
            if not row:
                raise ValueError("Unknown workflow run")
            return dict(
                row,
                events=[
                    r[0]
                    for r in db.execute("SELECT body FROM flow_events WHERE run=? ORDER BY id", (run_id,))
                ],
            )

    def tools(self, scope):
        tool = partial(Tool.bind, scope)
        return [
            tool(
                self.run,
                "workflow",
                "Run or resume a registered workflow. Available: " + ", ".join(self.scripts),
            ),
            tool(self.status, "workflow_status", "List workflow runs or inspect a run and its progress."),
        ]
