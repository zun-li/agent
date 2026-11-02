"""Persistent teammates, inboxes, plan gates and task-bound Git worktrees."""

import json
import re
import subprocess
from functools import partial
from pathlib import Path
from uuid import uuid4

from domain import Scope, Tool


class Teams:
    READ = frozenset({"glob", "read_file"})
    FILE = READ | {"bash", "write_file", "edit_file"}
    CONTROL = frozenset(
        {
            "list_tasks",
            "get_task",
            "claim_task",
            "complete_task",
            "send_message",
            "submit_plan",
            "read_context",
        }
    )

    def __init__(self, work):
        self.work = work
        self.tree_root = work.state / "worktrees"
        with work.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS mates(
                    session TEXT, name TEXT, role TEXT, status TEXT, task TEXT,
                    plan TEXT DEFAULT 'none', request TEXT, stop_request TEXT,
                    PRIMARY KEY(session,name));
                CREATE TABLE IF NOT EXISTS inbox(
                    id INTEGER PRIMARY KEY, session TEXT, recipient TEXT, body TEXT);
            """)

    def _git(self, cwd, *args):
        result = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", "-C", str(cwd), *args],
            check=False,
            text=True,
            capture_output=True,
            timeout=30,
        )
        if result.returncode:
            raise ValueError(result.stderr.strip() or "Git worktree operation failed")
        return result.stdout

    def directory(self, task):
        if not task["worktree"]:
            return self.work.workspace
        path = Path(task["worktree"])
        if not path.is_dir() or path.resolve().parent != self.tree_root.resolve():
            raise PermissionError("Task worktree is missing or outside the managed directory")
        listing = self._git(self.work.workspace, "worktree", "list", "--porcelain", "-z")
        if "worktree " + str(path.resolve()) not in listing.split("\0"):
            raise PermissionError("Task directory is not a registered worktree")
        return path.resolve()

    def create_tree(self, scope, task_id: str, name: str):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}", name):
            raise ValueError("Use a short alphanumeric worktree name")
        with self.work.db() as db:
            task = self.work.task(db, task_id)
            if task["status"] != "pending" or task["worktree"]:
                raise ValueError("Worktree requires a pending unbound task")
            self.tree_root.mkdir(exist_ok=True)
            path = self.tree_root / f"{name}-{task_id}"
            self._git(
                self.work.workspace,
                "worktree",
                "add",
                "-b",
                f"agent/{name}-{task_id}",
                str(path),
                "HEAD",
            )
            db.execute("UPDATE tasks SET worktree=? WHERE id=?", (str(path), task_id))
            return str(path)

    def remove_tree(self, scope, task_id, discard_changes=False):
        """Host-only cleanup. Keep the branch; discard requires a host decision."""
        with self.work.db() as db:
            task = self.work.task(db, task_id)
            if (
                not task["worktree"]
                or task["status"] != "completed"
                or db.execute("SELECT 1 FROM mates WHERE task=?", (task_id,)).fetchone()
            ):
                raise ValueError("Worktree is absent or still assigned")
            path = self.directory(task)
            if not discard_changes and self._git(path, "status", "--porcelain", "--ignored"):
                raise ValueError("Worktree contains local changes")
            args = ["worktree", "remove"] + (["--force"] if discard_changes else []) + [str(path)]
            self._git(self.work.workspace, *args)
            db.execute("UPDATE tasks SET worktree=NULL WHERE id=?", (task_id,))

    @staticmethod
    def mate(db, session, name, active=True):
        row = db.execute("SELECT * FROM mates WHERE session=? AND name=?", (session, name)).fetchone()
        if row is None or (active and row["status"] in ("closed", "failed")):
            raise ValueError("Unknown or inactive teammate")
        return row

    def cwd(self, scope):
        with self.work.db() as db:
            mate = self.mate(db, scope.session, scope.actor)
            if mate["task"]:
                return self.directory(self.work.task(db, mate["task"]))
            return self.work.workspace

    def guard(self, scope, name):
        with self.work.db() as db:
            mate = self.mate(db, scope.session, scope.actor)
            if mate["stop_request"] or name not in self.FILE | self.CONTROL:
                raise PermissionError("Tool unavailable for this teammate")
            if name in self.FILE:
                if not mate["task"]:
                    raise PermissionError("Claim a task before accessing the workspace")
                self.directory(self.work.task(db, mate["task"]))
            if name in {"bash", "write_file", "edit_file", "complete_task"} and mate["plan"] not in {
                "none",
                "approved",
            }:
                raise PermissionError("Lead approval of this task's plan is required")

    def enqueue(self, db, session, name, body):
        db.execute("INSERT INTO inbox(session,recipient,body) VALUES(?,?,?)", (session, name, body))

    def spawn(self, scope, name: str, role: str, task_id: str | None = None, require_plan: bool = False):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", name) or name.lower() in {"lead", "agent"}:
            raise ValueError("Invalid teammate name")
        if require_plan and not task_id:
            raise ValueError("A required plan must be bound to an initial task")
        if self.work.run_child is None:
            raise RuntimeError("Teammate runner offline")
        with self.work.db() as db:
            db.execute(
                "INSERT INTO mates(session,name,role,status) VALUES(?,?,?,'idle')",
                (scope.session, name, role),
            )
            if task_id:
                task = self.work.claim_in(db, scope.session, task_id, name)
                if require_plan:
                    db.execute(
                        "UPDATE mates SET plan='required' WHERE session=? AND name=?",
                        (scope.session, name),
                    )
                self.enqueue(
                    db, scope.session, name, "Assigned task: " + json.dumps(task, ensure_ascii=False)
                )
        self.work.spawn_thread(self._worker, scope.session, name)
        return name

    def resume(self):
        with self.work.db() as db:
            mates = db.execute("SELECT * FROM mates WHERE status IN ('idle','working')").fetchall()
            for m in mates:
                if m["status"] == "working":
                    self.enqueue(
                        db,
                        m["session"],
                        m["name"],
                        "Resume your interrupted assignment; inspect progress first.",
                    )
        for m in mates:
            self.work.spawn_thread(self._worker, m["session"], m["name"])

    def send(self, scope, to: str, text: str):
        with self.work.db() as db:
            if to == "lead":
                self.work.emit(db, scope.session, uuid4().hex, f"Message from teammate {scope.actor}: {text}")
            else:
                self.mate(db, scope.session, to)
                self.enqueue(db, scope.session, to, f"Message from {scope.actor}: {text}")
        return "Delivered"

    def list(self, scope):
        with self.work.db() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT name,role,status,task,plan,request,stop_request "
                    "FROM mates WHERE session=? ORDER BY name",
                    (scope.session,),
                )
            ]

    def request_plan(self, scope, teammate: str):
        with self.work.db() as db:
            m = self.mate(db, scope.session, teammate)
            if not m["task"]:
                raise ValueError("No assigned task")
            db.execute(
                "UPDATE mates SET plan='required',request=NULL WHERE session=? AND name=?",
                (scope.session, teammate),
            )
            self.enqueue(db, scope.session, teammate, "Lead requests a plan. Use submit_plan before editing.")
        return "Plan requested"

    def submit_plan(self, scope, plan: str):
        with self.work.db() as db:
            m = self.mate(db, scope.session, scope.actor)
            if not m["task"] or m["plan"] not in {"required", "rejected"}:
                raise ValueError("No plan requested")
            request = uuid4().hex
            db.execute(
                "UPDATE mates SET plan='pending',request=? WHERE session=? AND name=?",
                (request, scope.session, scope.actor),
            )
            self.work.emit(
                db,
                scope.session,
                request,
                f"Plan submitted by {scope.actor} for task {m['task']}. Request {request}: {plan}",
            )
            return request

    def review(self, scope, teammate: str, request_id: str, approve: bool, feedback: str = ""):
        with self.work.db() as db:
            m = self.mate(db, scope.session, teammate)
            if m["plan"] != "pending" or m["request"] != request_id:
                raise ValueError("Stale or unknown plan approval request")
            status = "approved" if approve else "rejected"
            db.execute(
                "UPDATE mates SET plan=?,request=NULL WHERE session=? AND name=?",
                (status, scope.session, teammate),
            )
            self.enqueue(db, scope.session, teammate, f"Lead {status} your plan. {feedback}")
        return status

    def shutdown(self, scope, teammate: str):
        with self.work.db() as db:
            m = self.mate(db, scope.session, teammate)
            request = m["stop_request"] or uuid4().hex
            db.execute(
                "UPDATE mates SET stop_request=? WHERE session=? AND name=?",
                (request, scope.session, teammate),
            )
        return request

    def _end(self, db, session, name, status):
        db.execute(
            "UPDATE tasks SET status='pending',owner=NULL,owner_session=NULL WHERE owner_session=? AND owner=? AND status='in_progress'",
            (session, name),
        )
        db.execute("DELETE FROM inbox WHERE session=? AND recipient=?", (session, name))
        db.execute(
            "UPDATE mates SET status=?,task=NULL,request=NULL,plan='none' WHERE session=? AND name=?",
            (status, session, name),
        )

    def _next(self, session, name):
        with self.work.db() as db:
            m = self.mate(db, session, name)
            if m["stop_request"]:
                self._end(db, session, name, "closed")
                self.work.emit(
                    db,
                    session,
                    m["stop_request"],
                    f"Teammate {name} acknowledged shutdown {m['stop_request']}.",
                )
                return None
            mail = db.execute(
                "SELECT id,body FROM inbox WHERE session=? AND recipient=? ORDER BY id", (session, name)
            ).fetchall()
            parts = [item["body"] for item in mail]
            if parts:
                db.execute("DELETE FROM inbox WHERE session=? AND recipient=?", (session, name))
            elif not m["task"]:
                for row in db.execute(
                    "SELECT id FROM tasks WHERE session=? AND status='pending' ORDER BY rowid",
                    (session,),
                ):
                    if self.work.ready(db, row[0]):
                        task = self.work.claim_in(db, session, row[0], name)
                        parts.append("Auto-claimed task: " + json.dumps(task, ensure_ascii=False))
                        break
            if not parts:
                return ""
            db.execute("UPDATE mates SET status='working' WHERE session=? AND name=?", (session, name))
            gate = " Submit a plan before edits." if m["plan"] == "required" else ""
            return (
                f"You are teammate {name}. Role: {m['role']}.{gate} Complete finished tasks with complete_task.\n"
                + "\n".join(parts)
            )

    def _worker(self, session, name):
        try:
            while not self.work.stop.is_set():
                prompt = self._next(session, name)
                if prompt is None:
                    return
                if not prompt:
                    self.work.stop.wait(0.25)
                    continue
                scope = Scope(session=session, actor=name, cwd=self.work.workspace)
                result = self.work.run_child(self.work.scope(scope), prompt)
                if self.work.stop.is_set():
                    return  # The next host resumes this assignment.
                with self.work.db() as db:
                    self.work.emit(db, session, uuid4().hex, f"Teammate {name} result: {result}")
                    self.work.emit(db, session, uuid4().hex, f"Teammate {name} is idle.")
                    db.execute(
                        "UPDATE mates SET status='idle',task=CASE WHEN task IN "
                        "(SELECT id FROM tasks WHERE status='completed') THEN NULL ELSE task END "
                        "WHERE session=? AND name=?",
                        (session, name),
                    )
        except Exception as error:  # noqa: BLE001 - release task ownership when a worker fails
            if self.work.stop.is_set():
                return
            with self.work.db() as db:
                self._end(db, session, name, "failed")
                self.work.emit(db, session, uuid4().hex, f"Teammate {name} failed: {error}")

    def tools(self, scope):
        tool = partial(Tool.bind, scope)
        send = tool(self.send, "send_message", "Send a direct message to a teammate or lead.")
        if scope.actor != "lead":
            return [send, tool(self.submit_plan, "submit_plan", "Submit your plan for lead approval.")]
        return [
            send,
            tool(
                self.spawn,
                "spawn_teammate",
                "Start a named persistent teammate, optionally with an initial task.",
                approval=True,
            ),
            tool(self.list, "list_teammates", "List teammates, assignments and plans."),
            tool(self.request_plan, "request_plan", "Require a plan before task edits."),
            tool(self.review, "review_plan", "Approve or reject a pending plan."),
            tool(
                self.shutdown, "request_shutdown", "Request graceful shutdown and receive an acknowledgement."
            ),
            tool(
                self.create_tree,
                "create_worktree",
                "Bind a new Git worktree to an unclaimed task.",
                approval=True,
            ),
        ]
