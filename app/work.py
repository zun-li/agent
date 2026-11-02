"""Workspace task board, session-scoped workers and scheduled deliveries."""

import logging
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime
from functools import partial
from pathlib import Path

from domain import Tool

from .teams import Teams
from .workflows import Workflows

log = logging.getLogger(__name__)


def uid():
    return uuid.uuid4().hex


def cron_fields(expression):
    parts = expression.split()
    if len(parts) != 5:
        raise ValueError("cron requires minute hour day month weekday")
    result = []
    for part, (lo, hi) in zip(parts, ((0, 59), (0, 23), (1, 31), (1, 12), (0, 6))):
        numbers = set()
        for item in part.split(","):
            term, sep, step = item.partition("/")
            stride = int(step) if sep else 1
            if term == "*":
                start, end = lo, hi
            else:
                left, dash, right = term.partition("-")
                start = int(left)
                end = int(right) if dash else start
            if not lo <= start <= end <= hi or stride < 1:
                raise ValueError(f"Invalid cron field: {item}")
            numbers.update(range(start, end + 1, stride))
        result.append(numbers)
    return result


def cron_matches(expression, now):
    minute, hour, day, month, weekday = cron_fields(expression)
    dom, dow = now.day in day, (now.weekday() + 1) % 7 in weekday
    both_restricted = expression.split()[2] != "*" and expression.split()[4] != "*"
    day_match = (dom or dow) if both_restricted else (dom and dow)
    return now.minute in minute and now.hour in hour and now.month in month and day_match


class Work:
    def __init__(self, state: Path, workspace: Path):
        self.state = state.resolve()
        self.state.mkdir(parents=True, exist_ok=True)
        self.workspace = workspace.resolve()
        self.path = self.state / "work.sqlite3"
        self.instance = uid()
        self.stop = threading.Event()
        self.threads = []
        self.lock = threading.RLock()
        self.lease = None
        self.run_child = self.notify = self.run_shell = None
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS tasks(
                    session TEXT, id TEXT PRIMARY KEY, subject TEXT, description TEXT,
                    status TEXT DEFAULT 'pending', owner TEXT, owner_session TEXT,
                    worktree TEXT, result TEXT DEFAULT '');
                CREATE INDEX IF NOT EXISTS tasks_session ON tasks(session,status);
                CREATE TABLE IF NOT EXISTS deps(task TEXT, parent TEXT, PRIMARY KEY(task,parent));
                CREATE TABLE IF NOT EXISTS background(
                    session TEXT, id TEXT PRIMARY KEY, status TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS cron(
                    session TEXT, id TEXT PRIMARY KEY, expression TEXT, prompt TEXT,
                    recurring INTEGER, instance TEXT, last_fired TEXT);
                CREATE TABLE IF NOT EXISTS outbox(
                    session TEXT, id TEXT PRIMARY KEY, body TEXT);
            """)
            if "owner_session" not in {row["name"] for row in db.execute("PRAGMA table_info(tasks)")}:
                db.execute("ALTER TABLE tasks ADD COLUMN owner_session TEXT")
                db.execute("UPDATE tasks SET owner_session=session WHERE owner IS NOT NULL")
        self.teams = Teams(self)
        self.workflows = Workflows(self)

    @contextmanager
    def db(self):
        db = sqlite3.connect(str(self.path), timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                db.execute("BEGIN IMMEDIATE")
                yield db
        finally:
            db.close()

    def start(self, run_child, notify, run_shell):
        if self.run_child is not None:
            raise RuntimeError("work runtime already started")
        lease = sqlite3.connect(str(self.state / "worker-lock.sqlite3"), timeout=0, check_same_thread=False)
        try:
            lease.execute("BEGIN EXCLUSIVE")
        except sqlite3.OperationalError as exc:
            lease.close()
            raise RuntimeError("Another agent runtime already owns this state directory") from exc
        self.lease = lease
        self.run_child, self.notify, self.run_shell = run_child, notify, run_shell
        with self.db() as db:
            db.execute("DELETE FROM cron WHERE instance IS NOT NULL AND instance!=?", (self.instance,))
            for row in db.execute("SELECT * FROM background WHERE status='running'"):
                db.execute(
                    "UPDATE background SET status='interrupted', result=? WHERE id=?",
                    ("Agent stopped before the command returned", row["id"]),
                )
                self.emit(db, row["session"], row["id"], f"Background {row['id']} interrupted.")
        self.teams.resume()
        self.spawn_thread(self._monitor)

    def spawn_thread(self, fn, *args):
        with self.lock:
            if self.stop.is_set():
                raise RuntimeError("work runtime is closing")
            self.threads = [t for t in self.threads if t.is_alive()]
            thread = threading.Thread(target=fn, args=args, daemon=True)
            self.threads.append(thread)
            thread.start()

    def close(self):
        with self.lock:
            self.stop.set()
            threads = tuple(self.threads)
        for thread in threads:
            if thread != threading.current_thread():
                thread.join(timeout=1)
        with self.db() as db:
            db.execute("DELETE FROM cron WHERE instance=?", (self.instance,))
        if self.lease is not None:
            self.lease.close()
            self.lease = None

    @staticmethod
    def emit(db, session, event_id, body):
        db.execute("INSERT OR IGNORE INTO outbox VALUES(?,?,?)", (session, event_id, body))

    def _monitor(self):
        while not self.stop.is_set():
            try:
                self.tick(datetime.now())
                with self.db() as db:
                    events = db.execute("SELECT * FROM outbox ORDER BY rowid LIMIT 100").fetchall()
                for event in events:
                    if self.stop.is_set():
                        return
                    self.notify(event["session"], event["id"], event["body"])
                    with self.db() as db:
                        db.execute("DELETE FROM outbox WHERE id=?", (event["id"],))
            except Exception:
                log.exception("Scheduled work or event delivery failed")
            self.stop.wait(0.5)

    def tick(self, now):
        stamp = now.strftime("%Y-%m-%dT%H:%M")
        with self.db() as db:
            jobs = db.execute(
                "SELECT * FROM cron WHERE (instance IS NULL OR instance=?) "
                "AND (last_fired IS NULL OR last_fired!=?)",
                (self.instance, stamp),
            ).fetchall()
            for job in jobs:
                if not cron_matches(job["expression"], now):
                    continue
                self.emit(db, job["session"], f"cron-{job['id']}-{stamp}", "[Scheduled] " + job["prompt"])
                if job["recurring"]:
                    db.execute("UPDATE cron SET last_fired=? WHERE id=?", (stamp, job["id"]))
                else:
                    db.execute("DELETE FROM cron WHERE id=?", (job["id"],))

    def schedule(self, scope, cron: str, prompt: str, recurring: bool = True, durable: bool = False):
        cron_fields(cron)
        if not prompt.strip():
            raise ValueError("Scheduled prompt is empty")
        job_id = uid()
        with self.db() as db:
            db.execute(
                "INSERT INTO cron VALUES(?,?,?,?,?,?,NULL)",
                (scope.session, job_id, cron, prompt, recurring, None if durable else self.instance),
            )
        return job_id

    def list_crons(self, scope):
        with self.db() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT id,expression,prompt,recurring,"
                    "(instance IS NULL) AS durable FROM cron WHERE session=? AND (instance IS NULL OR instance=?)",
                    (scope.session, self.instance),
                )
            ]

    def cancel_cron(self, scope, job_id: str):
        with self.db() as db:
            if not db.execute(
                "DELETE FROM cron WHERE id=? AND session=? AND (instance IS NULL OR instance=?)",
                (job_id, scope.session, self.instance),
            ).rowcount:
                raise ValueError("Unknown schedule")
        return "Cancelled"

    def background(self, scope, command: str):
        if self.run_shell is None or self.stop.is_set():
            raise RuntimeError("Background runner is offline")
        job_id = uid()
        with self.db() as db:
            db.execute("INSERT INTO background VALUES(?,?,'running','')", (scope.session, job_id))

        def execute():
            try:
                output, status = self.run_shell(scope.cwd, command), "completed"
            except Exception as error:  # noqa: BLE001 - report background failures to the originating turn
                output, status = str(error), "failed"
            if self.stop.is_set():
                return  # The next host records this command as interrupted.
            with self.db() as db:
                db.execute("UPDATE background SET status=?,result=? WHERE id=?", (status, output, job_id))
                self.emit(db, scope.session, job_id, f"Background {job_id} {status}\n{output}")

        self.spawn_thread(execute)
        return job_id

    def background_status(self, scope, job_id: str):
        with self.db() as db:
            row = db.execute(
                "SELECT id,status,result FROM background WHERE id=? AND session=?",
                (job_id, scope.session),
            ).fetchone()
            if not row:
                raise ValueError("Unknown background command")
            return dict(row)

    @staticmethod
    def task(db, task_id):
        row = db.execute("SELECT * FROM tasks WHERE id=?", (task_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown task")
        return dict(
            row, blockedBy=[r[0] for r in db.execute("SELECT parent FROM deps WHERE task=?", (task_id,))]
        )

    def create(self, scope, subject: str, description: str = ""):
        if not subject.strip():
            raise ValueError("Task needs a subject")
        task_id = uid()
        with self.db() as db:
            db.execute(
                "INSERT INTO tasks(id,session,subject,description) VALUES(?,?,?,?)",
                (task_id, scope.session, subject, description),
            )
            return self.task(db, task_id)

    def update(self, scope, task_id: str, addBlockedBy: list[str]):
        with self.db() as db:
            task = self.task(db, task_id)
            if task["status"] != "pending":
                raise ValueError("Only pending tasks can gain dependencies")
            for parent in addBlockedBy:
                self.task(db, parent)
                cycle = db.execute(
                    """WITH RECURSIVE ancestors(id) AS
                    (SELECT ? UNION SELECT parent FROM deps JOIN ancestors ON task=id)
                    SELECT 1 FROM ancestors WHERE id=?""",
                    (parent, task_id),
                ).fetchone()
                if cycle:
                    raise ValueError("Dependency would form a cycle")
                db.execute("INSERT OR IGNORE INTO deps VALUES(?,?)", (task_id, parent))
            return self.task(db, task_id)

    @staticmethod
    def ready(db, task_id):
        return not db.execute(
            "SELECT 1 FROM deps JOIN tasks ON id=parent WHERE task=? AND status!='completed'", (task_id,)
        ).fetchone()

    def claim_in(self, db, session, task_id, owner):
        task = self.task(db, task_id)
        if task["status"] != "pending" or not self.ready(db, task_id):
            raise ValueError("Task already owned, completed or blocked")
        if db.execute(
            "SELECT 1 FROM tasks WHERE owner_session=? AND owner=? AND status='in_progress'",
            (session, owner),
        ).fetchone():
            raise ValueError("Finish current assignment first")
        self.teams.directory(task)
        db.execute(
            "UPDATE tasks SET status='in_progress',owner=?,owner_session=? WHERE id=?",
            (owner, session, task_id),
        )
        if owner != "lead":
            db.execute(
                "UPDATE mates SET task=?,plan='none',request=NULL WHERE session=? AND name=?",
                (task_id, session, owner),
            )
        return self.task(db, task_id)

    def claim(self, scope, task_id: str):
        with self.db() as db:
            return self.claim_in(db, scope.session, task_id, scope.actor)

    def complete(self, scope, task_id: str, result: str = ""):
        with self.db() as db:
            task = self.task(db, task_id)
            if task["status"] != "in_progress" or (task["owner_session"], task["owner"]) != (
                scope.session,
                scope.actor,
            ):
                raise PermissionError("Only the current task owner can complete it")
            db.execute("UPDATE tasks SET status='completed',result=? WHERE id=?", (result, task_id))
            unblocked = [
                row[0]
                for row in db.execute(
                    "SELECT task FROM deps JOIN tasks ON id=task WHERE parent=? AND status='pending'",
                    (task_id,),
                )
                if self.ready(db, row[0])
            ]
            return dict(self.task(db, task_id), unblocked=unblocked)

    def list_tasks(self, scope):
        with self.db() as db:
            return [self.task(db, row[0]) for row in db.execute("SELECT id FROM tasks ORDER BY rowid")]

    def get(self, scope, task_id: str):
        with self.db() as db:
            return self.task(db, task_id)

    def scope(self, scope):
        if scope.child:
            return scope
        if scope.actor != "lead":
            return replace(scope, cwd=self.teams.cwd(scope))
        with self.db() as db:
            task = db.execute(
                "SELECT * FROM tasks WHERE owner_session=? AND owner='lead' AND status='in_progress'",
                (scope.session,),
            ).fetchone()
            cwd = self.teams.directory(task) if task else self.workspace
        return replace(scope, cwd=cwd)

    def busy(self, session, delivered=()):
        with self.db() as db:
            return bool(
                db.execute(
                    """SELECT 1 FROM background WHERE session=? AND status='running'
                UNION ALL SELECT 1 FROM mates WHERE session=? AND status='working'
                UNION ALL SELECT 1 FROM inbox WHERE session=?
                LIMIT 1""",
                    (session, session, session),
                ).fetchone()
            ) or any(
                row[0] not in delivered
                for row in db.execute("SELECT id FROM outbox WHERE session=?", (session,))
            )

    def guard(self, scope, tool_name):
        if (
            scope.child
            and scope.actor.startswith("workflow-")
            and tool_name not in {"read_file", "glob", "read_context"}
        ):
            raise PermissionError("Workflow steps may only inspect files")
        if not scope.child and scope.actor != "lead":
            self.teams.guard(scope, tool_name)

    def tools(self, scope):
        tool = partial(Tool.bind, scope)
        if scope.child:
            return []
        tools = [
            tool(self.get, "get_task", "View a workspace task and its prerequisites."),
            tool(self.list_tasks, "list_tasks", "List the shared workspace task board."),
            tool(self.claim, "claim_task", "Claim an unblocked task for yourself."),
            tool(self.complete, "complete_task", "Complete your current task."),
        ] + self.teams.tools(scope)
        if scope.actor != "lead":
            return tools
        return (
            tools
            + [
                tool(self.create, "create_task", "Create a persistent task."),
                tool(self.update, "update_task", "Add prerequisite task IDs."),
                tool(
                    self.background,
                    "background_bash",
                    "Run a shell command in the background and return its job ID.",
                    approval=True,
                ),
                tool(
                    self.background_status,
                    "background_status",
                    "Get a background command’s status and output.",
                ),
                tool(
                    self.schedule,
                    "schedule_cron",
                    "Schedule a prompt using a local five-field cron expression.",
                    approval=True,
                ),
                tool(self.list_crons, "list_crons", "List session schedules."),
                tool(self.cancel_cron, "cancel_cron", "Cancel a session schedule."),
            ]
            + self.workflows.tools(scope)
        )
