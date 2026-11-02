import subprocess
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from app.work import Work, cron_matches
from domain import Scope


class WorkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.scope = Scope("session-a", cwd=self.repo)
        self.work = Work(self.root / "state", self.repo)
        self.addCleanup(self.work.close)

    def until(self, read, timeout=3):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            result = read()
            if result:
                return result
            time.sleep(0.02)
        self.fail("event did not arrive")

    def test_task_graph_atomic_claim_and_tenant(self):
        first, second = [self.work.create(self.scope, s)["id"] for s in ["schema", "api"]]
        self.work.update(self.scope, second, [first])
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.work.update(self.scope, first, [second])
        with self.assertRaisesRegex(ValueError, "blocked"):
            self.work.claim(self.scope, second)

        def claim(owner):
            try:
                self.work.claim(Scope(self.scope.session, actor=owner, cwd=self.repo), first)
                return owner
            except ValueError:
                return None

        with ThreadPoolExecutor(max_workers=2) as workers:
            winners = list(filter(None, workers.map(claim, ["a", "b"])))
        self.assertEqual(len(winners), 1)
        with self.assertRaises(PermissionError):
            self.work.complete(self.scope, first)
        result = self.work.complete(Scope(self.scope.session, winners[0], self.repo), first)
        self.assertEqual(result["unblocked"], [second])
        other_session = Scope("session-b", cwd=self.repo)
        self.assertEqual({r["id"] for r in self.work.list_tasks(other_session)}, {first, second})
        self.work.claim(other_session, second)
        with self.assertRaises(PermissionError):
            self.work.complete(self.scope, second)
        self.assertEqual(self.work.complete(other_session, second)["owner_session"], "session-b")
        self.assertEqual(Work(self.work.state, self.repo).get(self.scope, first)["status"], "completed")

    def test_background_delivers_success_or_failure_separately(self):
        done = threading.Event()
        events = []

        def shell(cwd, command):
            self.assertEqual(cwd, self.repo)
            done.wait(1)
            if command == "bad":
                raise ValueError("exit 1")
            return "output"

        self.work.start(None, lambda *event: events.append(event), shell)
        tools = {t.name: t for t in self.work.tools(self.scope)}
        self.assertTrue(tools["background_bash"].approval)
        jobs = [tools["background_bash"].execute({"command": c}) for c in ["ok", "bad"]]
        self.assertTrue(self.work.busy(self.scope.session))
        done.set()
        self.until(lambda: len(events) == 2)
        self.assertEqual(
            [self.work.background_status(self.scope, i)["status"] for i in jobs], ["completed", "failed"]
        )
        self.assertEqual({e[1] for e in events}, set(jobs))
        self.until(lambda: not self.work.busy(self.scope.session))

    def test_cron_single_recurring_and_durable(self):
        wed = datetime(2026, 11, 4, 9, 15)
        self.assertTrue(cron_matches("*/15 9 * * 1-5", wed))
        with self.assertRaises(ValueError):
            self.work.schedule(self.scope, "60 3 * * *", "x")
        single = self.work.schedule(self.scope, "15 9 * * *", "once", recurring=False, durable=True)
        repeating = self.work.schedule(self.scope, "15 9 * * *", "repeat", durable=True)
        self.work.schedule(self.scope, "* * * * *", "temporary")
        next_host = Work(self.work.state, self.repo)
        self.assertEqual({r["id"] for r in next_host.list_crons(self.scope)}, {single, repeating})
        next_host.tick(wed)
        next_host.tick(wed)
        with next_host.db() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0], 2)
        self.assertEqual([r["id"] for r in next_host.list_crons(self.scope)], [repeating])
        next_host.cancel_cron(self.scope, repeating)

    def test_team_plan_messages_approval_and_shutdown(self):
        events = []

        def respond(scope, prompt):
            if "Assigned task:" in prompt:
                self.work.guard(scope, "read_file")
                with self.assertRaises(PermissionError):
                    self.work.guard(scope, "write_file")
                self.work.teams.submit_plan(scope, "fix docs")
                return "waiting for approval"
            if "approved" in prompt:
                self.work.guard(scope, "write_file")
                self.work.complete(scope, task_id, "finished")
                return "finished"
            return "received"

        self.work.start(respond, lambda *event: events.append(event), lambda *_: "ok")
        task_id = self.work.create(self.scope, "documentation")["id"]
        self.work.teams.spawn(self.scope, "writer", "technical writer", task_id, True)
        self.until(lambda: self.work.teams.list(self.scope)[0]["plan"] == "pending")
        m = self.work.teams.list(self.scope)[0]
        with self.assertRaises(ValueError):
            self.work.teams.review(self.scope, "writer", "stale id", True)
        self.work.teams.review(self.scope, "writer", m["request"], True)
        self.until(lambda: self.work.get(self.scope, task_id)["status"] == "completed")
        second = self.work.create(self.scope, "second task")["id"]
        self.until(lambda: self.work.get(self.scope, second)["owner"] == "writer")
        request = self.work.teams.shutdown(self.scope, "writer")
        self.until(lambda: self.work.teams.list(self.scope)[0]["status"] == "closed")
        self.until(lambda: any(request == e[1] for e in events))
        self.assertEqual(self.work.get(self.scope, second)["status"], "pending")

    def test_worktree_binding_and_cleanup(self):
        def git(*args):
            subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, check=True)

        git("init", "-q")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "--allow-empty",
            "-qm",
            "base",
        )
        task_id = self.work.create(self.scope, "isolated checkout")["id"]
        path = Path(self.work.teams.create_tree(self.scope, task_id, "feature"))
        with self.work.db() as db:
            db.execute(
                "INSERT INTO mates(session,name,role,status) VALUES(?, 'dev', '', 'idle')",
                (self.scope.session,),
            )
        dev = Scope(self.scope.session, "dev", self.repo)
        self.work.claim(dev, task_id)
        self.assertEqual(self.work.scope(dev).cwd, path)
        (path / "local.txt").write_text("uncommitted")
        self.assertFalse((self.repo / "local.txt").exists())
        self.work.complete(dev, task_id)
        with self.assertRaisesRegex(ValueError, "assigned"):
            self.work.teams.remove_tree(self.scope, task_id)
        with self.work.db() as db:
            db.execute("UPDATE mates SET task=NULL WHERE name='dev'")
        with self.assertRaisesRegex(ValueError, "local changes"):
            self.work.teams.remove_tree(self.scope, task_id)
        self.work.teams.remove_tree(self.scope, task_id, discard_changes=True)
        self.assertFalse(path.exists())
        self.assertIsNone(self.work.get(self.scope, task_id)["worktree"])

    def test_workflow_failure_resume_and_schema_retry(self):
        prompts = []

        def child(scope, prompt):
            prompts.append(prompt)
            with self.assertRaises(PermissionError):
                self.work.guard(scope, "bash")
            if prompt.startswith("b") and len(prompts) < 4:
                return "bad"
            return '{"ok":true}'

        self.work.start(child, lambda *_: None, lambda *_: "")
        calls = self.work.workflows

        def flow(ctx, args):
            schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
            ctx.phase("Check")
            ctx.agent("a", "a", schema)
            return ctx.agent("b", "b", schema)

        calls.register("checks", "Run checks", flow)
        first = calls.run(self.scope, "checks", {})
        self.assertEqual(first["status"], "failed")
        second = calls.run(self.scope, "checks", resume_from_run_id=first["run_id"])
        self.assertEqual(second["result"], {"ok": True})
        self.assertEqual(sum(p.startswith("a") for p in prompts), 1)
        self.assertIn("cached: a", calls.status(self.scope, first["run_id"])["events"])
        with self.assertRaisesRegex(ValueError, "original session"):
            calls.run(Scope("session-b"), "checks", resume_from_run_id=first["run_id"])
        with self.assertRaisesRegex(ValueError, "Unknown saved workflow"):
            calls.run(self.scope, "run-user-script", {})
