import copy
import json
import os
import re
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from adapters.tools import ShellRunner, builtin, run_shell
from app.context import ContextManager, Hooks
from domain import Scope, message


class ContextTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.root = self.base / "project"
        self.root.mkdir()
        self.state = self.root / ".agent"
        self.ctx = ContextManager(self.state, self.root)
        self.scope = Scope("session-a", cwd=self.root)
        self.tools = self._tools(self.scope)

    def _tools(self, scope):
        return {t.name: t.execute for t in self.ctx.tools(scope)}

    def test_file_tools_and_boundaries(self):
        tools = {t.name: t for t in builtin(self.scope)}
        tools["write_file"].execute({"path": "a/b.txt", "content": "pear pear\nnext"})
        tools["edit_file"].execute({"path": "a/b.txt", "old_text": "pear", "new_text": "apple"})
        self.assertEqual(tools["read_file"].execute({"path": "a/b.txt", "limit": 1}), "apple pear\n")
        self.assertEqual(tools["glob"].execute({"pattern": "**/*.txt"}), "a/b.txt")
        self.assertTrue(tools["bash"].approval)
        self.state.mkdir()
        (self.state / "secret.txt").touch()
        (self.root / "private").symlink_to(self.state)
        (self.root / "outside").symlink_to(self.base)
        for path in [".agent/secret.txt", "private/secret.txt", "outside/x", "../x"]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                tools["write_file"].execute({"path": path, "content": "bad"})
        self.assertNotIn("secret", tools["glob"].execute({"pattern": "**/*.txt"}))

    def test_dotenv_is_private_but_example_is_a_normal_file(self):
        dotenv = self.root / ".env"
        dotenv.write_text("DEEPSEEK_API_KEY=test-only-token")
        (self.root / "config.txt").symlink_to(dotenv)
        tools = {t.name: t.execute for t in builtin(self.scope)}
        for path in [".env", "config.txt"]:
            for name, args in [
                ("read_file", {}),
                ("write_file", {"content": "bad"}),
                ("edit_file", {"old_text": "token", "new_text": "bad"}),
            ]:
                with self.subTest(path=path, tool=name), self.assertRaises(ValueError):
                    tools[name]({"path": path, **args})
        tools["write_file"]({"path": ".env.example", "content": "DEEPSEEK_API_KEY="})
        self.assertEqual(tools["glob"]({"pattern": "*"}), ".env.example")
        self.assertEqual(tools["read_file"]({"path": ".env.example"}), "DEEPSEEK_API_KEY=")
        self.assertEqual(dotenv.read_text(), "DEEPSEEK_API_KEY=test-only-token")
        os.link(dotenv, self.root / "hardlink")
        with self.assertRaises(ValueError):
            tools["read_file"]({"path": "hardlink"})

    def test_tool_links_cannot_load_dotenv_as_a_skill_or_context(self):
        dotenv = self.root / ".env"
        dotenv.write_text("DEEPSEEK_API_KEY=test-only-token")
        skill = self.root / "skills" / "demo"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").symlink_to(dotenv)
        with self.assertRaisesRegex(ValueError, "Unknown skill"):
            self.tools["load_skill"]({"name": "demo"})
        (skill / "SKILL.md").unlink()
        (skill / ".env").write_text("DEEPSEEK_API_KEY=another-test-only-token")
        (skill / "SKILL.md").symlink_to(skill / ".env")
        with self.assertRaisesRegex(ValueError, "Unknown skill"):
            self.tools["load_skill"]({"name": "demo"})
        ref = self.ctx._archive(self.scope, "safe output")
        archived = self.ctx._dir(self.scope) / "archive" / ref
        archived.unlink()
        archived.symlink_to(dotenv)
        with self.assertRaisesRegex(ValueError, "Unknown context reference"):
            self.tools["read_context"]({"ref": ref})

    @unittest.skipUnless(os.name == "posix", "safe file descriptors")
    def test_link_cannot_redirect_a_write_after_validation(self):
        path = self.root / "a.txt"
        path.write_text("before")
        protected = self.base / "protected"
        protected.write_text("unchanged")

        def swap_after_validation(*_):
            path.unlink()
            path.symlink_to(protected)
            return path

        write = {t.name: t.execute for t in builtin(self.scope)}["write_file"]
        with (
            patch("adapters.tools._file", side_effect=swap_after_validation),
            self.assertRaises(OSError),
        ):
            write({"path": "a.txt", "content": "changed"})
        path.unlink()
        os.link(protected, path)
        with self.assertRaises(ValueError):
            write({"path": "a.txt", "content": "changed"})
        self.assertEqual(protected.read_text(), "unchanged")

    @unittest.skipUnless(os.name == "posix", "Unix shell syntax")
    def test_shell_failure_timeout_and_credentials(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "should-not-pass"}):
            self.assertEqual(run_shell(self.root, "printf '%s' \"$ANTHROPIC_API_KEY\""), "(no output)")
        with self.assertRaisesRegex(ValueError, "status 7"):
            run_shell(self.root, "exit 7")
        with (
            patch("adapters.tools.SHELL_TIMEOUT", 0.02),
            self.assertRaisesRegex(ValueError, "timed out"),
        ):
            run_shell(self.root, "sleep 3")

    def test_hook_veto_stops_dispatch(self):
        hooks, events = Hooks(), []
        hooks.on("PreToolUse", lambda tool: events.append(tool))
        hooks.on("PreToolUse", lambda _: "denied")
        hooks.on("PreToolUse", lambda _: self.fail("unexpected hook"))
        self.assertEqual(hooks.emit("PreToolUse", "bash"), "denied")
        self.assertEqual(events, ["bash"])

    @unittest.skipUnless(os.name == "posix", "Unix shell syntax")
    def test_shell_shutdown_also_terminates_worktree_commands(self):
        tree = self.root / ".agent" / "worktrees" / "worker"
        tree.mkdir(parents=True)
        shell = ShellRunner()
        self.addCleanup(shell.close)
        with ThreadPoolExecutor() as pool:
            future = pool.submit(shell.run, tree, "touch started; sleep 20")
            deadline = time.monotonic() + 2
            while not (tree / "started").exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            shell.close()
            self.assertTrue((tree / "started").exists())
            with self.assertRaisesRegex(ValueError, "Command exited"):
                future.result(timeout=2)
        with self.assertRaisesRegex(ValueError, "closed"):
            shell.run(self.root, "touch after-close")
        self.assertFalse((self.root / "after-close").exists())

    @unittest.skipUnless(os.name == "posix", "Unix shell syntax")
    def test_shell_output_remains_retrievable_after_compaction(self):
        text = "x" * 55_000 + "END"
        (self.root / "long").write_text(text)
        output = run_shell(self.root, "cat long")
        view = self.ctx.prepare(
            self.scope, [message("user", "job"), *self._pair("1", output)], lambda _: self.fail()
        )
        preview = view[-1]["content"][0]["content"]
        ref = re.search("ref=([0-9a-f]{64})", preview)[1]
        self.assertLess(len(preview), 6500)
        self.assertEqual(self.tools["read_context"]({"ref": ref, "offset": 55000}), "END")

    def test_skills_todos_and_reminder(self):
        skill = self.root / "skills" / "example"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: lint\ndescription: run checks\n---\nprivate body")
        self.assertIn("lint: run checks", self.ctx.instruction(self.scope, ""))
        self.assertNotIn("private body", self.ctx.instruction(self.scope, ""))
        self.assertIn("private body", self.tools["load_skill"]({"name": "lint"}))
        item = {"content": "inspect", "status": "in_progress"}
        self.tools["todo_write"]({"todos": [item]})
        with self.assertRaises(ValueError):
            self.tools["todo_write"]({"todos": [item, item]})
        history = [message("user", "task")]
        for n in range(3):
            history.extend(self._pair(str(n), "out"))
        self.ctx.prepare(self.scope, history, lambda _: "unused")
        self.assertIn("Check progress", self.ctx.instruction(self.scope, ""))
        self.assertNotIn("inspect", self.ctx.instruction(Scope("other", cwd=self.root), ""))

    @staticmethod
    def _pair(ident, text):
        return [
            message("assistant", [{"type": "tool_use", "id": ident, "name": "read_file", "input": {}}]),
            message("user", [{"type": "tool_result", "tool_use_id": ident, "content": text}]),
        ]

    def test_archive_paging_and_scope(self):
        contents = "长" * (self.ctx.OUTPUT_LIMIT + 12) + "END"
        history = [message("user", "job"), *self._pair("1", contents)]
        original = copy.deepcopy(history)
        output = self.ctx.prepare(self.scope, history, lambda _: self.fail())[2]["content"][0]["content"]
        ref = re.search("ref=([0-9a-f]{64})", output)[1]
        self.assertTrue(
            self.tools["read_context"]({"ref": ref, "offset": self.ctx.OUTPUT_LIMIT}).endswith("END")
        )
        with self.assertRaisesRegex(ValueError, "Unknown"):
            self._tools(Scope("other", cwd=self.root))["read_context"]({"ref": ref})
        self.assertEqual(history, original)

    def test_compaction_pairing_cache_and_recovery(self):
        history = [message("user", "find bug")]
        for n in range(8):
            history.extend(self._pair(str(n), "result"))
        original = copy.deepcopy(history)
        self.tools["compact"]({})
        prepared = self.ctx.prepare(self.scope, history, lambda _: "identified cause")
        self.assertIn("find bug", prepared[0]["content"])
        self.assertIn("identified cause", prepared[0]["content"])
        self.assertEqual(len(prepared[1:]) % 2, 0)
        for i in range(1, len(prepared), 2):
            self.assertEqual(prepared[i]["content"][0]["id"], prepared[i + 1]["content"][0]["tool_use_id"])
        fresh = ContextManager(self.state, self.root)
        self.assertEqual(fresh.prepare(self.scope, history, lambda _: self.fail("re-summarized")), prepared)
        self.assertEqual(history, original)

    def test_middle_transcript_is_archived_before_using_model(self):
        history = [message("user", "original request")]
        for n in range(30):
            history.extend(self._pair(str(n), f"result-{n}"))
        compacted = self.ctx.prepare(self.scope, history, lambda _: self.fail("unnecessary model call"))
        self.assertLess(len(compacted), len(history))
        ref = re.search("messages archived: read_context ref=([0-9a-f]{64})", str(compacted))[1]
        middle = json.loads(self.tools["read_context"]({"ref": ref}))
        self.assertEqual(middle[0]["role"], "assistant")
        self.assertIn("result-29", str(compacted[-1]))

    def test_memory_extraction_recall_and_child_isolation(self):
        hist = [message("user", "Remember I prefer pytest.")]
        self.ctx.remember(
            self.scope,
            hist,
            lambda _: json.dumps([{"name": "test preference", "type": "user", "body": "Use pytest."}]),
        )
        self.ctx.remember(self.scope, hist, lambda _: self.fail("re-extracted"))
        fresh = ContextManager(self.state, self.root)
        self.assertIn("Use pytest.", fresh.instruction(Scope("s2", cwd=self.root), "check code"))
        child = Scope("s2", "worker", self.root, True)
        self.assertNotIn("pytest", fresh.instruction(child, "pytest"))
        self.assertNotIn("memory_search", self._tools(child))
        fresh.remember(child, hist, lambda _: self.fail("child extracted memory"))
        self.tools["memory_save"]({"name": "test preference", "type": "user", "body": "Use unittest."})
        self.assertNotIn("Use pytest.", fresh.instruction(self.scope, "test"))


if __name__ == "__main__":
    unittest.main()
