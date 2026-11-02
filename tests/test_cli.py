import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from adapters.cli import chat
from main import build
from tests.test_service import Script, call


class TerminalTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def service(self, model, *reviews):
        service = build(self.root, model, Script(*reviews))
        self.addCleanup(service.close)
        return service

    def interact(self, service, lines, session=None):
        output = io.StringIO()
        with redirect_stdout(output), patch("builtins.input", side_effect=lines):
            chat(service, session)
        return output.getvalue()

    def test_chat_and_resume_in_same_terminal(self):
        (self.root / "sample").write_text("pear")
        model = Script(call("read_file", {"path": "sample"}), "found pear", "still pear")
        service = self.service(model, '{"allow":true}')
        out = self.interact(service, ["/help", "read sample", "/exit"])
        (sid,) = service.list_sessions()
        self.assertIn(sid, out)
        self.assertIn("read_file", out)
        self.assertIn("found pear", out)
        self.assertIn("/resume", out)
        resumed = self.interact(service, ["what did it say?", EOFError()], sid)
        self.assertIn("found pear", resumed)
        self.assertIn("still pear", resumed)
        self.assertIn("found pear", str(model.inputs[-1]))

    def test_confirmation_displays_arguments_and_defaults_to_no(self):
        model = Script(
            call("bash", {"command": "echo first > a"})
            + call("bash", {"command": "echo second > b"}, "second"),
            "finished",
        )
        service = self.service(model, '{"allow":true}', '{"allow":true}')
        text = self.interact(service, ["run commands", "", "y", "/exit"])
        self.assertIn("echo first > a", text)
        self.assertIn("echo second > b", text)
        self.assertFalse((self.root / "a").exists())
        self.assertEqual((self.root / "b").read_text(), "second\n")
        self.assertIn("finished", text)

    def test_exit_pending_and_restore_approval(self):
        service = self.service(
            Script(call("bash", {"command": "echo ran > a"}), "complete"), '{"allow":true}'
        )
        self.interact(service, ["write a", EOFError()])
        (sid,) = service.list_sessions()
        self.assertFalse((self.root / "a").exists())
        self.assertEqual(len(service.view(sid)["approvals"]), 1)
        out = self.interact(service, ["y", "/exit"], sid)
        self.assertIn("complete", out)
        self.assertEqual((self.root / "a").read_text(), "ran\n")
