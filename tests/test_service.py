import tempfile
import unittest
from pathlib import Path

from domain import message
from main import build


class Script:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.inputs = []

    def generate(self, instruction, messages, tools, max_tokens=4096):
        self.inputs.append(messages)
        return message("assistant", self.replies.pop(0))


def call(name, args, ident="call-1"):
    return [{"type": "tool_use", "id": ident, "name": name, "input": args}]


class ServiceTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)

    def service(self, model, reviewer=None):
        service = build(self.root, model, reviewer or Script('{"allow":true}'))
        self.addCleanup(service.close)
        return service

    def test_read_then_restart(self):
        (self.root / "sample").write_text("pear")
        model = Script(call("read_file", {"path": "sample"}), "found pear")
        service = self.service(model)
        session = service.new_session()
        self.assertEqual(service.chat(session, "read sample")["text"], "found pear")
        result = model.inputs[1][-1]["content"][0]
        self.assertEqual(result["tool_use_id"], "call-1")
        self.assertIn("pear", result["content"])
        service.close()
        next_model = Script("pear again")
        new = self.service(next_model)
        self.assertEqual(new.chat(session, "what did it say?")["text"], "pear again")
        self.assertIn("found pear", str(next_model.inputs))

    def test_review_denial(self):
        model = Script(call("write_file", {"path": "bad", "content": "x"}), "denied")
        service = self.service(model, Script('{"allow":false,"reason":"outside request"}'))
        session = service.new_session()
        service.chat(session, "just inspect")
        self.assertIn("outside request", str(model.inputs[-1]))
        self.assertFalse((self.root / "bad").exists())

    def test_approval_resumes_persisted_batch_once(self):
        model = Script(
            call("bash", {"command": "echo yes >> marker"}) + call("read_file", {"path": "marker"}, "two"),
            "ok",
        )
        service = self.service(model, Script('{"allow":true}', '{"allow":true}'))
        session = service.new_session()
        reply = service.chat(session, "make marker")
        self.assertEqual(reply["status"], "approval")
        self.assertFalse((self.root / "marker").exists())
        service.approve(session, reply["approval"]["id"], True)
        with self.assertRaises(ValueError):
            service.approve(session, reply["approval"]["id"], True)
        self.assertEqual((self.root / "marker").read_text().strip(), "yes")
        self.assertEqual(len(model.inputs[-1][-1]["content"]), 2)

    def test_approval_survives_restart(self):
        service = self.service(Script(call("bash", {"command": "echo restarted > marker"})))
        session = service.new_session()
        pending = service.chat(session, "write marker")["approval"]
        service.close()
        service = self.service(Script("written"), Script())
        self.assertEqual(service.view(session)["approvals"][0]["id"], pending["id"])
        self.assertEqual(service.approve(session, pending["id"], True)["status"], "done")
        self.assertEqual((self.root / "marker").read_text().strip(), "restarted")

    def test_recovery_does_not_repeat_unknown_side_effect(self):
        model = Script(call("bash", {"command": "echo twice >> marker"}))
        service = self.service(model)
        sid = service.new_session()
        request = service.chat(sid, "append once")["approval"]
        # Simulate power loss between the command returning and storing its result.
        for event in ({"kind": "decision", "allow": True}, {"kind": "started"}):
            service.store.append(sid, {"actor": "lead", "id": request["id"], **event})
        (self.root / "marker").write_text("once\n")
        service.close()
        recovered = Script("needs inspection")
        service = self.service(recovered)
        self.assertEqual(service.resume(sid)["status"], "done")
        self.assertIn("Outcome unknown", str(recovered.inputs[-1]))
        self.assertEqual((self.root / "marker").read_text(), "once\n")

    def test_notices_during_approval_delivered_once_afterwards(self):
        import time

        model = Script(call("bash", {"command": "echo ok"}), "approved", "received notice")
        service = self.service(model)
        sid = service.new_session()
        pending = service.chat(sid, "run echo")["approval"]
        service.notify(sid, "build-42", "build completed")
        service.notify(sid, "build-42", "build completed")
        service.approve(sid, pending["id"], True)
        for _ in range(100):
            if len(model.inputs) == 3:
                break
            time.sleep(0.01)
        events = service.store.read(sid)
        self.assertEqual(sum("build-42" in e.get("deliveries", []) for e in events), 1)
        self.assertEqual(len(model.inputs), 3)
        self.assertIn("build completed", str(model.inputs[-1][-1]))

    def test_goal_requires_evidence_and_remembers_failed_assessment(self):
        model = Script("I think so", call("read_file", {"path": "proof"}), "verified")
        reviewer = Script(
            '{"state":"continue","reason":"read the evidence"}',
            '{"allow":true}',
            '{"state":"complete","reason":"proof exists"}',
        )
        (self.root / "proof").write_text("green")
        service = self.service(model, reviewer)
        sid = service.new_session()
        result = service.chat(sid, "/goal Verify proof is green")
        self.assertEqual(result, {"status": "done", "text": "verified"})
        self.assertIn("read the evidence", str(model.inputs[1]))
        self.assertIn("green", str(reviewer.inputs[-1]))
        self.assertEqual(service.chat(sid, "/goal")["text"], "")

    def test_goal_waits_for_background_evidence(self):
        from unittest.mock import patch

        service = self.service(Script("waiting", "it passed"), Script('{"state":"complete"}'))
        sid = service.new_session()
        with patch.object(service.work, "busy", return_value=True):
            self.assertEqual(service.chat(sid, "/goal wait for test")["status"], "waiting")
        import time

        service.notify(sid, "test-1", "pytest exit 0")
        for _ in range(100):
            if not service.view(sid)["goal"]:
                break
            time.sleep(0.01)
        self.assertEqual(service.view(sid)["goal"], "")

    def test_host_deny_precedes_approval(self):
        model = Script(call("bash", {"command": "sudo echo should-never-run"}), "blocked")
        service = self.service(model, Script())
        sid = service.new_session()
        self.assertEqual(service.chat(sid, "run it")["status"], "done")
        self.assertIn("Host policy", str(model.inputs[-1]))
        self.assertEqual(service.view(sid)["approvals"], [])

    def test_restart_resumes_notification_already_in_history(self):
        import time

        service = self.service(Script())
        sid = service.new_session()
        service.store.append(sid, {"actor": "lead", "kind": "notice", "id": "timer-1", "text": "run tests"})
        service.store.append(
            sid,
            {
                "actor": "lead",
                "kind": "message",
                "turn": True,
                "input_source": "runtime",
                "deliveries": ["timer-1"],
                "message": message("user", "Scheduled: run tests"),
            },
        )
        service.close()
        model = Script("ran")
        service = self.service(model)
        for _ in range(100):
            if service.view(sid)["status"] == "done":
                break
            time.sleep(0.01)
        self.assertEqual(service.view(sid)["status"], "done")
        self.assertEqual(len(model.inputs), 1)
        self.assertEqual(len(service.view(sid)["messages"]), 2)

    def test_delegate_has_its_own_history_and_returns_only_summary(self):
        model = Script(
            call("delegate", {"task": "inspect sample"}),
            call("read_file", {"path": "sample"}),
            "child summary",
            "parent answer",
        )
        reviewer = Script('{"allow":true}', '{"allow":true}')
        (self.root / "sample").write_text("raw child evidence")
        service = self.service(model, reviewer)
        sid = service.new_session()
        self.assertEqual(service.chat(sid, "parent request")["text"], "parent answer")
        self.assertEqual(model.inputs[1], [message("user", "inspect sample")])
        self.assertIn("child summary", str(model.inputs[-1]))
        self.assertNotIn("raw child evidence", str(model.inputs[-1]))
