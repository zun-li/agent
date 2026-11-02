import json
import os
import unittest
from io import BytesIO
from unittest.mock import Mock, patch
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPSHandler, build_opener

from adapters.models import ContextLimitError, Model, ModelError

USER = {"role": "user", "content": "Check the file"}
SCHEMA = {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}
TOOLS = [{"name": "read_file", "description": "Read a file", "input_schema": SCHEMA}]
CALL = {"type": "tool_use", "id": "call_abc", "name": "read_file", "input": {"path": "x"}}
RESULT = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "call_abc", "content": "hello"}]}


def response(payload, status=200):
    return Mock(status=status, read=Mock(return_value=json.dumps(payload).encode()))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.addCleanup(patch.stopall)
        self.send = patch("adapters.models.build_opener").start().return_value.open
        self.sleep = patch("adapters.models.time.sleep").start()

    def invoke(self, provider, model, payload, effort="off", messages=None):
        self.send.return_value = response(payload)
        reply = Model(provider, model, "test-key", effort).generate(
            "Be helpful", messages or [USER], TOOLS, max_tokens=2048
        )
        request = self.send.call_args.args[0]
        headers = {key.lower(): value for key, value in request.header_items()}
        return reply, urlsplit(request.full_url).path, json.loads(request.data), headers

    def test_anthropic_protocols_preserve_entire_assistant_content(self):
        thinking = {"type": "thinking", "thinking": "", "signature": "opaque"}
        payload = {"content": [thinking, CALL], "stop_reason": "tool_use"}
        for provider, model, path in [
            ("deepseek", "deepseek-v4-pro", "/anthropic/v1/messages"),
            ("anthropic", "claude-sonnet-4-6", "/v1/messages"),
        ]:
            with self.subTest(provider=provider):
                reply, url, req, headers = self.invoke(provider, model, payload, effort="high")
                self.assertEqual(url, path)
                self.assertEqual(headers["x-api-key"], "test-key")
                self.assertEqual(req["tools"], TOOLS)
                self.assertEqual(req["output_config"], {"effort": "high"})
                self.assertEqual(reply, {"role": "assistant", **payload})
                _, _, followup, _ = self.invoke(provider, model, payload, messages=[USER, reply, RESULT])
                self.assertEqual(followup["messages"][1], {"role": "assistant", "content": [thinking, CALL]})
                self.assertEqual(followup["messages"][2], RESULT)

    def test_openai_tool_result_round_trip(self):
        function = {"name": "read_file", "arguments": '{"path":"x"}'}
        message = {
            "content": None,
            "tool_calls": [{"id": CALL["id"], "type": "function", "function": function}],
        }
        payload = {"choices": [{"message": message, "finish_reason": "tool_calls"}]}
        reply, path, req, headers = self.invoke("openai", "gpt-5.4", payload, "medium")
        self.assertEqual(reply, {"role": "assistant", "content": [CALL], "stop_reason": "tool_use"})
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers["authorization"], "Bearer test-key")
        self.assertEqual(req["reasoning_effort"], "medium")
        self.assertEqual(req["tools"][0]["function"]["parameters"], SCHEMA)
        _, _, req, _ = self.invoke("openai", "gpt-5.4", payload, messages=[USER, reply, RESULT])
        self.assertEqual(
            req["messages"][-1], {"role": "tool", "tool_call_id": "call_abc", "content": "hello"}
        )
        self.assertEqual(req["messages"][-2]["tool_calls"][0]["id"], "call_abc")
        self.assertEqual(
            json.loads(req["messages"][-2]["tool_calls"][0]["function"]["arguments"]), CALL["input"]
        )

    def test_gemini_signature_and_wire_call_id_round_trip(self):
        for wire_id in ["call_abc", None]:
            with self.subTest(wire_id=wire_id):
                function = {"name": "read_file", "args": {"path": "x"}}
                if wire_id:
                    function["id"] = wire_id
                parts = [
                    {"text": "", "thought": True, "thoughtSignature": "hidden"},
                    {"functionCall": function, "thoughtSignature": "signature"},
                ]
                payload = {
                    "candidates": [{"content": {"role": "model", "parts": parts}, "finishReason": "STOP"}]
                }
                reply, path, req, headers = self.invoke("google", "gemini-2.5-flash", payload)
                self.assertEqual(path, "/v1beta/models/gemini-2.5-flash:generateContent")
                self.assertEqual(headers["x-goog-api-key"], "test-key")
                self.assertEqual(req["generationConfig"]["thinkingConfig"]["thinkingBudget"], 0)
                self.assertEqual(reply["stop_reason"], "tool_use")
                call = next(b for b in reply["content"] if b["type"] == "tool_use")
                result = {"role": "user", "content": [{**RESULT["content"][0], "tool_use_id": call["id"]}]}
                _, _, req, _ = self.invoke(
                    "google", "gemini-2.5-flash", payload, messages=[USER, reply, result]
                )
                self.assertEqual(req["contents"][-2], {"role": "model", "parts": parts})
                returned = req["contents"][-1]["parts"][0]["functionResponse"]
                self.assertEqual(returned["name"], "read_file")
                self.assertEqual(returned.get("id"), wire_id)
                self.assertEqual(returned["response"], {"output": "hello"})

    def test_partial_function_arguments_are_never_executable(self):
        payload = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "a",
                                "type": "function",
                                "function": {"name": "read_file", "arguments": '{"p'},
                            }
                        ],
                    },
                }
            ]
        }
        reply, *_ = self.invoke("openai", "gpt-5.4", payload)
        self.assertEqual(reply["stop_reason"], "max_tokens")
        self.assertFalse(any(block["type"] == "tool_use" for block in reply["content"]))

    def test_transient_errors_have_bounded_retries(self):
        client = Model("deepseek", "deepseek-flash", "test-key")
        unavailable = response({"error": {"message": "overloaded"}}, 529)
        good = response({"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"})
        self.send.side_effect = [unavailable, good]
        self.assertEqual(client.generate("", [USER], [])["content"][0]["text"], "done")
        self.assertEqual(self.sleep.call_count, 1)
        self.send.side_effect = [unavailable] * 3
        with self.assertRaisesRegex(ModelError, "529"):
            client.generate("", [USER], [])

    def test_input_limit_errors_request_compaction_without_retry(self):
        self.send.return_value = response({"error": {"message": "prompt is too long"}}, 400)
        with self.assertRaises(ContextLimitError):
            Model("deepseek", "deepseek-flash", "test-key").generate("", [USER], [])
        self.sleep.assert_not_called()

    def test_bad_output_is_reported_and_credentials_are_not_exposed(self):
        payload = {"choices": [{"message": {"content": None}, "finish_reason": "content_filter"}]}
        with self.assertRaises(ModelError):
            self.invoke("openai", "gpt-5.4", payload)
        self.send.side_effect = HTTPError(
            "https://example.com",
            401,
            "Unauthorized",
            {},
            BytesIO(b'{"error": {"message": "invalid key test-key"}}'),
        )
        with self.assertRaises(ModelError) as error:
            Model("deepseek", "deepseek-flash", "test-key").generate("", [USER], [])
        self.assertIn("HTTP 401", str(error.exception))
        self.assertIn(".env", str(error.exception))
        self.assertNotIn("invalid key", str(error.exception))
        self.assertNotIn("test-key", str(error.exception))
        self.sleep.assert_not_called()

    def test_fallback_only_after_exhausting_overload_retries(self):
        client = Model("deepseek", "preferred", "test-key", fallback_model="alternate")
        overloaded = response({}, 503)
        good = response({"content": [{"type": "text", "text": "done"}], "stop_reason": "end_turn"})
        self.send.side_effect = [overloaded] * 3 + [good, good]
        client.generate("", [USER], [])
        client.generate("", [USER], [])
        models = [json.loads(args[0][0].data)["model"] for args in self.send.call_args_list]
        self.assertEqual(models, ["preferred"] * 3 + ["alternate", "preferred"])
        self.send.reset_mock()
        self.send.side_effect = [response({}, 429)] * 3
        with self.assertRaisesRegex(ModelError, "429"):
            client.generate("", [USER], [])
        self.assertEqual(self.send.call_count, 3)

    def test_proxy_and_no_proxy_select_connect_target_without_sending_credentials(self):
        # Exercise urllib's real proxy routing, stopping just before the TLS connection.
        good = response({"content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn"})
        good.code = 200
        client = Model("deepseek", "deepseek-flash", "test-key")
        for bypass, host in [("", "localhost:3128"), ("api.deepseek.com", "api.deepseek.com")]:
            with (
                self.subTest(bypass=bypass),
                patch.dict(
                    os.environ, {"https_proxy": "http://localhost:3128", "no_proxy": bypass}, clear=True
                ),
                patch("adapters.models.build_opener", build_opener),
                patch.object(HTTPSHandler, "https_open", return_value=good) as send,
            ):
                client.generate("", [USER], [])
                request = send.call_args.args[0]
                self.assertEqual(request.type, "https")
                self.assertEqual(request.host, host)
