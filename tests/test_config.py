import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from main import load_settings, model_from_env


class ConfigurationTests(unittest.TestCase):
    def test_dotenv_supplies_model_and_reviewer_credentials_without_exporting_them(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp, ".env")
            path.write_text(
                "# Local\nexport DEEPSEEK_API_KEY='fixture'  # token\n"
                "AGENT_MODEL=custom\nREVIEWER_MODEL=fast\n",
                encoding="utf-8",
            )
            settings = load_settings(path)
            self.assertEqual(model_from_env(settings, "AGENT_").model, "custom")
            self.assertEqual(model_from_env(settings, "REVIEWER_").model, "fast")
            self.assertNotIn("DEEPSEEK_API_KEY", os.environ)

    def test_nonempty_local_key_wins_and_other_settings_keep_shell_overrides(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            path = Path(tmp, ".env")
            with self.assertRaisesRegex(ValueError, r"DEEPSEEK_API_KEY.*\.env"):
                model_from_env(load_settings(path), "AGENT_")
            os.environ["DEEPSEEK_API_KEY"] = "environment-value"
            os.environ["AGENT_MODEL"] = "shell-model"
            for contents in ("", "DEEPSEEK_API_KEY=\n"):
                path.write_text(contents, encoding="utf-8")
                self.assertEqual(load_settings(path)["DEEPSEEK_API_KEY"], "environment-value")
            path.write_text("DEEPSEEK_API_KEY=file-value\nAGENT_MODEL=file-model\n", encoding="utf-8")
            settings = load_settings(path)
            self.assertEqual(settings["DEEPSEEK_API_KEY"], "file-value")
            self.assertEqual(settings["AGENT_MODEL"], "shell-model")
            self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "environment-value")
