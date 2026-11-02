"""Start a coding session in the terminal."""

import argparse
import os
import re
import shlex
import signal
from pathlib import Path

from adapters.models import Model
from adapters.store import JsonlSessionStore
from adapters.tools import ShellRunner, builtin
from app.context import ContextManager, Hooks
from app.service import AgentService
from app.work import Work

PROVIDERS = {
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek-flash"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-sonnet-4-6"),
    "openai": ("OPENAI_API_KEY", "gpt-5.4"),
    "google": ("GEMINI_API_KEY", "gemini-2.5-flash"),
}


def load_settings(path):
    """Read model settings without executing or exporting the contents of .env."""
    values = {}
    keys = {key for key, _ in PROVIDERS.values()}
    keys.update(
        prefix + name
        for prefix in ("AGENT_", "REVIEWER_")
        for name in ("PROVIDER", "MODEL", "EFFORT", "FALLBACK_MODEL")
    )
    if path.is_file():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            key, sep, raw = line.strip().removeprefix("export ").partition("=")
            key = key.strip()
            if sep and key in keys:
                try:
                    parts = shlex.split(raw, comments=True)
                except ValueError:
                    raise ValueError(f"Invalid value for {key} in {path}:{number}") from None
                values[key] = " ".join(parts)
    settings = values | dict(os.environ)
    for key, _ in PROVIDERS.values():
        if values.get(key):
            settings[key] = values[key]
    return settings


def model_from_env(settings, prefix, default="deepseek"):
    provider = settings.get(prefix + "PROVIDER", default)
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; choose from {', '.join(PROVIDERS)}")
    key, default_model = PROVIDERS[provider]
    if not settings.get(key):
        raise ValueError(f"Set {key} in .env or the environment to use {provider}")
    return Model(
        provider,
        settings.get(prefix + "MODEL", default_model),
        settings[key],
        settings.get(prefix + "EFFORT", "off"),
        settings.get(prefix + "FALLBACK_MODEL"),
    )


def check_command(name, arguments):
    if name in {"bash", "background_bash"}:
        command = arguments.get("command", "")
        if re.search(r"\bsudo\b|\b(?:shutdown|reboot|mkfs)\b|\brm\s+-[rfRF]+\s+/(?:\s|$)", command):
            return "Host policy blocks this privileged or destructive command."


def build(workspace, model=None, reviewer=None):
    workspace = Path(workspace).resolve(strict=True)
    state = workspace / ".agent"
    if state.is_symlink():
        raise ValueError(".agent must be a local directory")
    settings = load_settings(Path.cwd() / ".env")
    if model is None:
        model = model_from_env(settings, "AGENT_")
    if reviewer is None:
        if any(settings.get("REVIEWER_" + k) for k in ("MODEL", "PROVIDER", "EFFORT", "FALLBACK_MODEL")):
            reviewer = model_from_env(settings, "REVIEWER_", settings.get("AGENT_PROVIDER", "deepseek"))
        else:
            reviewer = model
    state.mkdir(mode=0o700, exist_ok=True)
    hooks = Hooks()
    hooks.on("PreToolUse", check_command)
    shell = ShellRunner()
    return AgentService(
        model,
        reviewer,
        JsonlSessionStore(state / "sessions"),
        ContextManager(state, workspace),
        Work(state, workspace),
        hooks,
        workspace,
        lambda scope: builtin(scope, shell.run),
        shell,
    )


def main():
    from adapters.cli import chat

    parser = argparse.ArgumentParser(prog="agent", description="Chat with a coding agent in your terminal")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--resume", metavar="SESSION_ID", help="continue a saved conversation")
    args = parser.parse_args()
    try:
        service = build(args.workspace)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))

    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        chat(service, args.resume)
    except (EOFError, KeyboardInterrupt):
        print()
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    finally:
        service.close()


if __name__ == "__main__":
    main()
