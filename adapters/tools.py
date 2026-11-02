"""Workspace file tools and the approval-gated shell tool."""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import tempfile
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from threading import Lock

from domain import Scope, Tool, schema

SHELL_TIMEOUT = 120
PROTECTED_NAMES = {".agent", ".env"}


def _kill_group(proc: subprocess.Popen) -> None:
    if os.name == "nt":
        if proc.poll() is None:
            proc.kill()
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class ShellRunner:
    """One runner owns every foreground and background command of a service."""

    def __init__(self):
        self._lock = Lock()
        self._active = set()
        self._closed = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for proc in self._active:
                _kill_group(proc)

    def run(self, cwd: Path, command: str) -> str:
        if not isinstance(command, str) or not command.strip():
            raise ValueError("command must be a non-empty string")
        # Model credentials and other inherited secrets have no use in child processes.
        allowed = ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT")
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        argv = (
            ["cmd.exe", "/d", "/c", command]
            if os.name == "nt"
            else ["/bin/bash", "--noprofile", "--norc", "-c", command]
        )
        with tempfile.TemporaryFile() as stream:
            with self._lock:
                if self._closed:
                    raise ValueError("Shell runner closed")
                proc = subprocess.Popen(
                    argv,
                    cwd=cwd,
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                )
                self._active.add(proc)
            try:
                proc.wait(timeout=SHELL_TIMEOUT)
            except subprocess.TimeoutExpired as exc:
                raise ValueError(f"Command timed out after {SHELL_TIMEOUT}s") from exc
            finally:
                with self._lock:
                    # Clean up children in the process group even when the shell exited normally.
                    _kill_group(proc)
                    self._active.remove(proc)
                proc.wait()
            stream.seek(0)
            # ContextManager archives the full result and sends a small preview to the model.
            output = stream.read().decode("utf-8", "replace")
        if proc.returncode:
            raise ValueError(f"Command exited with status {proc.returncode}\n{output}")
        return output or "(no output)"


def run_shell(cwd: Path, command: str) -> str:
    return ShellRunner().run(cwd, command)


def _file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("path must be a non-empty string")
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(s.casefold() in PROTECTED_NAMES for s in relative.parts)
    ):
        raise ValueError("File access is restricted to workspace files")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or any(
        s.casefold() in PROTECTED_NAMES for s in path.relative_to(root).parts
    ):
        raise ValueError("File access is restricted to workspace files")
    return path


@contextmanager
def _open_file(root: Path, name: str, mode: str):
    path = _file(root, name)
    parts = path.relative_to(root).parts
    if not parts:
        raise ValueError("Expected a file path")
    if os.name != "posix":
        if mode == "w":
            path.parent.mkdir(parents=True, exist_ok=True)
        with path.open(mode, encoding="utf-8") as stream:
            yield stream
        return

    # Bind each component to a directory descriptor. A concurrent symlink swap
    # after _file resolves a valid path cannot redirect a read or write outside it.
    dir_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory = os.open(root, dir_flags)
    try:
        for part in parts[:-1]:
            if mode == "w":
                try:
                    os.mkdir(part, dir_fd=directory)
                except FileExistsError:
                    pass
            child = os.open(part, dir_flags, dir_fd=directory)
            os.close(directory)
            directory = child
        access = {"r": os.O_RDONLY, "w": os.O_WRONLY | os.O_CREAT, "r+": os.O_RDWR}[mode]
        fd = os.open(parts[-1], access | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, mode, encoding="utf-8") as stream:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Only regular files are supported")
        if metadata.st_nlink > 1:
            dotenv = root / ".env"
            if mode != "r" or dotenv.is_file() and os.path.samestat(metadata, dotenv.stat()):
                raise ValueError("File access is restricted through hard links")
        if mode == "w":
            stream.truncate(0)
        yield stream


def builtin(scope: Scope, shell=run_shell) -> list[Tool]:
    root = Path(scope.cwd).resolve()

    def read(a: dict) -> str:
        limit = a.get("limit")
        if limit is not None and (type(limit) is not int or limit < 1):
            raise ValueError("limit must be a positive integer")
        with _open_file(root, a["path"], "r") as reader:
            if limit is None:
                return reader.read()
            return "".join(islice(reader, limit))

    def write(a: dict) -> str:
        if not isinstance(a["content"], str):
            raise TypeError("content must be a string")
        with _open_file(root, a["path"], "w") as writer:
            writer.write(a["content"])
        return f"Wrote {a['path']}"

    def edit(a: dict) -> str:
        old = a["old_text"]
        with _open_file(root, a["path"], "r+") as stream:
            original = stream.read()
            if not old or old not in original:
                raise ValueError("old_text must match existing text")
            updated = original.replace(old, a["new_text"], 1)
            stream.seek(0)
            stream.write(updated)
            stream.truncate()
        return f"Edited {a['path']}"

    def glob(a: dict) -> str:
        pattern = a["pattern"]
        _file(root, pattern)
        found = []
        for path in root.glob(pattern):
            relative = str(path.relative_to(root))
            try:
                _file(root, relative)
            except (ValueError, RuntimeError, OSError):
                continue
            found.append(relative)
            if len(found) > 200:
                break
        shown = sorted(found[:200])
        if len(found) > 200:
            shown.append("[more matches omitted]")
        return "\n".join(shown) or "(no matches)"

    read_schema = schema(path="string")
    read_schema["properties"]["limit"] = {"type": "integer", "minimum": 1}
    return [
        Tool(
            "bash",
            "Run a shell command in the workspace.",
            schema(command="string"),
            lambda a: shell(root, a["command"]),
            approval=True,
        ),
        Tool("read_file", "Read a UTF-8 file, optionally limiting lines.", read_schema, read),
        Tool(
            "write_file", "Create or overwrite a UTF-8 file.", schema(path="string", content="string"), write
        ),
        Tool(
            "edit_file",
            "Replace the first exact match in a UTF-8 file.",
            schema(path="string", old_text="string", new_text="string"),
            edit,
        ),
        Tool("glob", "Find workspace paths matching a glob, including **.", schema(pattern="string"), glob),
    ]
