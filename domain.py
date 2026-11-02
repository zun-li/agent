"""Shared vocabulary. Tool arguments use JSON Schema, messages use typed content blocks."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from inspect import Parameter, signature
from pathlib import Path
from typing import Protocol


class ModelError(RuntimeError):
    """A model request failed or produced an unusable response."""


class ContextLimitError(ModelError):
    """The input must be compacted before attempting the request again."""


@dataclass(frozen=True)
class Scope:
    session: str
    actor: str = "lead"
    cwd: Path = Path(".")
    child: bool = False


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    execute: Callable[[dict], str]
    approval: bool = False

    def spec(self):
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}

    @classmethod
    def bind(cls, scope, method, name, description, approval=False):
        """Publish a scoped task method; its signature is also its tool contract."""
        handler = partial(method, scope)
        params = signature(handler).parameters
        types = {
            str: "string",
            str | None: "string",
            bool: "boolean",
            dict | None: "object",
            list[str]: {"type": "array", "items": {"type": "string"}},
        }
        parameters = schema(**{key: types[p.annotation] for key, p in params.items()})
        parameters["required"] = [key for key, p in params.items() if p.default is Parameter.empty]

        def execute(arguments):
            result = handler(**arguments)
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)

        return cls(name, description, parameters, execute, approval)


def schema(**fields):
    return {
        "type": "object",
        "properties": {
            name: {"type": field} if isinstance(field, str) else field for name, field in fields.items()
        },
        "required": list(fields),
        "additionalProperties": False,
    }


class ModelClient(Protocol):
    def generate(
        self, instruction: str, messages: list[dict], tools: list[dict], max_tokens: int = 4096
    ) -> dict: ...


class SessionStore(Protocol):
    def create(self) -> str: ...
    def list(self) -> list[str]: ...
    def read(self, session: str) -> list[dict]: ...
    def append(self, session: str, event: dict) -> None: ...


def message(role, content):
    return {"role": role, "content": content}


def reply_text(msg):
    content = msg["content"]
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in content if b["type"] == "text")


def check_shape(value, shape):
    """Validate the primitive, object and array schemas used by saved workflows."""
    kinds = {
        "object": (dict,),
        "array": (list,),
        "string": (str,),
        "boolean": (bool,),
        "integer": (int,),
        "number": (int, float),
        "null": (type(None),),
    }
    if "type" in shape and type(value) not in kinds[shape["type"]]:
        raise ValueError("Expected " + shape["type"])
    if "enum" in shape and value not in shape["enum"]:
        raise ValueError("Value is outside enum")
    if isinstance(value, dict):
        if set(shape.get("required", ())) - value.keys():
            raise ValueError("Missing required fields")
        props = shape.get("properties", {})
        if shape.get("additionalProperties") is False and value.keys() - props.keys():
            raise ValueError("Unexpected object fields")
        for name in value.keys() & props.keys():
            check_shape(value[name], props[name])
    if isinstance(value, list) and "items" in shape:
        for item in value:
            check_shape(item, shape["items"])
