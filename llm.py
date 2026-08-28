from __future__ import annotations

import json
from typing import Protocol, Any


class LLMClient(Protocol):
    async def complete_json(self, system: str, user: str) -> dict[str, Any]: ...
    async def complete_text(self, system: str, user: str) -> str: ...


class ScriptedClient:
    """Deterministic demo client for tests; replace with a real provider adapter."""
    def __init__(self, responses: list[dict[str, Any] | str]):
        self.responses = list(responses)

    async def complete_json(self, system: str, user: str) -> dict[str, Any]:
        if not self.responses:
            raise RuntimeError("ScriptedClient exhausted")
        item = self.responses.pop(0)
        if isinstance(item, str):
            return json.loads(item)
        return item

    async def complete_text(self, system: str, user: str) -> str:
        if not self.responses:
            raise RuntimeError("ScriptedClient exhausted")
        item = self.responses.pop(0)
        return item if isinstance(item, str) else json.dumps(item)
