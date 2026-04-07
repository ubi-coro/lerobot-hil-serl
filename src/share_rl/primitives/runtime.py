from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from share_rl.primitives.config import PrimitiveGraphConfig


class Primitive(Protocol):
    def reset(self) -> dict:
        ...

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        ...


@dataclass(slots=True)
class PrimitiveGraphEnv:
    """Runtime orchestrator for multiple primitives sharing the same hardware stack."""

    config: PrimitiveGraphConfig
    primitives: dict[str, Primitive]
    _active_primitive_id: str = field(init=False)

    def __post_init__(self) -> None:
        configured_ids = {node.primitive_id for node in self.config.nodes}
        missing = sorted(configured_ids - set(self.primitives.keys()))
        if missing:
            raise ValueError(f"missing primitive runtime implementations: {missing}")
        self._active_primitive_id = self.config.start_primitive_id

    @property
    def active_primitive_id(self) -> str:
        return self._active_primitive_id

    def reset(self) -> dict:
        self._active_primitive_id = self.config.start_primitive_id
        observation = self.primitives[self._active_primitive_id].reset()
        return self._attach_primitive_context(observation)

    def step(self, action: dict) -> tuple[dict, float, bool, dict]:
        primitive = self.primitives[self._active_primitive_id]
        observation, reward, done, info = primitive.step(action)

        transition_key = info.get("transition")
        if transition_key:
            node = self.config.node_by_id(self._active_primitive_id)
            if transition_key in node.transitions:
                self._active_primitive_id = node.transitions[transition_key]
                info = dict(info)
                info["active_primitive_id"] = self._active_primitive_id

        return self._attach_primitive_context(observation), reward, done, info

    def _attach_primitive_context(self, observation: dict) -> dict:
        enriched = dict(observation)
        enriched["primitive_id"] = self._active_primitive_id
        return enriched
