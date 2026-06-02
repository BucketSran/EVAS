"""Indexed simulation data helpers.

This module is a migration boundary for a future native/Rust backend.  It does
not replace the current dict-based simulator path; it only provides stable node
and state numbering utilities that can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableSequence, Sequence


@dataclass
class NodeIndex:
    """Stable node-name to integer-id mapping."""

    _ids: Dict[str, int] = field(default_factory=dict)
    _names: List[str] = field(default_factory=list)

    def intern(self, name: str) -> int:
        if not isinstance(name, str) or not name:
            raise ValueError("node name must be a non-empty string")
        existing = self._ids.get(name)
        if existing is not None:
            return existing
        node_id = len(self._names)
        self._ids[name] = node_id
        self._names.append(name)
        return node_id

    def intern_many(self, names: Iterable[str]) -> List[int]:
        return [self.intern(name) for name in names]

    def id_of(self, name: str) -> int:
        try:
            return self._ids[name]
        except KeyError as exc:
            raise KeyError(f"unknown node: {name}") from exc

    def name_of(self, node_id: int) -> str:
        try:
            return self._names[node_id]
        except IndexError as exc:
            raise IndexError(f"unknown node id: {node_id}") from exc

    @property
    def names(self) -> Sequence[str]:
        return tuple(self._names)

    def __len__(self) -> int:
        return len(self._names)


@dataclass
class StateIndex:
    """Stable state-name to integer-id mapping for model-local state."""

    _ids: Dict[str, int] = field(default_factory=dict)
    _names: List[str] = field(default_factory=list)

    def intern(self, name: str) -> int:
        if not isinstance(name, str) or not name:
            raise ValueError("state name must be a non-empty string")
        existing = self._ids.get(name)
        if existing is not None:
            return existing
        state_id = len(self._names)
        self._ids[name] = state_id
        self._names.append(name)
        return state_id

    def id_of(self, name: str) -> int:
        try:
            return self._ids[name]
        except KeyError as exc:
            raise KeyError(f"unknown state: {name}") from exc

    @property
    def names(self) -> Sequence[str]:
        return tuple(self._names)

    def __len__(self) -> int:
        return len(self._names)


@dataclass
class IndexedVoltages:
    """Array-backed voltage storage with dict conversion helpers."""

    node_index: NodeIndex
    values: List[float]

    @classmethod
    def zeros(cls, node_index: NodeIndex) -> "IndexedVoltages":
        return cls(node_index=node_index, values=[0.0] * len(node_index))

    @classmethod
    def from_mapping(cls, node_index: NodeIndex, voltages: Mapping[str, float]) -> "IndexedVoltages":
        values = [0.0] * len(node_index)
        for name, value in voltages.items():
            values[node_index.id_of(name)] = float(value)
        return cls(node_index=node_index, values=values)

    def get(self, name: str) -> float:
        return self.values[self.node_index.id_of(name)]

    def set(self, name: str, value: float) -> None:
        self.values[self.node_index.id_of(name)] = float(value)

    def snapshot(self) -> List[float]:
        return list(self.values)

    def restore(self, snapshot: Sequence[float]) -> None:
        if len(snapshot) != len(self.values):
            raise ValueError(
                f"snapshot length {len(snapshot)} does not match node count {len(self.values)}"
            )
        self.values[:] = [float(value) for value in snapshot]

    def to_mapping(self) -> Dict[str, float]:
        return {
            name: self.values[node_id]
            for node_id, name in enumerate(self.node_index.names)
        }


def build_node_index(*name_groups: Iterable[str]) -> NodeIndex:
    index = NodeIndex()
    for names in name_groups:
        index.intern_many(names)
    return index


def copy_values_into(target: MutableSequence[float], source: Sequence[float]) -> None:
    if len(target) != len(source):
        raise ValueError(f"target length {len(target)} does not match source length {len(source)}")
    for idx, value in enumerate(source):
        target[idx] = float(value)
