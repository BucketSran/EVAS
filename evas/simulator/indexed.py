"""Indexed simulation data helpers.

This module is a migration boundary for a future native/Rust backend.  It does
not replace the current dict-based simulator path; it only provides stable node
and state numbering utilities that can be tested independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
)


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

    def has(self, name: str) -> bool:
        return name in self._ids

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


@dataclass
class IndexedVoltageSnapshotter:
    """Opt-in profiler for dict voltage snapshots.

    It mirrors a mapping into stable node-id order and returns array snapshots.
    The current simulator still consumes dicts; this helper exists so migration
    work can measure and validate the indexed representation before relying on
    it in the hot loop.
    """

    node_index: NodeIndex
    values: List[float] = field(default_factory=list)
    dynamic_interns: int = 0

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "IndexedVoltageSnapshotter":
        node_index = build_node_index(names)
        return cls(node_index=node_index, values=[0.0] * len(node_index))

    @classmethod
    def from_mapping(cls, voltages: Mapping[str, float]) -> "IndexedVoltageSnapshotter":
        snapshotter = cls.from_names(voltages.keys())
        snapshotter.update_from_mapping(voltages)
        return snapshotter

    @property
    def node_count(self) -> int:
        return len(self.node_index)

    def _ensure_nodes(self, names: Iterable[str]) -> None:
        before = len(self.node_index)
        for name in names:
            if not self.node_index.has(name):
                self.node_index.intern(name)
        added = len(self.node_index) - before
        if added:
            self.values.extend([0.0] * added)
            self.dynamic_interns += added

    def update_from_mapping(self, voltages: Mapping[str, float]) -> None:
        self._ensure_nodes(voltages.keys())
        for name, value in voltages.items():
            self.values[self.node_index.id_of(name)] = float(value)

    def snapshot_from_mapping(self, voltages: Mapping[str, float]) -> List[float]:
        self.update_from_mapping(voltages)
        return list(self.values)

    def max_abs_diff(self, snapshot: Sequence[float], voltages: Mapping[str, float]) -> Tuple[float, str, int]:
        if len(snapshot) != len(self.node_index):
            raise ValueError(
                f"snapshot length {len(snapshot)} does not match node count {len(self.node_index)}"
            )
        max_diff = 0.0
        max_node = ""
        checked = 0
        for name, value in voltages.items():
            diff = abs(float(value) - float(snapshot[self.node_index.id_of(name)]))
            checked += 1
            if diff > max_diff:
                max_diff = diff
                max_node = name
        return max_diff, max_node, checked


@dataclass
class IndexedVoltageArray:
    """Persistent array-backed voltage mirror for opt-in simulator paths.

    Unlike ``IndexedVoltages``, this helper can grow when model output nodes
    appear at runtime.  The dict-backed simulator remains authoritative while
    this mirror lets low-risk hot-loop reads exercise the node-id layout that a
    native backend will consume.
    """

    node_index: NodeIndex
    values: List[float] = field(default_factory=list)
    dynamic_interns: int = 0

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "IndexedVoltageArray":
        node_index = build_node_index(names)
        return cls(node_index=node_index, values=[0.0] * len(node_index))

    @classmethod
    def from_mapping(cls, voltages: Mapping[str, float]) -> "IndexedVoltageArray":
        array = cls.from_names(voltages.keys())
        array.update_from_mapping(voltages)
        return array

    @property
    def node_count(self) -> int:
        return len(self.node_index)

    def _ensure_node(self, name: str) -> int:
        before = len(self.node_index)
        node_id = self.node_index.intern(name)
        if len(self.node_index) > before:
            self.values.append(0.0)
            self.dynamic_interns += 1
        return node_id

    def ensure_nodes(self, names: Iterable[str]) -> int:
        before = len(self.node_index)
        for name in names:
            if not self.node_index.has(name):
                self._ensure_node(name)
        return len(self.node_index) - before

    def update_from_mapping(self, voltages: Mapping[str, float]) -> int:
        added = self.ensure_nodes(voltages.keys())
        for name, value in voltages.items():
            self.values[self.node_index.id_of(name)] = float(value)
        return added

    def set(self, name: str, value: float) -> None:
        self.values[self._ensure_node(name)] = float(value)

    def get(self, name: str, default: float = 0.0) -> float:
        if not self.node_index.has(name):
            return float(default)
        return self.values[self.node_index.id_of(name)]

    def values_for_ids(self, node_ids: Sequence[int], default: float = 0.0) -> List[float]:
        values = self.values
        fallback = float(default)
        return [
            float(values[node_id]) if 0 <= int(node_id) < len(values) else fallback
            for node_id in node_ids
        ]

    def get_from_snapshot(
        self,
        snapshot: Optional[Sequence[float]],
        name: str,
        default: float = 0.0,
    ) -> float:
        if snapshot is None or not self.node_index.has(name):
            return float(default)
        node_id = self.node_index.id_of(name)
        if node_id >= len(snapshot):
            return float(default)
        return float(snapshot[node_id])

    def snapshot(self) -> List[float]:
        return list(self.values)

    def to_mapping(self) -> Dict[str, float]:
        return {
            name: self.values[node_id]
            for node_id, name in enumerate(self.node_index.names)
        }

    def max_abs_diff_mapping(self, voltages: Mapping[str, float]) -> Tuple[float, str, int]:
        self.ensure_nodes(voltages.keys())
        max_diff = 0.0
        max_node = ""
        checked = 0
        for name, value in voltages.items():
            diff = abs(float(value) - self.values[self.node_index.id_of(name)])
            checked += 1
            if diff > max_diff:
                max_diff = diff
                max_node = name
        return max_diff, max_node, checked

    def max_abs_diff_names(
        self,
        voltages: Mapping[str, float],
        names: Iterable[str],
    ) -> Tuple[float, str, int]:
        max_diff = 0.0
        max_node = ""
        checked = 0
        for name in names:
            if name not in voltages:
                continue
            node_id = self._ensure_node(name)
            diff = abs(float(voltages[name]) - self.values[node_id])
            checked += 1
            if diff > max_diff:
                max_diff = diff
                max_node = name
        return max_diff, max_node, checked


@dataclass
class IndexedRunPlan:
    """Opt-in lowering plan for a dict-backed Simulator instance.

    The plan is intentionally sidecar-only: it records the node ids that a
    future indexed/Rust backend would consume, but it does not mutate the
    current Simulator.
    """

    node_index: NodeIndex
    source_node_ids: Tuple[int, ...]
    recorded_node_ids: Tuple[int, ...]
    model_node_ids: Tuple[int, ...]

    @property
    def node_count(self) -> int:
        return len(self.node_index)


@dataclass(frozen=True)
class DynamicBranchAccessIO:
    """Dynamic node-array access boundary for future bus lowering."""

    role: str
    base_node: str
    dimensions: int
    context: str


@dataclass(frozen=True)
class IndexedStateArrayLayout:
    """Model-local array state layout for future indexed/native state storage."""

    name: str
    lo: int
    hi: int
    length: int
    integer: bool


@dataclass(frozen=True)
class IndexedModelIO:
    """Node-id boundary for one compiled model instance."""

    model_path: Tuple[int, ...]
    model_class: str
    mapped_port_node_ids: Tuple[int, ...]
    output_node_ids: Tuple[int, ...]
    static_voltage_read_node_ids: Tuple[int, ...] = ()
    event_trigger_voltage_node_ids: Tuple[int, ...] = ()
    event_voltage_read_node_ids: Tuple[int, ...] = ()
    event_body_voltage_read_node_ids: Tuple[int, ...] = ()
    static_output_write_node_ids: Tuple[int, ...] = ()
    dynamic_branch_accesses: Tuple[DynamicBranchAccessIO, ...] = ()
    dynamic_voltage_read_count: int = 0
    dynamic_output_write_count: int = 0
    state_scalar_names: Tuple[str, ...] = ()
    state_scalar_ids: Tuple[int, ...] = ()
    integer_state_names: Tuple[str, ...] = ()
    state_array_layouts: Tuple[IndexedStateArrayLayout, ...] = ()


@dataclass
class IndexedModelIOPlan:
    """Per-model node-id IO boundary for future indexed/native evaluation."""

    node_index: NodeIndex
    model_ios: Tuple[IndexedModelIO, ...]

    @property
    def node_count(self) -> int:
        return len(self.node_index)

    @property
    def model_count(self) -> int:
        return len(self.model_ios)

    @property
    def mapped_port_count(self) -> int:
        return sum(len(model_io.mapped_port_node_ids) for model_io in self.model_ios)

    @property
    def output_count(self) -> int:
        return sum(len(model_io.output_node_ids) for model_io in self.model_ios)

    @property
    def static_voltage_read_count(self) -> int:
        return sum(
            len(model_io.static_voltage_read_node_ids)
            for model_io in self.model_ios
        )

    @property
    def event_voltage_read_count(self) -> int:
        return sum(
            len(model_io.event_voltage_read_node_ids)
            for model_io in self.model_ios
        )

    @property
    def event_trigger_voltage_count(self) -> int:
        return sum(
            len(model_io.event_trigger_voltage_node_ids)
            for model_io in self.model_ios
        )

    @property
    def event_body_voltage_read_count(self) -> int:
        return sum(
            len(model_io.event_body_voltage_read_node_ids)
            for model_io in self.model_ios
        )

    @property
    def static_output_write_count(self) -> int:
        return sum(
            len(model_io.static_output_write_node_ids)
            for model_io in self.model_ios
        )

    @property
    def dynamic_voltage_read_count(self) -> int:
        return sum(model_io.dynamic_voltage_read_count for model_io in self.model_ios)

    @property
    def dynamic_output_write_count(self) -> int:
        return sum(model_io.dynamic_output_write_count for model_io in self.model_ios)

    @property
    def dynamic_branch_access_count(self) -> int:
        return sum(len(model_io.dynamic_branch_accesses) for model_io in self.model_ios)

    @property
    def scalar_state_count(self) -> int:
        return sum(len(model_io.state_scalar_names) for model_io in self.model_ios)

    @property
    def integer_state_count(self) -> int:
        return sum(len(model_io.integer_state_names) for model_io in self.model_ios)

    @property
    def state_array_count(self) -> int:
        return sum(len(model_io.state_array_layouts) for model_io in self.model_ios)

    @property
    def state_array_slot_count(self) -> int:
        return sum(
            array_layout.length
            for model_io in self.model_ios
            for array_layout in model_io.state_array_layouts
        )


@dataclass
class IndexedTrace:
    """Array-style waveform trace used by the indexed parity harness."""

    node_index: NodeIndex
    time: List[float]
    signal_node_ids: List[int]
    values: List[List[float]]

    @classmethod
    def from_result(
        cls,
        result: Any,
        node_index: Optional[NodeIndex] = None,
        signal_names: Optional[Iterable[str]] = None,
        extra_nodes: Iterable[str] = (),
    ) -> "IndexedTrace":
        signals = getattr(result, "signals", {})
        if node_index is None:
            node_index = build_node_index(extra_nodes, signals.keys())
        else:
            node_index.intern_many(extra_nodes)
            node_index.intern_many(signals.keys())

        requested = list(signal_names) if signal_names is not None else list(signals.keys())
        valid_signals = [name for name in requested if name in signals]
        return cls(
            node_index=node_index,
            time=[float(value) for value in getattr(result, "time", [])],
            signal_node_ids=[node_index.id_of(name) for name in valid_signals],
            values=[
                [float(value) for value in signals[name]]
                for name in valid_signals
            ],
        )

    @property
    def signal_names(self) -> Tuple[str, ...]:
        return tuple(self.node_index.name_of(node_id) for node_id in self.signal_node_ids)

    def to_signal_mapping(self) -> Dict[str, List[float]]:
        return {
            self.node_index.name_of(node_id): list(values)
            for node_id, values in zip(self.signal_node_ids, self.values)
        }


@dataclass(frozen=True)
class IndexedParityReport:
    """Result of dict waveform → indexed trace → dict waveform comparison."""

    checked_signals: int
    checked_samples: int
    missing_signals: Tuple[str, ...]
    max_abs_diff: float
    max_abs_diff_signal: str
    length_mismatches: Tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.length_mismatches and self.max_abs_diff == 0.0

    def summary(self) -> str:
        status = "passed" if self.passed else "failed"
        missing = len(self.missing_signals)
        return (
            f"{status}: checked_signals={self.checked_signals}, "
            f"checked_samples={self.checked_samples}, "
            f"max_abs_diff={self.max_abs_diff:g}, "
            f"max_abs_diff_signal={self.max_abs_diff_signal or 'n/a'}, "
            f"missing_requested_signals={missing}"
        )


def _iter_model_tree(model: Any):
    yield model
    for child in getattr(model, "_child_models", []) or []:
        yield from _iter_model_tree(child)


def _iter_model_tree_with_path(model: Any, path: Tuple[int, ...] = ()):
    yield path, model
    for idx, child in enumerate(getattr(model, "_child_models", []) or []):
        yield from _iter_model_tree_with_path(child, path + (idx,))


def _id_tuple(index: NodeIndex, names: Iterable[str]) -> Tuple[int, ...]:
    return tuple(index.id_of(name) for name in names)


def _resolve_model_node(model: Any, node: str) -> str:
    """Resolve a model-local node through node_map and one or more parents."""
    ext = getattr(model, "node_map", {}).get(node, node)
    current = model
    visited = 0
    while (
        isinstance(ext, str)
        and ext
        and ext[0] == "@"
        and ext.startswith("@parent:")
        and getattr(current, "_parent_model", None) is not None
        and visited < 16
    ):
        parent = current._parent_model
        pnode = ext[len("@parent:"):]
        ext = getattr(parent, "node_map", {}).get(pnode, pnode)
        current = parent
        visited += 1
    return ext if isinstance(ext, str) and ext else node


def build_indexed_model_io_plan(
    simulator: Any,
    extra_nodes: Iterable[str] = (),
) -> IndexedModelIOPlan:
    """Build a sidecar model IO plan without changing dict-backed execution."""

    model_entries = []
    io_names: List[str] = []
    for root_index, model in enumerate(getattr(simulator, "models", []) or []):
        for path, tree_model in _iter_model_tree_with_path(model, (root_index,)):
            model_cls = getattr(tree_model, "__class__", type(tree_model))
            mapped_ports = [
                _resolve_model_node(tree_model, local_name)
                for local_name in getattr(tree_model, "node_map", {}).keys()
            ]
            output_nodes = [
                _resolve_model_node(tree_model, local_name)
                for local_name in getattr(tree_model, "output_nodes", {}).keys()
            ]
            static_reads = [
                _resolve_model_node(tree_model, local_name)
                for local_name in getattr(model_cls, "_static_voltage_read_nodes", ()) or ()
            ]
            event_trigger_reads = [
                _resolve_model_node(tree_model, local_name)
                for local_name in getattr(model_cls, "_event_trigger_voltage_read_nodes", ()) or ()
            ]
            event_reads = [
                _resolve_model_node(tree_model, local_name)
                for local_name in getattr(model_cls, "_event_voltage_read_nodes", ()) or ()
            ]
            event_body_reads = [
                _resolve_model_node(tree_model, local_name)
                for local_name in (
                    getattr(model_cls, "_event_body_voltage_read_nodes", None)
                    or getattr(model_cls, "_event_voltage_read_nodes", ())
                    or ()
                )
            ]
            static_writes = [
                _resolve_model_node(tree_model, local_name)
                for local_name in getattr(model_cls, "_static_output_write_nodes", ()) or ()
            ]
            dynamic_read_count = int(
                getattr(model_cls, "_dynamic_voltage_read_count", 0) or 0
            )
            dynamic_write_count = int(
                getattr(model_cls, "_dynamic_output_write_count", 0) or 0
            )
            dynamic_branch_accesses = tuple(
                DynamicBranchAccessIO(
                    role=str(role),
                    base_node=str(base_node),
                    dimensions=int(dimensions),
                    context=str(context),
                )
                for role, base_node, dimensions, context in (
                    getattr(model_cls, "_dynamic_branch_accesses", ()) or ()
                )
            )
            state_index = StateIndex()
            state_scalar_names = tuple(
                str(name)
                for name in getattr(model_cls, "_state_scalar_names", ()) or ()
            )
            state_scalar_ids = tuple(state_index.intern(name) for name in state_scalar_names)
            integer_state_names = tuple(
                str(name)
                for name in getattr(model_cls, "_integer_state_names", ()) or ()
            )
            state_array_layouts = tuple(
                IndexedStateArrayLayout(
                    name=str(name),
                    lo=int(lo),
                    hi=int(hi),
                    length=max(0, int(hi) - int(lo) + 1),
                    integer=bool(integer),
                )
                for name, lo, hi, integer in (
                    getattr(model_cls, "_state_array_ranges", ()) or ()
                )
            )
            model_entries.append(
                (
                    path,
                    tree_model,
                    mapped_ports,
                    output_nodes,
                    static_reads,
                    event_trigger_reads,
                    event_reads,
                    event_body_reads,
                    static_writes,
                    dynamic_branch_accesses,
                    dynamic_read_count,
                    dynamic_write_count,
                    state_scalar_names,
                    state_scalar_ids,
                    integer_state_names,
                    state_array_layouts,
                )
            )
            io_names.extend(mapped_ports)
            io_names.extend(output_nodes)
            io_names.extend(static_reads)
            io_names.extend(event_trigger_reads)
            io_names.extend(event_reads)
            io_names.extend(event_body_reads)
            io_names.extend(static_writes)

    index = build_node_index(extra_nodes, io_names)
    model_ios = tuple(
        IndexedModelIO(
            model_path=path,
            model_class=getattr(getattr(model, "__class__", type(model)), "__name__", "model"),
            mapped_port_node_ids=_id_tuple(index, mapped_ports),
            output_node_ids=_id_tuple(index, output_nodes),
            static_voltage_read_node_ids=_id_tuple(index, static_reads),
            event_trigger_voltage_node_ids=_id_tuple(index, event_trigger_reads),
            event_voltage_read_node_ids=_id_tuple(index, event_reads),
            event_body_voltage_read_node_ids=_id_tuple(index, event_body_reads),
            static_output_write_node_ids=_id_tuple(index, static_writes),
            dynamic_branch_accesses=dynamic_branch_accesses,
            dynamic_voltage_read_count=dynamic_read_count,
            dynamic_output_write_count=dynamic_write_count,
            state_scalar_names=state_scalar_names,
            state_scalar_ids=state_scalar_ids,
            integer_state_names=integer_state_names,
            state_array_layouts=state_array_layouts,
        )
        for (
            path,
            model,
            mapped_ports,
            output_nodes,
            static_reads,
            event_trigger_reads,
            event_reads,
            event_body_reads,
            static_writes,
            dynamic_branch_accesses,
            dynamic_read_count,
            dynamic_write_count,
            state_scalar_names,
            state_scalar_ids,
            integer_state_names,
            state_array_layouts,
        ) in model_entries
    )
    return IndexedModelIOPlan(node_index=index, model_ios=model_ios)


def build_indexed_run_plan(simulator: Any, extra_nodes: Iterable[str] = ()) -> IndexedRunPlan:
    """Build a sidecar node-id plan for the current dict-backed simulator.

    This is the first migration checkpoint before a native backend: every
    source, recorded node, model port mapping, and caller-supplied netlist node
    gets a stable integer id.  The function only reads simulator state.
    """

    source_names: List[str] = []
    for src in getattr(simulator, "sources", []) or []:
        node = getattr(src, "node", None)
        if node:
            source_names.append(node)

    recorded_names = list(getattr(simulator, "recorded_signals", {}).keys())
    voltage_names = list(getattr(simulator, "node_voltages", {}).keys())
    model_names: List[str] = []
    for model in getattr(simulator, "models", []) or []:
        for tree_model in _iter_model_tree(model):
            model_names.extend(getattr(tree_model, "node_map", {}).values())
            model_names.extend(getattr(tree_model, "output_nodes", {}).keys())

    index = build_node_index(extra_nodes, source_names, recorded_names, voltage_names, model_names)
    return IndexedRunPlan(
        node_index=index,
        source_node_ids=_id_tuple(index, source_names),
        recorded_node_ids=_id_tuple(index, recorded_names),
        model_node_ids=_id_tuple(index, model_names),
    )


def check_indexed_trace_round_trip(
    result: Any,
    node_index: Optional[NodeIndex] = None,
    signal_names: Optional[Iterable[str]] = None,
    extra_nodes: Iterable[str] = (),
) -> IndexedParityReport:
    """Check that result waveform lowering to indexed trace is lossless."""

    signals = getattr(result, "signals", {})
    requested = list(signal_names) if signal_names is not None else list(signals.keys())
    missing = tuple(name for name in requested if name not in signals)
    valid_signals = [name for name in requested if name in signals]
    trace = IndexedTrace.from_result(
        result,
        node_index=node_index,
        signal_names=valid_signals,
        extra_nodes=extra_nodes,
    )
    round_trip = trace.to_signal_mapping()

    max_abs_diff = 0.0
    max_abs_diff_signal = ""
    checked_samples = 0
    length_mismatches: List[str] = []
    for name in valid_signals:
        original = [float(value) for value in signals[name]]
        lowered = round_trip[name]
        if len(original) != len(lowered):
            length_mismatches.append(name)
            continue
        checked_samples += len(original)
        for before, after in zip(original, lowered):
            diff = abs(before - after)
            if diff > max_abs_diff:
                max_abs_diff = diff
                max_abs_diff_signal = name

    return IndexedParityReport(
        checked_signals=len(valid_signals),
        checked_samples=checked_samples,
        missing_signals=missing,
        max_abs_diff=max_abs_diff,
        max_abs_diff_signal=max_abs_diff_signal,
        length_mismatches=tuple(length_mismatches),
    )
