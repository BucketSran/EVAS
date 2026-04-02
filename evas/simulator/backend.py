"""
compiler_backend.py — Compile Verilog-A AST into executable Python model classes.

Takes a parsed Module AST and generates a Python class that implements
the behavioral model with proper event handling, transition operators,
and state variable management.
"""
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from evas.compiler.ast_nodes import *
from evas.simulator.engine import (
    AboveDetector,
    CrossDetector,
    TransitionState,
)


class CompilationError(Exception):
    pass


class CompiledModel:
    """Base class for compiled Verilog-A models."""
    _module_registry: Dict[str, Any] = {}
    _module_ports: List[str] = []

    def __init__(self):
        self.params: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.arrays: Dict[str, Dict[int, Any]] = {}
        self.transitions: Dict[str, TransitionState] = {}
        self.cross_detectors: Dict[str, CrossDetector] = {}
        self.above_detectors: Dict[str, AboveDetector] = {}
        self.output_nodes: Dict[str, float] = {}
        self.node_map: Dict[str, str] = {}  # port_name -> external_node
        self.default_transition: float = 1e-12
        self._initial_step_done: bool = False
        self._strobe_log: List[str] = []
        self._event_time: float = 0.0  # $abstime inside cross/above event bodies
        self._temperature: float = 27.0  # degrees Celsius (expressions convert to Kelvin)
        self.timer_states: Dict[str, float] = {}  # key → next_fire_time
        self._bound_step: float = 0.0  # $bound_step limit (0 = no limit)
        # Lazy-allocated integrator states (only used when idt/idtmod appears)
        self._idt_states: Optional[Dict[str, Dict[str, float]]] = None
        self._file_handles: Dict[int, Any] = {}  # fd → file object
        self._next_fd: int = 1
        self._child_models: List["CompiledModel"] = []
        self._parent_model: Optional["CompiledModel"] = None

    def initial_step(self, node_voltages: Dict[str, float], time: float):
        pass

    def evaluate(self, node_voltages: Dict[str, float], time: float):
        pass

    def final_step(self, node_voltages: Dict[str, float], time: float):
        pass

    def next_breakpoint(self, time: float) -> Optional[float]:
        bps = []
        for ts in self.transitions.values():
            bp = ts.next_breakpoint(time)
            if bp is not None:
                bps.append(bp)
        for cd in self.cross_detectors.values():
            bp = cd.next_breakpoint()
            if bp is not None and bp > time:
                bps.append(bp)
        for ad in self.above_detectors.values():
            bp = ad.next_breakpoint()
            if bp is not None and bp > time:
                bps.append(bp)
        # Timer breakpoints
        for nf in self.timer_states.values():
            if nf > time:
                bps.append(nf)
        for child in self._child_models:
            bp = child.next_breakpoint(time)
            if bp is not None:
                bps.append(bp)
        return min(bps) if bps else None

    def _check_timer(self, key: str, time: float, period: float) -> bool:
        if key not in self.timer_states:
            self.timer_states[key] = period  # first fire at t=period
        next_fire = self.timer_states[key]
        if time >= next_fire - 1e-18:
            self.timer_states[key] = next_fire + period
            return True
        return False

    def _get_voltage(self, node: str, node_voltages: Dict[str, float]) -> float:
        """Get voltage of a node, resolving through node_map."""
        # Check if it's a mapped external node
        ext = self.node_map.get(node, node)
        if isinstance(ext, str) and ext.startswith('@parent:') and self._parent_model is not None:
            pnode = ext[len('@parent:'):]
            ext = self._parent_model.node_map.get(pnode, pnode)
        if ext in node_voltages:
            return node_voltages[ext]
        # Check output nodes (self-driven)
        if node in self.output_nodes:
            return self.output_nodes[node]
        return 0.0

    def _set_output(self, node: str, value: float, node_voltages: Dict[str, float]):
        """Set an output node voltage."""
        self.output_nodes[node] = value
        ext = self.node_map.get(node, node)
        if isinstance(ext, str) and ext.startswith('@parent:') and self._parent_model is not None:
            pnode = ext[len('@parent:'):]
            ext = self._parent_model.node_map.get(pnode, pnode)
        node_voltages[ext] = value

    def _transition(self, key: str, time: float, target: float,
                    delay: float = 0.0, rise: float = 0.0, fall: float = 0.0) -> float:
        """Evaluate a transition operator."""
        if key not in self.transitions:
            self.transitions[key] = TransitionState(current_val=target)
            return target
        ts = self.transitions[key]
        # Advance current_val to the actual value at this time before updating target.
        # Without this, a new target set at t overwrites the in-progress transition
        # before evaluate() can commit its endpoint into current_val.
        ts.evaluate(time)
        ts.set_target(time, target, delay, rise, fall, self.default_transition)
        return ts.evaluate(time)

    def _check_cross(self, key: str, time: float, val: float, direction: int = 0) -> bool:
        if key not in self.cross_detectors:
            self.cross_detectors[key] = CrossDetector(direction=direction)
        fired = self.cross_detectors[key].check(time, val)
        if fired:
            self._event_time = self.cross_detectors[key].t_cross
        return fired

    def _check_above(self, key: str, time: float, val: float, direction: int = 1) -> bool:
        if key not in self.above_detectors:
            self.above_detectors[key] = AboveDetector(direction=direction)
        fired = self.above_detectors[key].check(time, val)
        if fired:
            self._event_time = self.above_detectors[key].t_cross
        return fired

    def _idtmod(self, key: str, time: float, x: float,
                ic: float = 0.0, mod: float = 1.0) -> float:
        """
        Minimal idtmod integrator with trapezoidal update.

        idtmod(x, ic, mod) ≈ ic + ∫x dt wrapped into [0, mod).
        Notes:
        - Accuracy depends on external timestep control ($bound_step / tran step).
        - Multiple evaluations at the same time do not re-integrate.
        """
        if self._idt_states is None:
            self._idt_states = {}

        if key not in self._idt_states:
            self._idt_states[key] = {
                "y": float(ic),
                "last_t": float(time),
                "last_x": float(x),
                "last_eval_t": float(time),
            }
            y0 = float(ic)
            if mod is not None and float(mod) != 0.0:
                m = abs(float(mod))
                y0 = y0 % m
                self._idt_states[key]["y"] = y0
            return y0

        st = self._idt_states[key]
        if time == st["last_eval_t"]:
            return st["y"]

        dt = float(time) - float(st["last_t"])
        if dt > 0.0:
            st["y"] += 0.5 * (float(x) + st["last_x"]) * dt
            if mod is not None and float(mod) != 0.0:
                m = abs(float(mod))
                st["y"] = st["y"] % m
            st["last_t"] = float(time)
            st["last_x"] = float(x)
        elif dt < 0.0:
            # Time rollback (e.g., restart): re-seed from current value.
            st["last_t"] = float(time)
            st["last_x"] = float(x)

        st["last_eval_t"] = float(time)
        return st["y"]

    def _array_get(self, name: str, idx: int) -> Any:
        if name in self.arrays and idx in self.arrays[name]:
            return self.arrays[name][idx]
        return 0

    def _array_set(self, name: str, idx: int, val: Any):
        if name not in self.arrays:
            self.arrays[name] = {}
        self.arrays[name][idx] = val

    def _strobe(self, time: float, fmt: str, *args):
        try:
            msg = (fmt % args) if args else fmt
        except Exception as e:
            msg = f"{fmt}  [format error: {e}]"
        self._strobe_log.append((time, msg))

    def _fopen(self, filename: str, mode: str = 'w') -> int:
        fd = self._next_fd
        self._next_fd += 1
        self._file_handles[fd] = open(filename, mode)
        return fd

    def _fclose(self, fd: int):
        if fd in self._file_handles:
            self._file_handles[fd].close()
            del self._file_handles[fd]

    def _fstrobe(self, fd: int, fmt: str, *args):
        if fd in self._file_handles:
            try:
                msg = (fmt % args) if args else fmt
            except Exception as e:
                msg = f"{fmt}  [format error: {e}]"
            self._file_handles[fd].write(msg + '\n')

    def _cleanup_files(self):
        for f in self._file_handles.values():
            f.close()
        self._file_handles.clear()
        for child in self._child_models:
            child._cleanup_files()


def compile_module(module: Module, default_transition: float = None) -> type:
    """
    Compile a Module AST into a Python class.

    Returns a class (subclass of CompiledModel) that can be instantiated
    and connected to a Simulator.
    """
    compiler = _ModuleCompiler(module, default_transition)
    return compiler.compile()


class _ModuleCompiler:
    def __init__(self, module: Module, default_transition: float = None):
        self.module = module
        self.default_transition = default_transition or 1e-12
        self._trans_counter = 0
        self._cross_counter = 0
        self._above_counter = 0
        self._timer_counter = 0
        self._idt_counter = 0
        self._uses_idtmod = False
        self._indent = 2
        self._in_loop_var = None  # track if we're inside a for loop

    def compile(self) -> type:
        """Generate and return a compiled model class."""
        # Build the class dynamically
        mod = self.module

        self._validate_spectre_operator_rules()

        # Collect info (port lists reserved for future use)

        # Build arrays info
        array_vars = {}
        for v in mod.variables:
            if v.is_array:
                hi = v.array_hi if v.array_hi is not None else 0
                lo = v.array_lo if v.array_lo is not None else 0
                array_vars[v.name] = (hi, lo, v.init_values)

        # Build array port info
        array_ports = {}
        for p in mod.port_decls:
            if p.is_array:
                array_ports[p.name] = (p.array_hi, p.array_lo)

        # Generate code for the class
        lines = []
        lines.append(f"class {mod.name}_Model(CompiledModel):")
        lines.append(f"    _module_ports = {mod.ports!r}")
        lines.append("    def __init__(self):")
        lines.append("        super().__init__()")
        lines.append(f"        self.default_transition = {self.default_transition}")

        # Initialize parameters
        for p in mod.parameters:
            val = self._eval_expr_static(p.default_value)
            lines.append(f"        self.params[{p.name!r}] = {val!r}")

        # Initialize scalar state variables
        for v in mod.variables:
            if not v.is_array:
                init_val = 0
                if v.init_values and len(v.init_values) == 1:
                    init_val = self._eval_expr_static(v.init_values[0])
                lines.append(f"        self.state[{v.name!r}] = {init_val}")

        # Initialize array variables
        for name, (hi, lo, init_vals) in array_vars.items():
            lines.append(f"        self.arrays[{name!r}] = {{}}")
            lo_idx = min(hi, lo)
            hi_idx = max(hi, lo)
            if init_vals:
                for i, iv in enumerate(init_vals):
                    idx = hi_idx - i
                    val = self._eval_expr_static(iv)
                    lines.append(f"        self.arrays[{name!r}][{idx}] = {val!r}")
            else:
                for idx in range(lo_idx, hi_idx + 1):
                    lines.append(f"        self.arrays[{name!r}][{idx}] = 0")

        # Initialize hierarchical child instances.
        for inst in mod.instances:
            child_var = f"_child_{inst.instance_name}"
            lines.append(f"        _entry = self._module_registry.get({inst.module_name!r})")
            lines.append(f"        if _entry is None:")
            lines.append(f"            raise CompilationError('Unknown child module: {inst.module_name} in {mod.name}.{inst.instance_name}')")
            lines.append(f"        _child_cls, _child_mod = _entry")
            lines.append(f"        {child_var} = _child_cls()")
            lines.append(f"        {child_var}._parent_model = self")
            lines.append(f"        {child_var}.node_map = {{}}")
            # Positional and named port connections.
            for ci, c in enumerate(inst.connections):
                if c.port_name is not None:
                    port_expr = repr(c.port_name)
                else:
                    port_expr = f"_child_mod.ports[{ci}] if {ci} < len(_child_mod.ports) else None"
                target = self._compile_instance_target(c.expr)
                lines.append(f"        _pname = {port_expr!s}")
                lines.append(f"        if _pname is not None:")
                lines.append(f"            _target = {target}")
                lines.append(f"            if _target in self._module_ports:")
                lines.append(f"                _mapped = f'@parent:{{_target}}'")
                lines.append(f"            else:")
                lines.append(f"                _mapped = f'__{inst.instance_name}.{{_target}}'")
                lines.append(f"            {child_var}.node_map[_pname] = _mapped")
            lines.append(f"        self._child_models.append({child_var})")

        # Generate initial_step method
        lines.append("")
        lines.append("    def initial_step(self, nv, time):")
        lines.append("        if self._initial_step_done:")
        lines.append("            return")
        lines.append("        self._initial_step_done = True")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.initial_step(nv, time)")

        # Find and compile initial_step event blocks
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                if isinstance(stmt, EventStatement):
                    if self._is_initial_step_event(stmt.event):
                        body_lines = self._compile_statement(stmt.body, 2)
                        lines.extend(body_lines)

        lines.append("        pass")  # ensure method has body

        # Generate evaluate method
        self._trans_counter = 0
        self._cross_counter = 0
        self._above_counter = 0
        self._timer_counter = 0
        self._idt_counter = 0
        self._uses_idtmod = False

        lines.append("")
        lines.append("    def evaluate(self, nv, time):")
        lines.append("        self._event_time = time")
        lines.append("        self._bound_step = 0.0")
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.evaluate(nv, time)")

        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                stmt_lines = self._compile_statement(stmt, 2)
                lines.extend(stmt_lines)

        lines.append("        for _ch in self._child_models:")
        lines.append("            _bs = _ch._bound_step")
        lines.append("            if _bs > 0.0 and (self._bound_step <= 0.0 or _bs < self._bound_step):")
        lines.append("                self._bound_step = _bs")

        lines.append("        pass")

        # Generate final_step method
        lines.append("")
        lines.append("    def final_step(self, nv, time):")
        if mod.analog_block:
            for stmt in mod.analog_block.body.statements:
                if isinstance(stmt, EventStatement):
                    if self._is_final_step_event(stmt.event):
                        body_lines = self._compile_statement(stmt.body, 2)
                        lines.extend(body_lines)
        lines.append("        for _ch in self._child_models:")
        lines.append("            _ch.final_step(nv, time)")
        lines.append("        pass")

        # Compile the class
        code = '\n'.join(lines)

        # Create namespace with required imports
        namespace = {
            'CompiledModel': CompiledModel,
            'CompilationError': CompilationError,
            'math': math,
            'random': random,
            'pow': pow,
            'abs': abs,
            'int': int,
            'float': float,
        }

        try:
            exec(code, namespace)
        except Exception as e:
            raise CompilationError(
                f"Failed to compile module {mod.name}: {e}\n\nGenerated code:\n{code}"
            )

        cls = namespace[f'{mod.name}_Model']
        cls._uses_idtmod = self._uses_idtmod
        cls._generated_code = code  # Store for debugging
        return cls

    def _validate_spectre_operator_rules(self) -> None:
        """Reject patterns that Spectre VACOMP does not allow."""
        if not self.module.analog_block:
            return
        self._event_assigned_vars = set()
        self._non_event_assigned_vars = set()
        self._collect_assignment_contexts(self.module.analog_block.body, in_event=False)
        self._continuous_vars = self._infer_continuous_vars(self.module.analog_block.body)
        self._check_stmt_for_restricted_operators(
            self.module.analog_block.body,
            conditional_depth=0,
        )
        self._check_transition_targets(self.module.analog_block.body)

    def _check_stmt_for_restricted_operators(self, stmt, conditional_depth: int) -> None:
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._check_stmt_for_restricted_operators(child, conditional_depth)
            return

        if isinstance(stmt, IfStatement):
            self._check_stmt_for_restricted_operators(stmt.then_body, conditional_depth + 1)
            if stmt.else_body is not None:
                self._check_stmt_for_restricted_operators(stmt.else_body, conditional_depth + 1)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._check_stmt_for_restricted_operators(item.body, conditional_depth + 1)
            return

        if isinstance(stmt, EventStatement):
            self._check_stmt_for_restricted_operators(stmt.body, conditional_depth)
            return

        if isinstance(stmt, ForStatement):
            self._check_stmt_for_restricted_operators(stmt.body, conditional_depth)
            return

        if conditional_depth <= 0:
            return

        restricted = self._collect_restricted_calls_from_stmt(stmt)
        if restricted:
            ops = ', '.join(sorted(restricted))
            raise CompilationError(
                f"Module {self.module.name} uses Spectre-restricted operator(s) "
                f"{ops} inside a conditionally executed statement. "
                f"Move these operators out of if/case branches."
            )

    def _collect_restricted_calls_from_stmt(self, stmt) -> set:
        restricted = set()
        if isinstance(stmt, Assignment):
            restricted |= self._collect_restricted_calls_from_expr(stmt.value)
        elif isinstance(stmt, Contribution):
            restricted |= self._collect_restricted_calls_from_expr(stmt.expr)
        elif isinstance(stmt, SystemTask):
            for arg in stmt.args:
                restricted |= self._collect_restricted_calls_from_expr(arg)
        return restricted

    def _collect_restricted_calls_from_expr(self, expr: Expr) -> set:
        restricted = set()

        if isinstance(expr, FunctionCall):
            if expr.name in ('idtmod', 'transition'):
                restricted.add(expr.name)
            for arg in expr.args:
                restricted |= self._collect_restricted_calls_from_expr(arg)
            return restricted

        if isinstance(expr, BinaryExpr):
            restricted |= self._collect_restricted_calls_from_expr(expr.left)
            restricted |= self._collect_restricted_calls_from_expr(expr.right)
            return restricted

        if isinstance(expr, UnaryExpr):
            return self._collect_restricted_calls_from_expr(expr.operand)

        if isinstance(expr, TernaryExpr):
            restricted |= self._collect_restricted_calls_from_expr(expr.cond)
            restricted |= self._collect_restricted_calls_from_expr(expr.true_expr)
            restricted |= self._collect_restricted_calls_from_expr(expr.false_expr)
            return restricted

        if isinstance(expr, ArrayAccess):
            return self._collect_restricted_calls_from_expr(expr.index)

        if isinstance(expr, BranchAccess):
            if expr.node1_index is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node1_index)
            if expr.node1_index2 is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node1_index2)
            if expr.node2_index is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node2_index)
            if expr.node2_index2 is not None:
                restricted |= self._collect_restricted_calls_from_expr(expr.node2_index2)
            return restricted

        if isinstance(expr, MethodCall):
            for arg in expr.args:
                restricted |= self._collect_restricted_calls_from_expr(arg)
            return restricted

        return restricted

    def _infer_continuous_vars(self, stmt) -> set[str]:
        continuous_vars = set()
        changed = True
        while changed:
            changed = False
            for target_name, value_expr in self._iter_assignments(stmt):
                if target_name not in self._non_event_assigned_vars:
                    continue
                if self._expr_is_continuous(value_expr, continuous_vars) and target_name not in continuous_vars:
                    continuous_vars.add(target_name)
                    changed = True
        return continuous_vars

    def _collect_assignment_contexts(self, stmt, in_event: bool) -> None:
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._collect_assignment_contexts(child, in_event)
            return

        if isinstance(stmt, EventStatement):
            self._collect_assignment_contexts(stmt.body, True)
            return

        if isinstance(stmt, IfStatement):
            self._collect_assignment_contexts(stmt.then_body, in_event)
            if stmt.else_body is not None:
                self._collect_assignment_contexts(stmt.else_body, in_event)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._collect_assignment_contexts(item.body, in_event)
            return

        if isinstance(stmt, ForStatement):
            self._collect_assignment_contexts(stmt.body, in_event)
            return

        if isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, Identifier):
                name = target.name
            elif isinstance(target, ArrayAccess):
                name = target.name
            else:
                return
            if in_event:
                self._event_assigned_vars.add(name)
            else:
                self._non_event_assigned_vars.add(name)

    def _iter_assignments(self, stmt):
        if isinstance(stmt, Block):
            for child in stmt.statements:
                yield from self._iter_assignments(child)
            return

        if isinstance(stmt, Assignment):
            target = stmt.target
            if isinstance(target, Identifier):
                yield target.name, stmt.value
            elif isinstance(target, ArrayAccess):
                yield target.name, stmt.value
            return

        if isinstance(stmt, IfStatement):
            yield from self._iter_assignments(stmt.then_body)
            if stmt.else_body is not None:
                yield from self._iter_assignments(stmt.else_body)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                yield from self._iter_assignments(item.body)
            return

        if isinstance(stmt, EventStatement):
            yield from self._iter_assignments(stmt.body)
            return

        if isinstance(stmt, ForStatement):
            yield from self._iter_assignments(stmt.body)
            return

    def _expr_is_continuous(self, expr: Expr, continuous_vars: set[str]) -> bool:
        if isinstance(expr, NumberLiteral):
            return False

        if isinstance(expr, StringLiteral):
            return False

        if isinstance(expr, Identifier):
            return expr.name in continuous_vars

        if isinstance(expr, ArrayAccess):
            return expr.name in continuous_vars or self._expr_is_continuous(expr.index, continuous_vars)

        if isinstance(expr, BranchAccess):
            return True

        if isinstance(expr, BinaryExpr):
            return (
                self._expr_is_continuous(expr.left, continuous_vars)
                or self._expr_is_continuous(expr.right, continuous_vars)
            )

        if isinstance(expr, UnaryExpr):
            return self._expr_is_continuous(expr.operand, continuous_vars)

        if isinstance(expr, TernaryExpr):
            return (
                self._expr_is_continuous(expr.cond, continuous_vars)
                or self._expr_is_continuous(expr.true_expr, continuous_vars)
                or self._expr_is_continuous(expr.false_expr, continuous_vars)
            )

        if isinstance(expr, MethodCall):
            return any(self._expr_is_continuous(arg, continuous_vars) for arg in expr.args)

        if isinstance(expr, FunctionCall):
            if expr.name == 'transition':
                return True
            if expr.name == 'idtmod':
                return True
            return any(self._expr_is_continuous(arg, continuous_vars) for arg in expr.args)

        return False

    def _check_transition_targets(self, stmt) -> None:
        if isinstance(stmt, Block):
            for child in stmt.statements:
                self._check_transition_targets(child)
            return

        if isinstance(stmt, IfStatement):
            self._check_transition_targets(stmt.then_body)
            if stmt.else_body is not None:
                self._check_transition_targets(stmt.else_body)
            return

        if isinstance(stmt, CaseStatement):
            for item in stmt.items:
                self._check_transition_targets(item.body)
            return

        if isinstance(stmt, EventStatement):
            self._check_transition_targets(stmt.body)
            return

        if isinstance(stmt, ForStatement):
            self._check_transition_targets(stmt.body)
            return

        for call in self._iter_function_calls_in_stmt(stmt):
            if call.name == 'transition' and call.args:
                if self._transition_target_is_continuous(call.args[0]):
                    raise CompilationError(
                        f"Module {self.module.name} applies transition() to a continuous-valued "
                        f"expression. Spectre expects the transition target to be piecewise constant. "
                        f"Move continuous scaling outside transition() or contribute the signal directly."
                    )

    def _transition_target_is_continuous(self, expr: Expr) -> bool:
        if isinstance(expr, NumberLiteral):
            return False

        if isinstance(expr, StringLiteral):
            return False

        if isinstance(expr, Identifier):
            return expr.name in self._continuous_vars

        if isinstance(expr, ArrayAccess):
            return expr.name in self._continuous_vars

        if isinstance(expr, BranchAccess):
            return True

        if isinstance(expr, UnaryExpr):
            return self._transition_target_is_continuous(expr.operand)

        if isinstance(expr, BinaryExpr):
            return (
                self._transition_target_is_continuous(expr.left)
                or self._transition_target_is_continuous(expr.right)
            )

        if isinstance(expr, TernaryExpr):
            return (
                self._transition_target_is_continuous(expr.true_expr)
                or self._transition_target_is_continuous(expr.false_expr)
            )

        if isinstance(expr, MethodCall):
            return any(self._transition_target_is_continuous(arg) for arg in expr.args)

        if isinstance(expr, FunctionCall):
            if expr.name in ('idtmod', 'transition'):
                return True
            return any(self._transition_target_is_continuous(arg) for arg in expr.args)

        return False

    def _iter_function_calls_in_stmt(self, stmt):
        if isinstance(stmt, Assignment):
            yield from self._iter_function_calls_in_expr(stmt.value)
            return

        if isinstance(stmt, Contribution):
            yield from self._iter_function_calls_in_expr(stmt.expr)
            return

        if isinstance(stmt, SystemTask):
            for arg in stmt.args:
                yield from self._iter_function_calls_in_expr(arg)

    def _iter_function_calls_in_expr(self, expr: Expr):
        if isinstance(expr, FunctionCall):
            yield expr
            for arg in expr.args:
                yield from self._iter_function_calls_in_expr(arg)
            return

        if isinstance(expr, BinaryExpr):
            yield from self._iter_function_calls_in_expr(expr.left)
            yield from self._iter_function_calls_in_expr(expr.right)
            return

        if isinstance(expr, UnaryExpr):
            yield from self._iter_function_calls_in_expr(expr.operand)
            return

        if isinstance(expr, TernaryExpr):
            yield from self._iter_function_calls_in_expr(expr.cond)
            yield from self._iter_function_calls_in_expr(expr.true_expr)
            yield from self._iter_function_calls_in_expr(expr.false_expr)
            return

        if isinstance(expr, ArrayAccess):
            yield from self._iter_function_calls_in_expr(expr.index)
            return

        if isinstance(expr, BranchAccess):
            if expr.node1_index is not None:
                yield from self._iter_function_calls_in_expr(expr.node1_index)
            if expr.node1_index2 is not None:
                yield from self._iter_function_calls_in_expr(expr.node1_index2)
            if expr.node2_index is not None:
                yield from self._iter_function_calls_in_expr(expr.node2_index)
            if expr.node2_index2 is not None:
                yield from self._iter_function_calls_in_expr(expr.node2_index2)
            return

        if isinstance(expr, MethodCall):
            for arg in expr.args:
                yield from self._iter_function_calls_in_expr(arg)

    def _is_initial_step_event(self, event) -> bool:
        """Check if event includes initial_step."""
        if isinstance(event, EventExpr) and event.event_type == EventType.INITIAL_STEP:
            return True
        if isinstance(event, CombinedEvent):
            return any(e.event_type == EventType.INITIAL_STEP for e in event.events)
        return False

    def _is_final_step_event(self, event) -> bool:
        """Check if event includes final_step."""
        if isinstance(event, EventExpr) and event.event_type == EventType.FINAL_STEP:
            return True
        if isinstance(event, CombinedEvent):
            return any(e.event_type == EventType.FINAL_STEP for e in event.events)
        return False

    def _compile_statement(self, stmt, indent) -> List[str]:
        """Compile a statement to Python code lines."""
        prefix = '    ' * indent
        lines = []

        if isinstance(stmt, Block):
            for s in stmt.statements:
                lines.extend(self._compile_statement(s, indent))

        elif isinstance(stmt, EventStatement):
            lines.extend(self._compile_event_statement(stmt, indent))

        elif isinstance(stmt, Contribution):
            lines.extend(self._compile_contribution(stmt, indent))

        elif isinstance(stmt, Assignment):
            lines.extend(self._compile_assignment(stmt, indent))

        elif isinstance(stmt, IfStatement):
            cond = self._compile_expr(stmt.cond)
            lines.append(f"{prefix}if {cond}:")
            body_lines = self._compile_statement(stmt.then_body, indent + 1)
            lines.extend(body_lines)
            if not body_lines:
                lines.append(f"{prefix}    pass")
            if stmt.else_body:
                lines.append(f"{prefix}else:")
                else_lines = self._compile_statement(stmt.else_body, indent + 1)
                lines.extend(else_lines)
                if not else_lines:
                    lines.append(f"{prefix}    pass")

        elif isinstance(stmt, ForStatement):
            lines.extend(self._compile_for(stmt, indent))

        elif isinstance(stmt, CaseStatement):
            lines.extend(self._compile_case(stmt, indent))

        elif isinstance(stmt, SystemTask):
            # $strobe, $display → collect output
            if stmt.name in ('$strobe', '$display'):
                if stmt.args:
                    fmt_expr = self._compile_expr(stmt.args[0])
                    rest = ', '.join(self._compile_expr(a) for a in stmt.args[1:])
                    if rest:
                        lines.append(f"{prefix}self._strobe(time, {fmt_expr}, {rest})")
                    else:
                        lines.append(f"{prefix}self._strobe(time, {fmt_expr})")
                else:
                    lines.append(f"{prefix}self._strobe(time, '')")
            elif stmt.name == '$bound_step' and stmt.args:
                val = self._compile_expr(stmt.args[0])
                lines.append(f"{prefix}self._bound_step = {val}")
            elif stmt.name == '$fclose' and stmt.args:
                fd = self._compile_expr(stmt.args[0])
                lines.append(f"{prefix}self._fclose(int({fd}))")
            elif stmt.name in ('$fstrobe', '$fwrite', '$fdisplay') and stmt.args:
                fd = self._compile_expr(stmt.args[0])
                if len(stmt.args) > 1:
                    fmt_expr = self._compile_expr(stmt.args[1])
                    rest = ', '.join(self._compile_expr(a) for a in stmt.args[2:])
                    if rest:
                        lines.append(f"{prefix}self._fstrobe(int({fd}), {fmt_expr}, {rest})")
                    else:
                        lines.append(f"{prefix}self._fstrobe(int({fd}), {fmt_expr})")
                else:
                    lines.append(f"{prefix}self._fstrobe(int({fd}), '')")

        return lines

    def _compile_event_statement(self, stmt: EventStatement, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []
        event = stmt.event

        if isinstance(event, EventExpr):
            if event.event_type == EventType.INITIAL_STEP:
                # Skip in evaluate — handled in initial_step method
                return []

            elif event.event_type == EventType.FINAL_STEP:
                # Skip in evaluate — handled in final_step method
                return []

            elif event.event_type == EventType.CROSS:
                key = f"cross_{self._cross_counter}"
                self._cross_counter += 1
                expr = self._compile_expr(event.args[0])
                direction = event.direction if event.direction is not None else 0
                lines.append(f"{prefix}if self._check_cross({key!r}, time, {expr}, {direction}):")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_time = time")

            elif event.event_type == EventType.ABOVE:
                key = f"above_{self._above_counter}"
                self._above_counter += 1
                expr = self._compile_expr(event.args[0])
                direction = event.direction if event.direction is not None else 1
                lines.append(f"{prefix}if self._check_above({key!r}, time, {expr}, {direction}):")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_time = time")

            elif event.event_type == EventType.TIMER:
                key = f"timer_{self._timer_counter}"
                self._timer_counter += 1
                period_expr = self._compile_expr(event.args[0])
                lines.append(f"{prefix}if self._check_timer({key!r}, time, {period_expr}):")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_time = time")

        elif isinstance(event, CombinedEvent):
            # Combined events: @(initial_step or cross(...))
            conditions = []
            for e in event.events:
                if e.event_type == EventType.INITIAL_STEP:
                    # In evaluate, initial_step never fires again
                    continue
                elif e.event_type == EventType.FINAL_STEP:
                    # In evaluate, final_step never fires
                    continue
                elif e.event_type == EventType.CROSS:
                    key = f"cross_{self._cross_counter}"
                    self._cross_counter += 1
                    expr = self._compile_expr(e.args[0])
                    direction = e.direction if e.direction is not None else 0
                    conditions.append(f"self._check_cross({key!r}, time, {expr}, {direction})")
                elif e.event_type == EventType.ABOVE:
                    key = f"above_{self._above_counter}"
                    self._above_counter += 1
                    expr = self._compile_expr(e.args[0])
                    direction = e.direction if e.direction is not None else 1
                    conditions.append(f"self._check_above({key!r}, time, {expr}, {direction})")
                elif e.event_type == EventType.TIMER:
                    key = f"timer_{self._timer_counter}"
                    self._timer_counter += 1
                    period_expr = self._compile_expr(e.args[0])
                    conditions.append(f"self._check_timer({key!r}, time, {period_expr})")

            if conditions:
                cond = ' or '.join(conditions)
                lines.append(f"{prefix}if {cond}:")
                body_lines = self._compile_statement(stmt.body, indent + 1)
                lines.extend(body_lines)
                if not body_lines:
                    lines.append(f"{prefix}    pass")
                lines.append(f"{prefix}    self._event_time = time")

        return lines

    def _compile_contribution(self, stmt: Contribution, indent) -> List[str]:
        prefix = '    ' * indent
        branch = stmt.branch
        node = branch.node1
        expr = self._compile_expr(stmt.expr)

        if branch.node1_index is not None:
            # Dynamic array-indexed port: V(DOUT[i]) <+ or V(DOUT[i][j]) <+
            idx_expr = self._compile_expr(branch.node1_index)
            if branch.node1_index2 is not None:
                idx_expr2 = self._compile_expr(branch.node1_index2)
                return [f"{prefix}self._set_output(f'{node}[{{int({idx_expr})}}][{{int({idx_expr2})}}]', {expr}, nv)"]
            return [f"{prefix}self._set_output(f'{node}[{{int({idx_expr})}}]', {expr}, nv)"]
        else:
            return [f"{prefix}self._set_output({node!r}, {expr}, nv)"]

    def _compile_assignment(self, stmt: Assignment, indent) -> List[str]:
        prefix = '    ' * indent
        val = self._compile_expr(stmt.value)

        if isinstance(stmt.target, Identifier):
            name = stmt.target.name
            return [f"{prefix}self.state[{name!r}] = {val}"]

        elif isinstance(stmt.target, ArrayAccess):
            name = stmt.target.name
            idx = self._compile_expr(stmt.target.index)
            return [f"{prefix}self._array_set({name!r}, int({idx}), {val})"]

        return [f"{prefix}pass  # unknown assignment target"]

    def _compile_for(self, stmt: ForStatement, indent) -> List[str]:
        prefix = '    ' * indent
        lines = []

        # Extract loop variable, start, end, step
        loop_var = None
        if isinstance(stmt.init.target, Identifier):
            loop_var = stmt.init.target.name
        elif isinstance(stmt.init.target, ArrayAccess):
            loop_var = stmt.init.target.name

        if loop_var is None:
            return [f"{prefix}pass  # could not compile for loop"]

        init_val = self._compile_expr(stmt.init.value)

        # Track that we're in a loop (for dynamic transition/contribution keys)
        prev_loop_var = self._in_loop_var
        self._in_loop_var = loop_var

        # Use a while loop in Python
        lines.append(f"{prefix}self.state[{loop_var!r}] = {init_val}")
        # Replace loop var references in condition
        lines.append(f"{prefix}_loop_{loop_var} = {init_val}")
        cond_code2 = self._compile_expr_with_loop_var(stmt.cond, loop_var)
        lines.append(f"{prefix}while {cond_code2}:")
        lines.append(f"{prefix}    self.state[{loop_var!r}] = _loop_{loop_var}")
        body_lines = self._compile_statement_with_loop_var(stmt.body, indent + 1, loop_var)
        lines.extend(body_lines)
        update_code2 = self._compile_expr_with_loop_var(stmt.update.value, loop_var)
        lines.append(f"{prefix}    _loop_{loop_var} = {update_code2}")
        lines.append(f"{prefix}self.state[{loop_var!r}] = _loop_{loop_var}")

        self._in_loop_var = prev_loop_var
        return lines

    def _compile_case(self, stmt: CaseStatement, indent) -> List[str]:
        """Compile a case statement to chained if/elif/else."""
        prefix = '    ' * indent
        lines = []
        sel = self._compile_expr(stmt.expr)
        first = True
        default_lines = None
        for item in stmt.items:
            if not item.values:
                # default branch — emit last
                default_lines = self._compile_statement(item.body, indent + 1)
                continue
            cond_parts = [f"({sel} == {self._compile_expr(v)})" for v in item.values]
            cond = ' or '.join(cond_parts)
            keyword = 'if' if first else 'elif'
            lines.append(f"{prefix}{keyword} {cond}:")
            body_lines = self._compile_statement(item.body, indent + 1)
            lines.extend(body_lines)
            if not body_lines:
                lines.append(f"{prefix}    pass")
            first = False
        if default_lines is not None:
            if first:
                # only default, no value branches
                lines.extend(default_lines)
            else:
                lines.append(f"{prefix}else:")
                lines.extend(default_lines)
                if not default_lines:
                    lines.append(f"{prefix}    pass")
        return lines

    def _compile_expr(self, expr: Expr) -> str:
        """Compile an expression to Python code string."""
        if isinstance(expr, NumberLiteral):
            return repr(expr.value)

        if isinstance(expr, StringLiteral):
            return repr(expr.value)

        if isinstance(expr, Identifier):
            name = expr.name
            # Check if it's a parameter
            for p in self.module.parameters:
                if p.name == name:
                    return f"self.params[{name!r}]"
            # Check if it's a variable
            for v in self.module.variables:
                if v.name == name:
                    return f"self.state[{name!r}]"
            # Check if it's a special constant
            if name == 'inf':
                return "float('inf')"
            if name == '$abstime':
                return "self._event_time"
            if name == '$temperature':
                return "(self._temperature + 273.15)"
            if name == '$vt':
                return "(1.380649e-23 * (self._temperature + 273.15) / 1.602176634e-19)"
            return f"self.state[{name!r}]"

        if isinstance(expr, ArrayAccess):
            idx = self._compile_expr(expr.index)
            return f"self._array_get({expr.name!r}, int({idx}))"

        if isinstance(expr, BinaryExpr):
            left = self._compile_expr(expr.left)
            right = self._compile_expr(expr.right)
            op = expr.op
            if op == '^':
                # In Verilog-A, ^ is XOR for integers
                return f"(int({left}) ^ int({right}))"
            if op == '&':
                return f"(int({left}) & int({right}))"
            if op == '|':
                return f"(int({left}) | int({right}))"
            if op == '<<':
                return f"(int({left}) << int({right}))"
            if op == '>>':
                return f"(int({left}) >> int({right}))"
            if op == '&&':
                return f"(({left}) and ({right}))"
            if op == '||':
                return f"(({left}) or ({right}))"
            return f"({left} {op} {right})"

        if isinstance(expr, UnaryExpr):
            operand = self._compile_expr(expr.operand)
            if expr.op == '!':
                return f"(not ({operand}))"
            if expr.op == '~':
                return f"(~int({operand}))"
            return f"({expr.op}{operand})"

        if isinstance(expr, TernaryExpr):
            cond = self._compile_expr(expr.cond)
            true_e = self._compile_expr(expr.true_expr)
            false_e = self._compile_expr(expr.false_expr)
            return f"(({true_e}) if ({cond}) else ({false_e}))"

        if isinstance(expr, BranchAccess):
            node = expr.node1
            if expr.node2:
                n1 = self._compile_node_voltage(expr.node1, expr.node1_index, expr.node1_index2)
                n2 = self._compile_node_voltage(expr.node2, expr.node2_index, expr.node2_index2)
                if expr.access_type == 'V':
                    return f"({n1} - {n2})"
                return "0.0"  # I() not fully supported yet
            if expr.access_type == 'V':
                return self._compile_node_voltage(node, expr.node1_index, expr.node1_index2)
            return "0.0"

        if isinstance(expr, FunctionCall):
            return self._compile_function_call(expr)

        if isinstance(expr, MethodCall):
            return self._compile_method_call(expr)

        return "0.0"

    def _compile_node_voltage(self, node: str, index_expr=None, index_expr2=None) -> str:
        """Compile a node voltage reference."""
        if index_expr is not None:
            idx = self._compile_expr(index_expr)
            if index_expr2 is not None:
                idx2 = self._compile_expr(index_expr2)
                return f"self._get_voltage(f'{node}[{{int({idx})}}][{{int({idx2})}}]', nv)"
            return f"self._get_voltage(f'{node}[{{int({idx})}}]', nv)"
        return f"self._get_voltage({node!r}, nv)"

    def _compile_instance_target(self, expr: Expr) -> str:
        """Compile instance connection target into a node-name string expression."""
        if isinstance(expr, Identifier):
            return repr(expr.name)
        if isinstance(expr, ArrayAccess):
            idx = self._compile_expr(expr.index)
            return f"f'{expr.name}[{{int({idx})}}]'"
        # Fallback: allow unusual connection expressions as stringified value.
        return f"str({self._compile_expr(expr)})"

    def _compile_function_call(self, expr: FunctionCall) -> str:
        name = expr.name
        args = [self._compile_expr(a) for a in expr.args]

        if name == 'transition':
            base_key = f"trans_{self._trans_counter}"
            self._trans_counter += 1
            target = args[0] if len(args) > 0 else "0.0"
            delay = args[1] if len(args) > 1 else "0.0"
            rise = args[2] if len(args) > 2 else "0.0"
            fall = args[3] if len(args) > 3 else "0.0"
            if self._in_loop_var:
                # Dynamic key per loop iteration
                return f"self._transition(f'{base_key}_{{int(_loop_{self._in_loop_var})}}', time, {target}, {delay}, {rise}, {fall})"
            return f"self._transition({base_key!r}, time, {target}, {delay}, {rise}, {fall})"

        if name == 'idtmod':
            key = f"idtmod_{self._idt_counter}"
            self._idt_counter += 1
            self._uses_idtmod = True
            x = args[0] if len(args) > 0 else "0.0"
            ic = args[1] if len(args) > 1 else "0.0"
            mod = args[2] if len(args) > 2 else "1.0"
            return f"self._idtmod({key!r}, time, {x}, {ic}, {mod})"

        if name == 'cross':
            # cross() as a function (in some contexts)
            base_key = f"cross_fn_{self._cross_counter}"
            self._cross_counter += 1
            val = args[0]
            direction = args[1] if len(args) > 1 else "0"
            return f"self._check_cross({base_key!r}, time, {val}, {direction})"

        if name == 'ln':
            return f"math.log({args[0]})"
        if name == 'log':
            return f"math.log10({args[0]})"
        if name == 'exp':
            return f"math.exp({args[0]})"
        if name == 'sqrt':
            return f"math.sqrt({args[0]})"
        if name == 'abs':
            return f"abs({args[0]})"
        if name == 'pow':
            return f"pow({args[0]}, {args[1]})"
        if name == 'min':
            return f"min({args[0]}, {args[1]})"
        if name == 'max':
            return f"max({args[0]}, {args[1]})"
        if name == 'sin':
            return f"math.sin({args[0]})"
        if name == 'cos':
            return f"math.cos({args[0]})"
        if name == 'floor':
            return f"math.floor({args[0]})"
        if name == 'ceil':
            return f"math.ceil({args[0]})"
        if name == '$rdist_normal':
            # $rdist_normal(seed, mean, std_dev) — ignore seed, use random.gauss
            mean = args[1] if len(args) > 1 else "0.0"
            std  = args[2] if len(args) > 2 else "1.0"
            return f"random.gauss({mean}, {std})"
        if name == '$random':
            return "random.randint(-2147483648, 2147483647)"
        if name == '$dist_uniform':
            lo = args[1] if len(args) > 1 else "0.0"
            hi = args[2] if len(args) > 2 else "1.0"
            return f"random.uniform({lo}, {hi})"
        if name == '$fopen':
            filename = args[0] if len(args) > 0 else "'output.txt'"
            mode = args[1] if len(args) > 1 else "'w'"
            return f"self._fopen({filename}, {mode})"

        return "0.0"  # unknown function: {name}

    def _compile_method_call(self, expr: MethodCall) -> str:
        """Compile method calls like conf.substr(i, i)."""
        obj = expr.obj
        method = expr.method
        args = [self._compile_expr(a) for a in expr.args]

        if method == 'substr':
            # Verilog-A substr(start, end) → Python string slice
            return f"self.params[{obj!r}][int({args[0]}):int({args[1]})+1]"

        return "''"  # unknown method

    def _compile_expr_with_loop_var(self, expr: Expr, loop_var: str) -> str:
        """Compile expression using loop variable from local scope."""
        code = self._compile_expr(expr)
        code = code.replace(f"self.state[{loop_var!r}]", f"_loop_{loop_var}")
        return code

    def _compile_statement_with_loop_var(self, stmt, indent, loop_var) -> List[str]:
        """Compile statement but use loop var from local scope."""
        lines = self._compile_statement(stmt, indent)
        new_lines = []
        for line in lines:
            # Replace state access for loop var with local var
            # But only in index positions, not as assignment targets at top level
            new_line = line.replace(
                f"self.state[{loop_var!r}]",
                f"_loop_{loop_var}"
            )
            new_lines.append(new_line)
        return new_lines

    def _eval_expr_static(self, expr: Expr) -> Any:
        """Evaluate a constant expression statically."""
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, StringLiteral):
            return expr.value
        if isinstance(expr, UnaryExpr) and expr.op == '-':
            return -self._eval_expr_static(expr.operand)
        if isinstance(expr, Identifier):
            if expr.name == 'inf':
                return float('inf')
            return 0
        if isinstance(expr, BinaryExpr):
            lv = self._eval_expr_static(expr.left)
            rv = self._eval_expr_static(expr.right)
            if expr.op == '+':
                return lv + rv
            if expr.op == '-':
                return lv - rv
            if expr.op == '*':
                return lv * rv
            if expr.op == '/':
                return lv / rv if rv != 0 else 0
        return 0


def compile_va_file(va_path: str, source_dir: str = None) -> type:
    """
    Compile a .va file into a Python model class.

    Usage:
        ModelClass = compile_va_file('veriloga/L2_comparator.va')
        model = ModelClass()
        model.node_map = {'DCMPP': 'out_p', 'CLK': 'clk', ...}
    """
    from evas.compiler.parser import parse
    from evas.compiler.preprocessor import preprocess

    if source_dir is None:
        source_dir = str(Path(va_path).parent)

    source = Path(va_path).read_text(encoding='utf-8', errors='replace')
    pp_src, defines, default_trans = preprocess(source, source_dir=source_dir)
    module = parse(pp_src)
    module.defines = defines

    if default_trans is None:
        default_trans = 1e-12

    return compile_module(module, default_trans)
