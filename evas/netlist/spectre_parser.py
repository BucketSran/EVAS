"""
spectre_parser.py — Spectre .scs netlist parser for NexSim.

Parses real Cadence Spectre netlist syntax:
  simulator lang=spectre       — Accept and skip
  global 0                     — Record ground node
  parameters k=expr ...        — Global parameters with expression evaluation
  include "file" [section=X]   — Skip (process models)
  ahdl_include "file.va"       — VA model include
  Vname (n+ n-) vsource ...    — DC/pulse/pwl/sin source
  Iname (nodes) Model k=v      — VA model instance
  tran tran stop=val ...       — Transient analysis
  save node1 node2 ...         — Signals to record
  simulatorOptions options ...  — Parse temp, skip rest
  // comment                   — C-style line comments
  \\ at EOL                    — Line continuation
"""
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# AST nodes
# ---------------------------------------------------------------------------

@dataclass
class AhdlInclude:
    path: str  # VA file path as written in netlist


@dataclass
class SpectreSource:
    """A voltage source parsed from Spectre syntax."""
    name: str
    node_pos: str
    node_neg: str
    source_type: str  # 'dc', 'pulse', 'pwl', 'sin'
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectreInstance:
    """A subcircuit/model instance."""
    name: str
    nodes: List[str]
    model_name: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectreTran:
    """Transient analysis parameters."""
    stop: float
    step: Optional[float] = None  # maxstep or computed
    name: str = 'tran'
    refine_factor: int = 16   # step divisor after a cross event
    refine_steps: int = 8     # number of refined steps after a cross event


@dataclass
class SpectreMosfet:
    """A MOSFET device inside a subckt."""
    name: str         # M88
    nodes: List[str]  # [drain, gate, source, bulk]
    model: str        # nch_ulvt_mac
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SpectreSubckt:
    """A subcircuit definition containing MOSFETs and/or instances."""
    name: str
    ports: List[str]
    mosfets: List[SpectreMosfet] = field(default_factory=list)
    instances: List[SpectreInstance] = field(default_factory=list)
    body_lines: List[str] = field(default_factory=list)  # raw lines for pass-through


@dataclass
class SpectreInclude:
    """An include statement with optional section."""
    path: str
    section: Optional[str] = None  # e.g. "TOP_TT"


@dataclass
class SpectreNetlist:
    title: str = ''
    ground: str = '0'
    parameters: Dict[str, float] = field(default_factory=dict)
    ahdl_includes: List[AhdlInclude] = field(default_factory=list)
    sources: List[SpectreSource] = field(default_factory=list)
    instances: List[SpectreInstance] = field(default_factory=list)
    subckts: List[SpectreSubckt] = field(default_factory=list)
    includes: List[SpectreInclude] = field(default_factory=list)
    tran: Optional[SpectreTran] = None
    save_signals: List[str] = field(default_factory=list)
    save_formats: Dict[str, str] = field(default_factory=dict)  # sig -> fmt e.g. '6e', '10e', 'd'
    temp: float = 27.0
    source_dir: str = ''


# ---------------------------------------------------------------------------
# SPICE suffix parser
# ---------------------------------------------------------------------------

_SUFFIXES = {
    'T': 1e12, 'G': 1e9, 'MEG': 1e6, 'X': 1e6, 'K': 1e3,
    'M': 1e-3, 'U': 1e-6, 'N': 1e-9, 'P': 1e-12, 'F': 1e-15, 'A': 1e-18,
}


def _parse_suffix_number(s: str) -> Optional[float]:
    """Try to parse a number with optional SPICE suffix. Returns None if not a number."""
    s = s.strip()
    if not s:
        return None

    # Try direct float first
    try:
        return float(s)
    except ValueError:
        pass

    # Try suffix match (longest first)
    s_upper = s.upper()
    for suffix in sorted(_SUFFIXES.keys(), key=len, reverse=True):
        if s_upper.endswith(suffix):
            num_part = s_upper[:-len(suffix)]
            try:
                return float(num_part) * _SUFFIXES[suffix]
            except ValueError:
                continue

    return None


# ---------------------------------------------------------------------------
# Expression evaluator (for `parameters` block)
# ---------------------------------------------------------------------------

class _ExprEvaluator:
    """Simple recursive-descent expression evaluator.

    Supports: +, -, *, /, unary minus, parentheses, variable substitution,
    and SPICE-suffix numbers.
    """

    def __init__(self, variables: Dict[str, float]):
        self.variables = variables
        self.pos = 0
        self.expr = ''

    def evaluate(self, expr: str) -> float:
        self.expr = expr.strip()
        self.pos = 0
        result = self._parse_expr()
        return result

    def _peek(self) -> Optional[str]:
        self._skip_ws()
        if self.pos < len(self.expr):
            return self.expr[self.pos]
        return None

    def _skip_ws(self):
        while self.pos < len(self.expr) and self.expr[self.pos] in ' \t':
            self.pos += 1

    def _parse_expr(self) -> float:
        """expr = term (('+' | '-') term)*"""
        left = self._parse_term()
        while True:
            op = self._peek()
            if op == '+':
                self.pos += 1
                left += self._parse_term()
            elif op == '-':
                self.pos += 1
                left -= self._parse_term()
            else:
                break
        return left

    def _parse_term(self) -> float:
        """term = unary (('*' | '/') unary)*"""
        left = self._parse_unary()
        while True:
            op = self._peek()
            if op == '*':
                self.pos += 1
                left *= self._parse_unary()
            elif op == '/':
                self.pos += 1
                right = self._parse_unary()
                left = left / right if right != 0 else 0.0
            else:
                break
        return left

    def _parse_unary(self) -> float:
        """unary = '-' unary | '+' unary | atom"""
        ch = self._peek()
        if ch == '-':
            self.pos += 1
            return -self._parse_unary()
        if ch == '+':
            self.pos += 1
            return self._parse_unary()
        return self._parse_atom()

    def _parse_atom(self) -> float:
        """atom = '(' expr ')' | number_with_suffix | variable"""
        self._skip_ws()

        # Parenthesized expression
        if self.pos < len(self.expr) and self.expr[self.pos] == '(':
            self.pos += 1
            val = self._parse_expr()
            self._skip_ws()
            if self.pos < len(self.expr) and self.expr[self.pos] == ')':
                self.pos += 1
            return val

        # Read token: number or variable name
        # Handle scientific notation: e.g. 5e-11, 1.2E+3
        start = self.pos
        while self.pos < len(self.expr) and self.expr[self.pos] not in '+-*/() \t':
            self.pos += 1
        # Check if we stopped at +/- that's part of scientific notation (e.g. 5e-11)
        while (self.pos < len(self.expr) and
               self.pos > start and
               self.expr[self.pos] in '+-' and
               self.expr[self.pos - 1] in 'eE'):
            self.pos += 1  # consume the +/-
            # Continue reading digits/suffix
            while self.pos < len(self.expr) and self.expr[self.pos] not in '+-*/() \t':
                self.pos += 1
        token = self.expr[start:self.pos]

        if not token:
            return 0.0

        # Try as number with suffix
        val = _parse_suffix_number(token)
        if val is not None:
            return val

        # Try as variable
        if token in self.variables:
            return self.variables[token]

        # Unknown — try harder with case-insensitive variable lookup
        for k, v in self.variables.items():
            if k.lower() == token.lower():
                return v

        raise ValueError(f"Unknown variable or invalid number: {token!r}")


def evaluate_expr(expr: str, variables: Dict[str, float]) -> float:
    """Evaluate an expression string with variable substitution."""
    return _ExprEvaluator(variables).evaluate(expr)


# ---------------------------------------------------------------------------
# Line preprocessing: strip comments, handle continuation
# ---------------------------------------------------------------------------

def _preprocess_lines(raw_lines: List[str]) -> List[str]:
    """Strip comments, handle \\ continuation, return clean lines."""
    result = []
    continuation = ''

    for raw in raw_lines:
        # Strip trailing whitespace
        line = raw.rstrip()

        # Remove // comments (but not inside quotes)
        in_quote = False
        quote_char = None
        for i, ch in enumerate(line):
            if in_quote:
                if ch == quote_char:
                    in_quote = False
            else:
                if ch in ('"', "'"):
                    in_quote = True
                    quote_char = ch
                elif ch == '/' and i + 1 < len(line) and line[i + 1] == '/':
                    line = line[:i].rstrip()
                    break

        # Handle continuation
        if line.endswith('\\'):
            continuation += line[:-1].rstrip() + ' '
            continue

        if continuation:
            line = continuation + line.lstrip()
            continuation = ''

        stripped = line.strip()
        if stripped:
            result.append(stripped)

    # Flush any trailing continuation
    if continuation:
        result.append(continuation.strip())

    return result


# ---------------------------------------------------------------------------
# Source parameter parsing
# ---------------------------------------------------------------------------

def _parse_named_params(tokens: List[str], start: int,
                        variables: Dict[str, float]) -> Dict[str, Any]:
    """Parse key=value pairs from token list starting at index `start`."""
    params = {}
    for tok in tokens[start:]:
        if '=' not in tok:
            continue
        key, val_str = tok.split('=', 1)
        key = key.strip().lower()

        # Handle bracket-enclosed arrays: wave=[t0 v0 t1 v1 ...]
        if val_str.startswith('['):
            # Collect everything until ']'
            val_str = val_str[1:]  # strip leading [
            # This is handled separately for PWL
            params[key] = val_str
            continue

        # Try to evaluate as expression
        try:
            params[key] = evaluate_expr(val_str, variables)
        except (ValueError, ZeroDivisionError):
            params[key] = val_str  # keep as string

    return params


def _build_source(name: str, node_pos: str, node_neg: str,
                  params: Dict[str, Any]) -> SpectreSource:
    """Build a SpectreSource from parsed named parameters."""
    stype = str(params.get('type', 'dc')).lower()

    src = SpectreSource(
        name=name, node_pos=node_pos, node_neg=node_neg,
        source_type=stype, params=params,
    )
    return src


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_spectre(filepath: str) -> SpectreNetlist:
    """Parse a Spectre .scs netlist file into a SpectreNetlist AST."""
    filepath = Path(filepath).resolve()
    netlist = SpectreNetlist(source_dir=str(filepath.parent))

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        raw_lines = f.readlines()

    lines = _preprocess_lines(raw_lines)
    evaluator_vars = {}  # accumulated parameter variables

    # Extract title from first comment line if present
    for raw in raw_lines:
        stripped = raw.strip()
        if stripped.startswith('//'):
            netlist.title = stripped.lstrip('/').strip()
            break

    for idx in range(len(lines)):
        pass  # handled by while loop below

    idx = 0
    while idx < len(lines):
        line = lines[idx]
        low = line.lower().strip()

        # Skip directives
        if low.startswith('simulator'):
            idx += 1
            continue

        # Global ground
        if low.startswith('global'):
            parts = line.split()
            if len(parts) >= 2:
                netlist.ground = parts[1]
            idx += 1
            continue

        # Parameters block
        if low.startswith('parameters'):
            _parse_parameters(line, evaluator_vars)
            netlist.parameters = dict(evaluator_vars)
            idx += 1
            continue

        # Include (process models)
        if low.startswith('include'):
            _parse_include(line, netlist)
            idx += 1
            continue

        # ahdl_include
        if low.startswith('ahdl_include'):
            rest = line[len('ahdl_include'):].strip()
            path = rest.strip('"').strip("'")
            netlist.ahdl_includes.append(AhdlInclude(path=path))
            idx += 1
            continue

        # simulatorOptions
        if low.startswith('simulatoroptions') or low.startswith('simulatorOptions'):
            _parse_simulator_options(line, netlist, evaluator_vars)
            idx += 1
            continue

        # subckt block
        if low.startswith('subckt '):
            idx = _parse_subckt_block(lines, idx, netlist, evaluator_vars)
            continue

        # Transient analysis: "tran tran stop=..."
        if low.startswith('tran'):
            _parse_tran(line, netlist, evaluator_vars)
            idx += 1
            continue

        # saveOptions, info, finalTimeOP, etc. — skip
        if low.startswith('saveoptions') or low.startswith('info ') or \
           low.startswith('finaltimeop') or low.startswith('modelparameter') or \
           low.startswith('element ') or low.startswith('outputparameter') or \
           low.startswith('designparamvals') or low.startswith('primitives ') or \
           low.startswith('subckts '):
            idx += 1
            continue

        # save statement
        if low.startswith('save'):
            parts = line.split()
            for sig in parts[1:]:
                sig = _normalize_node_name(sig.strip())
                if not sig or sig.startswith('options'):
                    continue
                if ':' in sig:
                    name, fmt = sig.split(':', 1)
                    netlist.save_signals.append(name)
                    netlist.save_formats[name] = fmt
                else:
                    netlist.save_signals.append(sig)
            idx += 1
            continue

        # Voltage source: Vname (node node) vsource ...
        if line[0] in ('V', 'v') and '(' in line:
            _parse_vsource(line, netlist, evaluator_vars)
            idx += 1
            continue

        # Instance: Iname (nodes) ModelName k=v ...
        # Any line starting with a name and containing parenthesized nodes
        if '(' in line and not low.startswith('save'):
            _parse_instance(line, netlist, evaluator_vars)
            idx += 1
            continue

        idx += 1

    return netlist


def _parse_parameters(line: str, variables: Dict[str, float]):
    """Parse a `parameters` line, evaluating expressions in order."""
    rest = line[len('parameters'):].strip()

    # Split on whitespace, but respect that expressions don't have spaces
    # (Spectre parameters are space-separated key=value pairs)
    pairs = re.findall(r'(\w+)\s*=\s*([^\s]+)', rest)

    for name, expr in pairs:
        try:
            val = evaluate_expr(expr, variables)
            variables[name] = val
        except (ValueError, ZeroDivisionError):
            variables[name] = 0.0


def _parse_simulator_options(line: str, netlist: SpectreNetlist,
                              variables: Dict[str, float]):
    """Parse simulatorOptions — extract temp."""
    params = _parse_named_params(line.split(), 2, variables)
    if 'temp' in params:
        netlist.temp = float(params['temp'])


def _parse_tran(line: str, netlist: SpectreNetlist,
                variables: Dict[str, float]):
    """Parse transient analysis: tran tran stop=val [maxstep=val] ..."""
    tokens = line.split()
    params = _parse_named_params(tokens, 1, variables)

    stop = params.get('stop', 0.0)
    if isinstance(stop, str):
        stop = evaluate_expr(stop, variables)

    maxstep = params.get('maxstep', None)
    if maxstep is not None and isinstance(maxstep, str):
        maxstep = evaluate_expr(maxstep, variables)

    # Default step: stop / 1000
    step = maxstep if maxstep is not None else float(stop) / 1000.0

    refine_factor = int(params.get('refine_factor', 16))
    refine_steps  = int(params.get('refine_steps',  8))

    # Find analysis name (first token that's not a key=value)
    name = 'tran'
    for tok in tokens[1:]:
        if '=' not in tok:
            name = tok
            break

    netlist.tran = SpectreTran(stop=float(stop), step=float(step), name=name,
                               refine_factor=refine_factor, refine_steps=refine_steps)


def _normalize_node_name(name: str) -> str:
    """Normalize Cadence-escaped bus subscripts to plain angle-bracket form.

    Cadence Spectre exports bus pins as ``DOUT\\<9\\>`` (backslash-escaped).
    We strip the backslashes so node names become ``DOUT<9>`` throughout —
    cleaner in CSVs and consistent with what the user sees in Virtuoso.
    """
    return name.replace('\\<', '<').replace('\\>', '>')


def _extract_nodes(line: str) -> Tuple[str, List[str], str]:
    """Extract (name, nodes, remainder) from a line like 'Vname (n1 n2) rest...'

    Node names are normalized: Cadence escaped-bus notation ``DOUT\\<9\\>``
    is converted to plain ``DOUT<9>``.
    """
    # Find the first '(' and matching ')'
    paren_start = line.index('(')
    paren_end = line.index(')', paren_start)

    name = line[:paren_start].strip()
    nodes_str = line[paren_start + 1:paren_end].strip()
    remainder = line[paren_end + 1:].strip()

    nodes = [_normalize_node_name(n) for n in nodes_str.split()]
    return name, nodes, remainder


def _parse_vsource(line: str, netlist: SpectreNetlist,
                   variables: Dict[str, float]):
    """Parse a voltage source: Vname (n+ n-) vsource type=X dc=Y ..."""
    name, nodes, remainder = _extract_nodes(line)

    if len(nodes) < 2:
        return

    node_pos = nodes[0]
    node_neg = nodes[1]

    # remainder: "vsource dc=0.9 type=dc"
    tokens = remainder.split()

    # Skip "vsource" keyword
    param_start = 0
    if tokens and tokens[0].lower() == 'vsource':
        param_start = 1

    # Handle bracket-delimited wave= for PWL
    # Rejoin remainder after vsource to handle wave=[...] properly
    param_str = ' '.join(tokens[param_start:])

    # Check for wave=[...] pattern
    wave_data = None
    wave_match = re.search(r'wave\s*=\s*\[([^\]]*)\]', param_str)
    if wave_match:
        wave_text = wave_match.group(1)
        wave_data = wave_text
        # Remove wave=[...] from param_str
        param_str = param_str[:wave_match.start()] + param_str[wave_match.end():]

    params = _parse_named_params(param_str.split(), 0, variables)

    if wave_data is not None:
        # Evaluate each wave token; parameter refs (e.g. 'vdd') use variables
        wave_vals = []
        for tok in wave_data.split():
            tok = tok.strip()
            if not tok:
                continue
            val = _parse_suffix_number(tok)
            if val is None:
                try:
                    val = evaluate_expr(tok, variables)
                except (ValueError, ZeroDivisionError):
                    val = None
            if val is not None:
                wave_vals.append(val)
        params['wave'] = wave_vals

    src = _build_source(name, node_pos, node_neg, params)
    netlist.sources.append(src)


def _parse_instance(line: str, netlist: SpectreNetlist,
                    variables: Dict[str, float]):
    """Parse an instance: Iname (n1 n2 ...) ModelName k=v ..."""
    try:
        name, nodes, remainder = _extract_nodes(line)
    except ValueError:
        return  # no parens found

    tokens = remainder.split()
    if not tokens:
        return

    # First non-param token after nodes is the model name
    model_name = None
    param_tokens = []

    for tok in tokens:
        if '=' in tok:
            param_tokens.append(tok)
        elif model_name is None:
            model_name = tok
        else:
            param_tokens.append(tok)

    if model_name is None:
        return

    # If model_name is "vsource", this is a voltage source, not an instance
    if model_name.lower() == 'vsource':
        return

    params = _parse_named_params(param_tokens, 0, variables)

    netlist.instances.append(SpectreInstance(
        name=name, nodes=nodes, model_name=model_name, params=params,
    ))


# ---------------------------------------------------------------------------
# Include parser
# ---------------------------------------------------------------------------

def _parse_include(line: str, netlist: SpectreNetlist):
    """Parse include "path" [section=X] -> SpectreInclude."""
    # Extract quoted path
    m = re.search(r'"([^"]+)"', line)
    if not m:
        m = re.search(r"'([^']+)'", line)
    if not m:
        return
    path = m.group(1)

    # Check for section=XXX
    section = None
    sm = re.search(r'section\s*=\s*(\S+)', line)
    if sm:
        section = sm.group(1)

    netlist.includes.append(SpectreInclude(path=path, section=section))


# ---------------------------------------------------------------------------
# Subckt block parser
# ---------------------------------------------------------------------------

def _parse_mosfet_line(line: str, variables: Dict[str, float]) -> Optional[SpectreMosfet]:
    """Parse a MOSFET line: M88 (d g s b) model_name k=v ... -> SpectreMosfet."""
    try:
        name, nodes, remainder = _extract_nodes(line)
    except ValueError:
        return None

    tokens = remainder.split()
    if not tokens:
        return None

    model = None
    param_tokens = []
    for tok in tokens:
        if '=' in tok:
            param_tokens.append(tok)
        elif model is None:
            model = tok
        else:
            param_tokens.append(tok)

    if model is None:
        return None

    params = _parse_named_params(param_tokens, 0, variables)
    return SpectreMosfet(name=name, nodes=nodes, model=model, params=params)


def _parse_subckt_block(lines: List[str], start_idx: int,
                        netlist: SpectreNetlist,
                        variables: Dict[str, float]) -> int:
    """Parse a subckt...ends block. Returns index of next line after ends."""
    header = lines[start_idx]
    # "subckt name port1 port2 ..."
    parts = header.split()
    subckt_name = parts[1] if len(parts) > 1 else 'unknown'
    ports = parts[2:] if len(parts) > 2 else []

    subckt = SpectreSubckt(name=subckt_name, ports=ports)
    idx = start_idx + 1

    while idx < len(lines):
        line = lines[idx]
        low = line.lower().strip()

        # End of subckt
        if low.startswith('ends'):
            idx += 1
            break

        # M-device line (starts with M or m, has parenthesized nodes)
        if line[0] in ('M', 'm') and '(' in line:
            mosfet = _parse_mosfet_line(line, variables)
            if mosfet:
                subckt.mosfets.append(mosfet)
                subckt.body_lines.append(line)
                idx += 1
                continue

        # Instance inside subckt (e.g. subckt calls)
        if '(' in line:
            try:
                iname, inodes, iremainder = _extract_nodes(line)
                itokens = iremainder.split()
                imodel = None
                iparam_tokens = []
                for tok in itokens:
                    if '=' in tok:
                        iparam_tokens.append(tok)
                    elif imodel is None:
                        imodel = tok
                if imodel and imodel.lower() != 'vsource':
                    iparams = _parse_named_params(iparam_tokens, 0, variables)
                    subckt.instances.append(SpectreInstance(
                        name=iname, nodes=inodes, model_name=imodel, params=iparams,
                    ))
            except ValueError:
                pass
            subckt.body_lines.append(line)
            idx += 1
            continue

        # Any other line inside subckt — store as raw body
        subckt.body_lines.append(line)
        idx += 1

    netlist.subckts.append(subckt)
    return idx


# ---------------------------------------------------------------------------
# Detection helper
# ---------------------------------------------------------------------------

def has_transistors(netlist: SpectreNetlist) -> bool:
    """Check if a netlist contains transistor-level devices (subckts with MOSFETs
    or top-level M-prefix instances)."""
    if netlist.subckts:
        return True
    # Check for top-level M-prefix instances
    for inst in netlist.instances:
        if inst.name.upper().startswith('M'):
            return True
    return False
