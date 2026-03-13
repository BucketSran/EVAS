"""
ast_nodes.py — AST node definitions for Verilog-A
"""
from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum, auto


class Direction(Enum):
    INPUT = auto()
    OUTPUT = auto()
    INOUT = auto()


class ParamType(Enum):
    REAL = auto()
    INTEGER = auto()
    STRING = auto()


class EventType(Enum):
    CROSS = auto()
    ABOVE = auto()
    INITIAL_STEP = auto()


class BinOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
    MOD = '%'
    POW = '^'
    BITAND = '&'
    BITOR = '|'
    BITXOR = '^'
    LSHIFT = '<<'
    RSHIFT = '>>'
    GT = '>'
    LT = '<'
    GE = '>='
    LE = '<='
    EQ = '=='
    NE = '!='
    LAND = '&&'
    LOR = '||'


class UnaryOp(Enum):
    NEG = '-'
    NOT = '!'
    BITNOT = '~'


# --- Expression nodes ---

@dataclass
class NumberLiteral:
    value: float

@dataclass
class StringLiteral:
    value: str

@dataclass
class Identifier:
    name: str

@dataclass
class ArrayAccess:
    name: str
    index: 'Expr'

@dataclass
class BinaryExpr:
    op: str
    left: 'Expr'
    right: 'Expr'

@dataclass
class UnaryExpr:
    op: str
    operand: 'Expr'

@dataclass
class TernaryExpr:
    cond: 'Expr'
    true_expr: 'Expr'
    false_expr: 'Expr'

@dataclass
class FunctionCall:
    name: str
    args: List['Expr']

@dataclass
class BranchAccess:
    """V(a,b), I(a,b), V(a)"""
    access_type: str  # 'V' or 'I'
    node1: str
    node2: Optional[str] = None
    node1_index: Optional['Expr'] = None  # dynamic array index for node1
    node2_index: Optional['Expr'] = None  # dynamic array index for node2

@dataclass
class MethodCall:
    """e.g. conf.substr(i, i)"""
    obj: str
    method: str
    args: List['Expr']

Expr = Union[NumberLiteral, StringLiteral, Identifier, ArrayAccess,
             BinaryExpr, UnaryExpr, TernaryExpr, FunctionCall,
             BranchAccess, MethodCall]


# --- Statement nodes ---

@dataclass
class Assignment:
    target: Expr  # Identifier or ArrayAccess
    value: Expr

@dataclass
class Contribution:
    """V(a,b) <+ expr"""
    branch: BranchAccess
    expr: Expr

@dataclass
class EventExpr:
    event_type: EventType
    args: List[Expr] = field(default_factory=list)
    direction: Optional[int] = None  # +1, -1, or None

@dataclass
class CombinedEvent:
    """event1 or event2"""
    events: List[EventExpr]

@dataclass
class EventStatement:
    event: Union[EventExpr, CombinedEvent]
    body: 'Statement'

@dataclass
class Block:
    statements: List['Statement']

@dataclass
class IfStatement:
    cond: Expr
    then_body: 'Statement'
    else_body: Optional['Statement'] = None

@dataclass
class ForStatement:
    init: Assignment
    cond: Expr
    update: Assignment
    body: 'Statement'

@dataclass
class SystemTask:
    """$strobe, $display, etc."""
    name: str
    args: List[Expr]

Statement = Union[Assignment, Contribution, EventStatement, Block,
                  IfStatement, ForStatement, SystemTask]


# --- Declaration nodes ---

@dataclass
class PortDecl:
    name: str
    direction: Direction
    discipline: str = 'electrical'
    is_array: bool = False
    array_hi: Optional[int] = None
    array_lo: Optional[int] = None

@dataclass
class ParameterDecl:
    name: str
    param_type: ParamType
    default_value: Expr
    range_lo: Optional[Expr] = None
    range_hi: Optional[Expr] = None
    range_lo_inclusive: bool = True
    range_hi_inclusive: bool = True

@dataclass
class VariableDecl:
    name: str
    var_type: ParamType  # REAL or INTEGER
    is_array: bool = False
    array_hi: Optional[int] = None
    array_lo: Optional[int] = None
    init_values: Optional[List[Expr]] = None

@dataclass
class AnalogBlock:
    body: Block

@dataclass
class Module:
    name: str
    ports: List[str]
    port_decls: List[PortDecl] = field(default_factory=list)
    parameters: List[ParameterDecl] = field(default_factory=list)
    variables: List[VariableDecl] = field(default_factory=list)
    analog_block: Optional[AnalogBlock] = None
    default_transition: Optional[float] = None
    defines: dict = field(default_factory=dict)
