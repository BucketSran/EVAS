"""
lexer.py — Tokenizer for Verilog-A
"""
import re
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum, auto


class TokenType(Enum):
    # Keywords
    MODULE = auto()
    ENDMODULE = auto()
    ANALOG = auto()
    BEGIN = auto()
    END = auto()
    PARAMETER = auto()
    REAL = auto()
    INTEGER = auto()
    GENVAR = auto()
    ELECTRICAL = auto()
    VOLTAGE = auto()
    CURRENT = auto()
    INPUT = auto()
    OUTPUT = auto()
    INOUT = auto()
    IF = auto()
    ELSE = auto()
    FOR = auto()
    OR = auto()

    # Literals
    NUMBER = auto()
    STRING = auto()
    IDENT = auto()

    # Operators
    CONTRIB = auto()     # <+
    PLUS = auto()        # +
    MINUS = auto()       # -
    STAR = auto()        # *
    SLASH = auto()       # /
    PERCENT = auto()     # %
    CARET = auto()       # ^  (xor in context, or pow)
    AMP = auto()         # &
    PIPE = auto()        # |
    TILDE = auto()       # ~
    BANG = auto()         # !
    LSHIFT = auto()      # <<
    RSHIFT = auto()      # >>
    GT = auto()          # >
    LT = auto()          # <
    GE = auto()          # >=
    LE = auto()          # <=
    EQ = auto()          # ==
    NE = auto()          # !=
    LAND = auto()        # &&
    LOR = auto()         # ||
    ASSIGN = auto()      # =
    QUESTION = auto()    # ?
    COLON = auto()       # :
    DOT = auto()         # .

    # Delimiters
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    LBRACE = auto()      # {
    RBRACE = auto()      # }
    SEMI = auto()        # ;
    COMMA = auto()       # ,
    AT = auto()          # @

    # Special
    EOF = auto()
    NEWLINE = auto()

    # Attributes
    ATTR_BEGIN = auto()  # (*
    ATTR_END = auto()    # *)


KEYWORDS = {
    'module': TokenType.MODULE,
    'endmodule': TokenType.ENDMODULE,
    'analog': TokenType.ANALOG,
    'begin': TokenType.BEGIN,
    'end': TokenType.END,
    'parameter': TokenType.PARAMETER,
    'real': TokenType.REAL,
    'integer': TokenType.INTEGER,
    'genvar': TokenType.GENVAR,
    'electrical': TokenType.ELECTRICAL,
    'voltage': TokenType.VOLTAGE,
    'current': TokenType.CURRENT,
    'input': TokenType.INPUT,
    'output': TokenType.OUTPUT,
    'inout': TokenType.INOUT,
    'if': TokenType.IF,
    'else': TokenType.ELSE,
    'for': TokenType.FOR,
    'or': TokenType.OR,
}

# SI suffixes for numbers
SI_SUFFIXES = {
    'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3, 'K': 1e3,
    'm': 1e-3, 'u': 1e-6, 'n': 1e-9, 'p': 1e-12, 'f': 1e-15, 'a': 1e-18,
}


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


class LexerError(Exception):
    def __init__(self, msg, line, col):
        super().__init__(f"Lexer error at L{line}:{col}: {msg}")
        self.line = line
        self.col = col


def tokenize(source: str) -> List[Token]:
    """Tokenize preprocessed Verilog-A source code."""
    tokens = []
    i = 0
    line = 1
    col = 1
    n = len(source)

    while i < n:
        ch = source[i]

        # Skip whitespace (but not newlines for tracking)
        if ch in ' \t\r':
            i += 1
            col += 1
            continue

        if ch == '\n':
            line += 1
            col = 1
            i += 1
            continue

        # Skip single-line comments
        if ch == '/' and i + 1 < n and source[i + 1] == '/':
            while i < n and source[i] != '\n':
                i += 1
            continue

        # Skip block comments
        if ch == '/' and i + 1 < n and source[i + 1] == '*':
            i += 2
            col += 2
            while i + 1 < n and not (source[i] == '*' and source[i + 1] == '/'):
                if source[i] == '\n':
                    line += 1
                    col = 1
                else:
                    col += 1
                i += 1
            i += 2
            col += 2
            continue

        # Attribute begin (*
        if ch == '(' and i + 1 < n and source[i + 1] == '*':
            tokens.append(Token(TokenType.ATTR_BEGIN, '(*', line, col))
            i += 2
            col += 2
            continue

        # Attribute end *)
        if ch == '*' and i + 1 < n and source[i + 1] == ')':
            tokens.append(Token(TokenType.ATTR_END, '*)', line, col))
            i += 2
            col += 2
            continue

        # Strings
        if ch == '"':
            start = i
            start_col = col
            i += 1
            col += 1
            s = ''
            while i < n and source[i] != '"':
                if source[i] == '\\' and i + 1 < n:
                    s += source[i + 1]
                    i += 2
                    col += 2
                else:
                    s += source[i]
                    i += 1
                    col += 1
            if i < n:
                i += 1
                col += 1
            tokens.append(Token(TokenType.STRING, s, line, start_col))
            continue

        # Numbers
        if ch.isdigit() or (ch == '.' and i + 1 < n and source[i + 1].isdigit()):
            start = i
            start_col = col
            # Match number: optional leading digits, optional dot+digits, optional exponent
            while i < n and (source[i].isdigit() or source[i] == '.'):
                i += 1
                col += 1
            # Check for exponent
            if i < n and source[i] in 'eE':
                i += 1
                col += 1
                if i < n and source[i] in '+-':
                    i += 1
                    col += 1
                while i < n and source[i].isdigit():
                    i += 1
                    col += 1
            num_str = source[start:i]
            # Check for SI suffix
            multiplier = 1.0
            if i < n and source[i] in SI_SUFFIXES and not (i + 1 < n and source[i + 1].isalpha() and source[i + 1] != 'e'):
                # Check it's actually a suffix (not part of an identifier)
                suffix_ch = source[i]
                # Look ahead to see if this is really a suffix
                if i + 1 >= n or not source[i + 1].isalnum():
                    multiplier = SI_SUFFIXES[suffix_ch]
                    i += 1
                    col += 1
                elif suffix_ch in ('p', 'n', 'u', 'f', 'a', 'k', 'K', 'T', 'G', 'M', 'm'):
                    # Allow suffixes even before non-alpha chars
                    if i + 1 >= n or not source[i + 1].isalpha():
                        multiplier = SI_SUFFIXES[suffix_ch]
                        i += 1
                        col += 1

            try:
                val = float(num_str) * multiplier
            except ValueError:
                val = 0.0
            tokens.append(Token(TokenType.NUMBER, str(val), line, start_col))
            continue

        # Identifiers / Keywords / System tasks
        if ch.isalpha() or ch == '_' or ch == '$':
            start = i
            start_col = col
            i += 1
            col += 1
            while i < n and (source[i].isalnum() or source[i] == '_'):
                i += 1
                col += 1
            word = source[start:i]
            tt = KEYWORDS.get(word, TokenType.IDENT)
            tokens.append(Token(tt, word, line, start_col))
            continue

        # Two-character operators
        if i + 1 < n:
            two = source[i:i + 2]
            tt = None
            if two == '<+':
                tt = TokenType.CONTRIB
            elif two == '<<':
                tt = TokenType.LSHIFT
            elif two == '>>':
                tt = TokenType.RSHIFT
            elif two == '>=':
                tt = TokenType.GE
            elif two == '<=':
                tt = TokenType.LE
            elif two == '==':
                tt = TokenType.EQ
            elif two == '!=':
                tt = TokenType.NE
            elif two == '&&':
                tt = TokenType.LAND
            elif two == '||':
                tt = TokenType.LOR
            if tt is not None:
                tokens.append(Token(tt, two, line, col))
                i += 2
                col += 2
                continue

        # Single-character operators/delimiters
        SINGLE = {
            '+': TokenType.PLUS, '-': TokenType.MINUS,
            '*': TokenType.STAR, '/': TokenType.SLASH,
            '%': TokenType.PERCENT, '^': TokenType.CARET,
            '&': TokenType.AMP, '|': TokenType.PIPE,
            '~': TokenType.TILDE, '!': TokenType.BANG,
            '>': TokenType.GT, '<': TokenType.LT,
            '=': TokenType.ASSIGN, '?': TokenType.QUESTION,
            ':': TokenType.COLON, '.': TokenType.DOT,
            '(': TokenType.LPAREN, ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET, ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE, '}': TokenType.RBRACE,
            ';': TokenType.SEMI, ',': TokenType.COMMA,
            '@': TokenType.AT,
        }
        if ch in SINGLE:
            tokens.append(Token(SINGLE[ch], ch, line, col))
            i += 1
            col += 1
            continue

        # Skip backtick directives that survived preprocessing
        if ch == '`':
            start = i
            i += 1
            col += 1
            while i < n and (source[i].isalnum() or source[i] == '_'):
                i += 1
                col += 1
            # Silently skip unknown directives
            continue

        # Unknown character — skip
        i += 1
        col += 1

    tokens.append(Token(TokenType.EOF, '', line, col))
    return tokens
