"""
parser.py — Recursive descent parser for Verilog-A → AST
"""
from typing import List, Optional, Tuple

from .ast_nodes import *
from .lexer import Token, TokenType, tokenize


class ParseError(Exception):
    def __init__(self, msg, token=None):
        loc = f" at L{token.line}:{token.col}" if token else ""
        super().__init__(f"Parse error{loc}: {msg}")
        self.token = token


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Token:
        return self.tokens[self.pos]

    def advance(self) -> Token:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def expect(self, tt: TokenType) -> Token:
        t = self.peek()
        if t.type != tt:
            raise ParseError(f"Expected {tt.name}, got {t.type.name} ({t.value!r})", t)
        return self.advance()

    def match(self, *types) -> Optional[Token]:
        if self.peek().type in types:
            return self.advance()
        return None

    def at(self, *types) -> bool:
        return self.peek().type in types

    # ─── Module ───

    def parse_module(self) -> Module:
        # Skip everything before 'module' keyword (nature/discipline defs from includes, attributes)
        while not self.at(TokenType.MODULE, TokenType.EOF):
            if self.at(TokenType.ATTR_BEGIN):
                self.advance()
                while not self.at(TokenType.ATTR_END, TokenType.EOF):
                    self.advance()
                self.match(TokenType.ATTR_END)
            else:
                self.advance()

        self.expect(TokenType.MODULE)
        name_tok = self.expect(TokenType.IDENT)
        name = name_tok.value

        # Parse port list
        ports = []
        if self.match(TokenType.LPAREN):
            ports = self._parse_port_list_header()
            self.expect(TokenType.RPAREN)
        self.expect(TokenType.SEMI)

        module = Module(name=name, ports=[p for p in ports if isinstance(p, str)])

        # If ANSI-style, ports may contain PortDecl objects
        ansi_decls = [p for p in ports if isinstance(p, PortDecl)]
        if ansi_decls:
            module.ports = [p.name for p in ansi_decls]
            module.port_decls = ansi_decls

        # Parse module items until endmodule
        while not self.at(TokenType.ENDMODULE, TokenType.EOF):
            self._parse_module_item(module)

        self.expect(TokenType.ENDMODULE)
        return module

    def _parse_port_list_header(self) -> List:
        """Parse port list in module header. Can be ANSI or non-ANSI style."""
        ports = []
        # Check if ANSI-style (has direction in port list)
        if self.at(TokenType.INPUT, TokenType.OUTPUT, TokenType.INOUT):
            return self._parse_ansi_port_list()

        # Non-ANSI: just identifier list
        while True:
            if self.at(TokenType.IDENT):
                ports.append(self.advance().value)
            elif self.at(TokenType.LBRACKET):
                # array port in header like [7:0] FRAME - skip for now
                self._skip_range()
                if self.at(TokenType.IDENT):
                    ports.append(self.advance().value)
            if not self.match(TokenType.COMMA):
                break
        return ports

    def _parse_ansi_port_list(self) -> List:
        """Parse ANSI-style port declarations in module header.

        Returns a list where each element is either:
        - a string (port name) — for non-ANSI compatibility
        - a PortDecl — for ANSI-style declarations with direction/array info
        """
        ports = []
        while not self.at(TokenType.RPAREN, TokenType.EOF):
            direction = None
            if self.match(TokenType.INPUT):
                direction = Direction.INPUT
            elif self.match(TokenType.OUTPUT):
                direction = Direction.OUTPUT
            elif self.match(TokenType.INOUT):
                direction = Direction.INOUT

            # Skip discipline
            discipline = 'electrical'
            if self.at(TokenType.ELECTRICAL, TokenType.VOLTAGE, TokenType.CURRENT):
                discipline = self.advance().value

            # Check for array
            array_hi, array_lo = None, None
            if self.at(TokenType.LBRACKET):
                array_hi, array_lo = self._parse_range()

            name = self.expect(TokenType.IDENT).value

            if direction:
                pd = PortDecl(name=name, direction=direction, discipline=discipline)
                if array_hi is not None:
                    pd.is_array = True
                    pd.array_hi = array_hi
                    pd.array_lo = array_lo
                ports.append(pd)
            else:
                ports.append(name)

            if not self.match(TokenType.COMMA):
                break
        return ports

    def _parse_range(self) -> Tuple[int, int]:
        """Parse [hi:lo] range."""
        self.expect(TokenType.LBRACKET)
        hi = self._parse_const_expr()
        self.expect(TokenType.COLON)
        lo = self._parse_const_expr()
        self.expect(TokenType.RBRACKET)
        return int(hi), int(lo)

    def _skip_range(self):
        """Skip over a [...] range."""
        if self.match(TokenType.LBRACKET):
            depth = 1
            while depth > 0 and not self.at(TokenType.EOF):
                if self.match(TokenType.LBRACKET):
                    depth += 1
                elif self.match(TokenType.RBRACKET):
                    depth -= 1
                else:
                    self.advance()

    def _parse_const_expr(self) -> float:
        """Parse a constant integer/float expression (for ranges, etc.)."""
        neg = False
        if self.match(TokenType.MINUS):
            neg = True
        if self.at(TokenType.NUMBER):
            val = float(self.advance().value)
            # Handle simple arithmetic
            while self.at(TokenType.PLUS, TokenType.MINUS, TokenType.STAR):
                op = self.advance()
                val2 = float(self.expect(TokenType.NUMBER).value)
                if op.type == TokenType.PLUS:
                    val += val2
                elif op.type == TokenType.MINUS:
                    val -= val2
                elif op.type == TokenType.STAR:
                    val *= val2
            return -val if neg else val
        elif self.at(TokenType.IDENT):
            # Could be a define reference — return as identifier
            self.advance()
            return 0  # fallback
        if self.at(TokenType.LPAREN):
            self.advance()
            val = self._parse_const_expr()
            self.expect(TokenType.RPAREN)
            return -val if neg else val
        return 0

    # ─── Module Items ───

    def _parse_module_item(self, module: Module):
        """Parse one module-level item."""
        tok = self.peek()

        # Direction declarations: input, output, inout
        if tok.type in (TokenType.INPUT, TokenType.OUTPUT, TokenType.INOUT):
            self._parse_port_direction_decl(module)
            return

        # Discipline declarations: electrical, voltage, current
        if tok.type in (TokenType.ELECTRICAL, TokenType.VOLTAGE, TokenType.CURRENT):
            self._parse_discipline_decl(module)
            return

        # Parameter declarations
        if tok.type == TokenType.PARAMETER:
            self._parse_parameter_decl(module)
            return

        # Variable declarations: real, integer, genvar
        if tok.type in (TokenType.REAL, TokenType.INTEGER, TokenType.GENVAR):
            self._parse_variable_decl(module)
            return

        # Analog block
        if tok.type == TokenType.ANALOG:
            self.advance()
            block = self._parse_block_or_statement()
            if isinstance(block, Block):
                module.analog_block = AnalogBlock(body=block)
            else:
                module.analog_block = AnalogBlock(body=Block(statements=[block]))
            return

        # Skip unknown tokens
        self.advance()

    def _parse_port_direction_decl(self, module: Module):
        """Parse: input/output/inout [discipline] [range] name [, name] ;"""
        direction_tok = self.advance()
        direction = {
            TokenType.INPUT: Direction.INPUT,
            TokenType.OUTPUT: Direction.OUTPUT,
            TokenType.INOUT: Direction.INOUT,
        }[direction_tok.type]

        # Optional discipline
        discipline = 'electrical'

        # Check for array range before discipline
        array_hi, array_lo = None, None
        if self.at(TokenType.LBRACKET):
            array_hi, array_lo = self._parse_range()

        # Parse port names
        while True:
            if self.at(TokenType.ELECTRICAL, TokenType.VOLTAGE, TokenType.CURRENT):
                discipline = self.advance().value
                # Check again for range after discipline
                if self.at(TokenType.LBRACKET) and array_hi is None:
                    array_hi, array_lo = self._parse_range()
            if self.at(TokenType.IDENT):
                name = self.advance().value
                pd = PortDecl(name=name, direction=direction, discipline=discipline)
                if array_hi is not None:
                    pd.is_array = True
                    pd.array_hi = array_hi
                    pd.array_lo = array_lo
                # Update or add
                found = False
                for i, existing in enumerate(module.port_decls):
                    if existing.name == name:
                        module.port_decls[i] = pd
                        found = True
                        break
                if not found:
                    module.port_decls.append(pd)
                    if name not in module.ports:
                        module.ports.append(name)
            if not self.match(TokenType.COMMA):
                break
        self.match(TokenType.SEMI)

    def _parse_discipline_decl(self, module: Module):
        """Parse: electrical name [, name] ;

        Also handles 1-D and 2-D array forms:
          electrical [lo:hi] name ;          ← 1-D inner range before name
          electrical [lo:hi] name [lo2:hi2] ; ← 2-D: inner range before, outer range after name
        """
        discipline = self.advance().value

        # Check for inner array range before the name(s)
        array_hi, array_lo = None, None
        if self.at(TokenType.LBRACKET):
            array_hi, array_lo = self._parse_range()

        while True:
            if self.at(TokenType.IDENT):
                name = self.advance().value
                # Optional outer dimension range after the name (2-D array)
                if self.at(TokenType.LBRACKET):
                    self._skip_range()  # consume [hi:lo] — recorded in array_hi/lo already
                # Update existing port decl or add new one
                found = False
                for pd in module.port_decls:
                    if pd.name == name:
                        pd.discipline = discipline
                        if array_hi is not None:
                            pd.is_array = True
                            pd.array_hi = array_hi
                            pd.array_lo = array_lo
                        found = True
                        break
                if not found:
                    pd = PortDecl(name=name, direction=Direction.INOUT,
                                  discipline=discipline)
                    if array_hi is not None:
                        pd.is_array = True
                        pd.array_hi = array_hi
                        pd.array_lo = array_lo
                    module.port_decls.append(pd)
            if not self.match(TokenType.COMMA):
                break
        self.match(TokenType.SEMI)

    def _parse_parameter_decl(self, module: Module):
        """Parse: parameter [real|integer] name = value [from range] ;"""
        self.expect(TokenType.PARAMETER)

        param_type = ParamType.REAL
        if self.match(TokenType.REAL):
            param_type = ParamType.REAL
        elif self.match(TokenType.INTEGER):
            param_type = ParamType.INTEGER

        name = self.expect(TokenType.IDENT).value

        default_val = NumberLiteral(0)
        if self.match(TokenType.ASSIGN):
            default_val = self._parse_expression()

        # Optional range: from [lo:hi) or from (lo:hi]
        range_lo = None
        range_hi = None
        range_lo_incl = True
        range_hi_incl = True
        if self.at(TokenType.IDENT) and self.peek().value == 'from':
            self.advance()
            if self.match(TokenType.LBRACKET):
                range_lo_incl = True
            elif self.match(TokenType.LPAREN):
                range_lo_incl = False
            range_lo = self._parse_expression()
            self.expect(TokenType.COLON)
            range_hi = self._parse_expression()
            if self.match(TokenType.RBRACKET):
                range_hi_incl = True
            elif self.match(TokenType.RPAREN):
                range_hi_incl = False

        self.match(TokenType.SEMI)

        # Detect string type
        if isinstance(default_val, StringLiteral):
            param_type = ParamType.STRING

        module.parameters.append(ParameterDecl(
            name=name, param_type=param_type, default_value=default_val,
            range_lo=range_lo, range_hi=range_hi,
            range_lo_inclusive=range_lo_incl, range_hi_inclusive=range_hi_incl,
        ))

    def _parse_variable_decl(self, module: Module):
        """Parse: real|integer|genvar name[range] [= init] [, name] ;"""
        type_tok = self.advance()
        var_type = ParamType.REAL if type_tok.type == TokenType.REAL else ParamType.INTEGER

        while True:
            name = self.expect(TokenType.IDENT).value
            is_array = False
            array_hi, array_lo = None, None
            init_values = None

            if self.at(TokenType.LBRACKET):
                array_hi, array_lo = self._parse_range()
                is_array = True

            if self.match(TokenType.ASSIGN):
                if self.at(TokenType.LBRACE):
                    # Array initializer: {val1, val2, ...}
                    self.advance()
                    init_values = []
                    while not self.at(TokenType.RBRACE, TokenType.EOF):
                        init_values.append(self._parse_expression())
                        if not self.match(TokenType.COMMA):
                            break
                    self.expect(TokenType.RBRACE)
                else:
                    init_values = [self._parse_expression()]

            vd = VariableDecl(name=name, var_type=var_type, is_array=is_array,
                              array_hi=array_hi, array_lo=array_lo,
                              init_values=init_values)
            module.variables.append(vd)

            if not self.match(TokenType.COMMA):
                break

        self.match(TokenType.SEMI)

    # ─── Statements ───

    def _parse_block_or_statement(self) -> Statement:
        """Parse a begin/end block or a single statement."""
        if self.at(TokenType.BEGIN):
            return self._parse_block()
        return self._parse_statement()

    def _parse_block(self) -> Block:
        self.expect(TokenType.BEGIN)
        stmts = []
        while not self.at(TokenType.END, TokenType.EOF):
            stmts.append(self._parse_statement())
        self.expect(TokenType.END)
        return Block(statements=stmts)

    def _parse_statement(self) -> Statement:
        """Parse a single statement."""
        tok = self.peek()

        # begin/end block
        if tok.type == TokenType.BEGIN:
            return self._parse_block()

        # Event control: @(...)
        if tok.type == TokenType.AT:
            return self._parse_event_statement()

        # If statement
        if tok.type == TokenType.IF:
            return self._parse_if_statement()

        # For loop
        if tok.type == TokenType.FOR:
            return self._parse_for_statement()

        # Case statement
        if tok.type == TokenType.CASE:
            return self._parse_case_statement()

        # System task: $strobe, $display
        if tok.type == TokenType.IDENT and tok.value.startswith('$'):
            return self._parse_system_task()

        # Expression-based: assignment, contribution, or expression statement
        return self._parse_expr_statement()

    def _parse_event_statement(self) -> EventStatement:
        """Parse @(event) statement"""
        self.expect(TokenType.AT)
        self.expect(TokenType.LPAREN)

        events = []
        events.append(self._parse_single_event())

        while self.match(TokenType.OR):
            events.append(self._parse_single_event())

        self.expect(TokenType.RPAREN)

        if len(events) == 1:
            event = events[0]
        else:
            event = CombinedEvent(events=events)

        body = self._parse_block_or_statement()
        return EventStatement(event=event, body=body)

    def _parse_single_event(self) -> EventExpr:
        """Parse: cross(expr, dir) | above(expr, dir) | initial_step | timer(period) | final_step"""
        tok = self.peek()

        if tok.type == TokenType.IDENT:
            if tok.value == 'cross':
                self.advance()
                self.expect(TokenType.LPAREN)
                expr = self._parse_expression()
                direction = None
                if self.match(TokenType.COMMA):
                    dir_expr = self._parse_expression()
                    direction = self._eval_const(dir_expr)
                self.expect(TokenType.RPAREN)
                return EventExpr(EventType.CROSS, [expr],
                                 direction=int(direction) if direction else None)

            elif tok.value == 'above':
                self.advance()
                self.expect(TokenType.LPAREN)
                expr = self._parse_expression()
                direction = None
                if self.match(TokenType.COMMA):
                    dir_expr = self._parse_expression()
                    direction = self._eval_const(dir_expr)
                self.expect(TokenType.RPAREN)
                return EventExpr(EventType.ABOVE, [expr],
                                 direction=int(direction) if direction else None)

            elif tok.value == 'initial_step':
                self.advance()
                return EventExpr(EventType.INITIAL_STEP)

            elif tok.value == 'timer':
                self.advance()
                self.expect(TokenType.LPAREN)
                period_expr = self._parse_expression()
                self.expect(TokenType.RPAREN)
                return EventExpr(EventType.TIMER, [period_expr])

            elif tok.value == 'final_step':
                self.advance()
                return EventExpr(EventType.FINAL_STEP)

        raise ParseError(f"Expected event expression, got {tok.value!r}", tok)

    def _parse_if_statement(self) -> IfStatement:
        self.expect(TokenType.IF)
        self.expect(TokenType.LPAREN)
        cond = self._parse_expression()
        self.expect(TokenType.RPAREN)
        then_body = self._parse_block_or_statement()
        else_body = None
        if self.match(TokenType.ELSE):
            else_body = self._parse_block_or_statement()
        return IfStatement(cond=cond, then_body=then_body, else_body=else_body)

    def _parse_for_statement(self) -> ForStatement:
        self.expect(TokenType.FOR)
        self.expect(TokenType.LPAREN)
        init = self._parse_simple_assignment()
        self.expect(TokenType.SEMI)
        cond = self._parse_expression()
        self.expect(TokenType.SEMI)
        update = self._parse_simple_assignment()
        self.expect(TokenType.RPAREN)
        body = self._parse_block_or_statement()
        return ForStatement(init=init, cond=cond, update=update, body=body)

    def _parse_case_statement(self) -> CaseStatement:
        """Parse: case (expr) value: stmt ... default: stmt endcase"""
        self.expect(TokenType.CASE)
        self.expect(TokenType.LPAREN)
        sel_expr = self._parse_expression()
        self.expect(TokenType.RPAREN)

        items = []
        while not self.at(TokenType.ENDCASE, TokenType.EOF):
            # Check for 'default'
            if self.at(TokenType.IDENT) and self.peek().value == 'default':
                self.advance()
                self.expect(TokenType.COLON)
                body = self._parse_block_or_statement()
                items.append(CaseItem(values=[], body=body))
            else:
                # Parse comma-separated value expressions before ':'
                values = [self._parse_expression()]
                while self.match(TokenType.COMMA):
                    values.append(self._parse_expression())
                self.expect(TokenType.COLON)
                body = self._parse_block_or_statement()
                items.append(CaseItem(values=values, body=body))
        self.expect(TokenType.ENDCASE)
        return CaseStatement(expr=sel_expr, items=items)

    def _parse_simple_assignment(self) -> Assignment:
        """Parse: target = expr"""
        target = self._parse_expression()
        self.expect(TokenType.ASSIGN)
        value = self._parse_expression()
        return Assignment(target=target, value=value)

    def _parse_system_task(self) -> SystemTask:
        name = self.advance().value
        args = []
        if self.match(TokenType.LPAREN):
            while not self.at(TokenType.RPAREN, TokenType.EOF):
                args.append(self._parse_expression())
                if not self.match(TokenType.COMMA):
                    break
            self.expect(TokenType.RPAREN)
        self.match(TokenType.SEMI)
        return SystemTask(name=name, args=args)

    def _parse_expr_statement(self) -> Statement:
        """Parse assignment or contribution statement."""
        expr = self._parse_expression()

        # Contribution: V(a,b) <+ expr
        if self.match(TokenType.CONTRIB):
            rhs = self._parse_expression()
            self.match(TokenType.SEMI)
            if isinstance(expr, BranchAccess):
                return Contribution(branch=expr, expr=rhs)
            raise ParseError("Left side of <+ must be a branch access (V/I)")

        # Assignment: ident = expr
        if self.match(TokenType.ASSIGN):
            rhs = self._parse_expression()
            self.match(TokenType.SEMI)
            return Assignment(target=expr, value=rhs)

        self.match(TokenType.SEMI)
        # Expression statement (rare but possible)
        return Assignment(target=expr, value=NumberLiteral(0))

    # ─── Expressions (precedence climbing) ───

    def _parse_expression(self) -> Expr:
        return self._parse_ternary()

    def _parse_ternary(self) -> Expr:
        expr = self._parse_lor()
        if self.match(TokenType.QUESTION):
            true_expr = self._parse_expression()
            self.expect(TokenType.COLON)
            false_expr = self._parse_expression()
            return TernaryExpr(cond=expr, true_expr=true_expr, false_expr=false_expr)
        return expr

    def _parse_lor(self) -> Expr:
        left = self._parse_land()
        while self.match(TokenType.LOR):
            right = self._parse_land()
            left = BinaryExpr('||', left, right)
        return left

    def _parse_land(self) -> Expr:
        left = self._parse_bitor()
        while self.match(TokenType.LAND):
            right = self._parse_bitor()
            left = BinaryExpr('&&', left, right)
        return left

    def _parse_bitor(self) -> Expr:
        left = self._parse_bitxor()
        while self.match(TokenType.PIPE):
            right = self._parse_bitxor()
            left = BinaryExpr('|', left, right)
        return left

    def _parse_bitxor(self) -> Expr:
        left = self._parse_bitand()
        while self.match(TokenType.CARET):
            right = self._parse_bitand()
            left = BinaryExpr('^', left, right)
        return left

    def _parse_bitand(self) -> Expr:
        left = self._parse_equality()
        while self.match(TokenType.AMP):
            right = self._parse_equality()
            left = BinaryExpr('&', left, right)
        return left

    def _parse_equality(self) -> Expr:
        left = self._parse_relational()
        while True:
            if self.match(TokenType.EQ):
                right = self._parse_relational()
                left = BinaryExpr('==', left, right)
            elif self.match(TokenType.NE):
                right = self._parse_relational()
                left = BinaryExpr('!=', left, right)
            else:
                break
        return left

    def _parse_relational(self) -> Expr:
        left = self._parse_shift()
        while True:
            if self.match(TokenType.GT):
                right = self._parse_shift()
                left = BinaryExpr('>', left, right)
            elif self.match(TokenType.LT):
                right = self._parse_shift()
                left = BinaryExpr('<', left, right)
            elif self.match(TokenType.GE):
                right = self._parse_shift()
                left = BinaryExpr('>=', left, right)
            elif self.match(TokenType.LE):
                right = self._parse_shift()
                left = BinaryExpr('<=', left, right)
            else:
                break
        return left

    def _parse_shift(self) -> Expr:
        left = self._parse_additive()
        while True:
            if self.match(TokenType.LSHIFT):
                right = self._parse_additive()
                left = BinaryExpr('<<', left, right)
            elif self.match(TokenType.RSHIFT):
                right = self._parse_additive()
                left = BinaryExpr('>>', left, right)
            else:
                break
        return left

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while True:
            if self.match(TokenType.PLUS):
                right = self._parse_multiplicative()
                left = BinaryExpr('+', left, right)
            elif self.match(TokenType.MINUS):
                right = self._parse_multiplicative()
                left = BinaryExpr('-', left, right)
            else:
                break
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_unary()
        while True:
            if self.match(TokenType.STAR):
                right = self._parse_unary()
                left = BinaryExpr('*', left, right)
            elif self.match(TokenType.SLASH):
                right = self._parse_unary()
                left = BinaryExpr('/', left, right)
            elif self.match(TokenType.PERCENT):
                right = self._parse_unary()
                left = BinaryExpr('%', left, right)
            else:
                break
        return left

    def _parse_unary(self) -> Expr:
        if self.match(TokenType.MINUS):
            operand = self._parse_unary()
            # Optimize: -number → NumberLiteral(-value)
            if isinstance(operand, NumberLiteral):
                return NumberLiteral(-operand.value)
            return UnaryExpr('-', operand)
        if self.match(TokenType.BANG):
            operand = self._parse_unary()
            return UnaryExpr('!', operand)
        if self.match(TokenType.TILDE):
            operand = self._parse_unary()
            return UnaryExpr('~', operand)
        if self.match(TokenType.PLUS):
            return self._parse_unary()
        return self._parse_primary()

    def _parse_primary(self) -> Expr:
        tok = self.peek()

        # Number literal
        if tok.type == TokenType.NUMBER:
            self.advance()
            return NumberLiteral(float(tok.value))

        # String literal
        if tok.type == TokenType.STRING:
            self.advance()
            return StringLiteral(tok.value)

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self.advance()
            expr = self._parse_expression()
            self.expect(TokenType.RPAREN)
            return expr

        # V(...) or I(...) branch access
        if tok.type == TokenType.IDENT and tok.value in ('V', 'I'):
            access_type = tok.value
            self.advance()
            self.expect(TokenType.LPAREN)
            name1, idx1, idx1_2 = self._parse_node_ref()
            name2, idx2, idx2_2 = None, None, None
            if self.match(TokenType.COMMA):
                name2, idx2, idx2_2 = self._parse_node_ref()
            self.expect(TokenType.RPAREN)
            return BranchAccess(access_type=access_type, node1=name1, node2=name2,
                                node1_index=idx1, node2_index=idx2,
                                node1_index2=idx1_2, node2_index2=idx2_2)

        # Identifier (variable, parameter, function call)
        if tok.type == TokenType.IDENT:
            name = self.advance().value

            # Method call: name.method(args)
            if self.at(TokenType.DOT):
                self.advance()
                method = self.expect(TokenType.IDENT).value
                self.expect(TokenType.LPAREN)
                args = self._parse_arg_list()
                self.expect(TokenType.RPAREN)
                return MethodCall(obj=name, method=method, args=args)

            # Function call: name(args)
            if self.at(TokenType.LPAREN):
                self.advance()
                args = self._parse_arg_list()
                self.expect(TokenType.RPAREN)
                return FunctionCall(name=name, args=args)

            # Array access: name[index]
            if self.at(TokenType.LBRACKET):
                self.advance()
                index = self._parse_expression()
                self.expect(TokenType.RBRACKET)
                return ArrayAccess(name=name, index=index)

            return Identifier(name)

        # Inf keyword (used in parameter ranges)
        if tok.type == TokenType.IDENT and tok.value == 'inf':
            self.advance()
            return NumberLiteral(float('inf'))

        # Fallback
        self.advance()
        return NumberLiteral(0)

    def _parse_node_ref(self):
        """Parse a node reference which may be 1-D or 2-D array-indexed.

        Returns (name, index_expr, index_expr2) where either index may be None.
        Supports:
          name            → (name, None, None)
          name[i]         → (name, i_expr, None)
          name[i][j]      → (name, i_expr, j_expr)
        """
        name = self.expect(TokenType.IDENT).value
        index_expr = None
        index_expr2 = None
        if self.at(TokenType.LBRACKET):
            self.advance()
            index_expr = self._parse_expression()
            self.expect(TokenType.RBRACKET)
            # Optional second dimension: array[i][j]
            if self.at(TokenType.LBRACKET):
                self.advance()
                index_expr2 = self._parse_expression()
                self.expect(TokenType.RBRACKET)
        return name, index_expr, index_expr2

    def _parse_arg_list(self) -> List[Expr]:
        args = []
        if self.at(TokenType.RPAREN):
            return args
        args.append(self._parse_expression())
        while self.match(TokenType.COMMA):
            args.append(self._parse_expression())
        return args

    def _eval_const(self, expr: Expr) -> float:
        """Try to evaluate a constant expression."""
        if isinstance(expr, NumberLiteral):
            return expr.value
        if isinstance(expr, UnaryExpr) and expr.op == '-':
            return -self._eval_const(expr.operand)
        return 0


def parse(source: str) -> Module:
    """Parse preprocessed Verilog-A source into a Module AST."""
    tokens = tokenize(source)
    parser = Parser(tokens)
    return parser.parse_module()
