'use strict';

const { T } = require('./tokenizer');
const AST = require('./ast');

/**
 * Parse a token stream into a FlatPPL AST.
 * Returns { ast: Program, diagnostics: Diagnostic[] }.
 */
function parse(tokens) {
  const diagnostics = [];
  let pos = 0;

  function peek() { return tokens[pos]; }
  function at(type) { return tokens[pos].type === type; }
  function atValue(type, val) { return tokens[pos].type === type && tokens[pos].value === val; }
  function advance() { return tokens[pos++]; }
  function expect(type) {
    if (at(type)) return advance();
    const tok = peek();
    diagnostics.push({
      severity: 'error',
      message: `Expected ${type}, got ${tok.type} '${tok.value}'`,
      loc: tok.loc,
    });
    return null;
  }

  function skipNewlines() {
    while (at(T.NEWLINE) || at(T.COMMENT)) {
      if (at(T.COMMENT)) advance(); // skip comments in body for now
      else advance();
    }
  }

  function skipToNewline() {
    while (!at(T.NEWLINE) && !at(T.EOF)) advance();
  }

  // --- Expression parsing (precedence climbing) ---

  function parseExpr() {
    return parseComparison();
  }

  function parseComparison() {
    let left = parseAddition();
    const ops = [T.EQEQ, T.NEQ, T.LT, T.GT, T.LTE, T.GTE];
    if (ops.includes(peek().type) || atValue(T.IDENT, 'in')) {
      const opTok = advance();
      const right = parseAddition();
      left = AST.BinaryExpr(opTok.value, left, right,
        AST.loc(left.loc.start.line, left.loc.start.col, right.loc.end.line, right.loc.end.col));
    }
    return left;
  }

  function parseAddition() {
    let left = parseMultiplication();
    while (at(T.PLUS) || at(T.MINUS)) {
      const opTok = advance();
      const right = parseMultiplication();
      left = AST.BinaryExpr(opTok.value, left, right,
        AST.loc(left.loc.start.line, left.loc.start.col, right.loc.end.line, right.loc.end.col));
    }
    return left;
  }

  function parseMultiplication() {
    let left = parseUnary();
    while (at(T.STAR) || at(T.SLASH)) {
      const opTok = advance();
      const right = parseUnary();
      left = AST.BinaryExpr(opTok.value, left, right,
        AST.loc(left.loc.start.line, left.loc.start.col, right.loc.end.line, right.loc.end.col));
    }
    return left;
  }

  function parseUnary() {
    if (at(T.MINUS)) {
      const opTok = advance();
      const operand = parseUnary();
      return AST.UnaryExpr('-', operand,
        AST.loc(opTok.loc.start.line, opTok.loc.start.col, operand.loc.end.line, operand.loc.end.col));
    }
    return parsePostfix();
  }

  function parsePostfix() {
    let expr = parsePrimary();

    while (true) {
      if (at(T.LPAREN)) {
        // Function call: expr(args)
        advance(); // (
        const args = parseArgList();
        const rparen = expect(T.RPAREN);
        const endLoc = rparen ? rparen.loc : args.length > 0 ? args[args.length - 1].loc : expr.loc;
        expr = AST.CallExpr(expr, args,
          AST.loc(expr.loc.start.line, expr.loc.start.col, endLoc.end.line, endLoc.end.col));
      } else if (at(T.LBRACKET)) {
        // Index: expr[indices]
        advance(); // [
        const indices = parseIndexList();
        const rbracket = expect(T.RBRACKET);
        const endLoc = rbracket ? rbracket.loc : expr.loc;
        expr = AST.IndexExpr(expr, indices,
          AST.loc(expr.loc.start.line, expr.loc.start.col, endLoc.end.line, endLoc.end.col));
      } else if (at(T.DOT)) {
        // Field access: expr.field
        advance(); // .
        const fieldTok = expect(T.IDENT);
        if (fieldTok) {
          expr = AST.FieldAccess(expr, fieldTok.value,
            AST.loc(expr.loc.start.line, expr.loc.start.col, fieldTok.loc.end.line, fieldTok.loc.end.col));
        }
      } else {
        break;
      }
    }
    return expr;
  }

  function parsePrimary() {
    const tok = peek();

    // Number
    if (at(T.NUMBER)) {
      advance();
      return AST.NumberLiteral(parseFloat(tok.value), tok.value, tok.loc);
    }

    // String
    if (at(T.STRING)) {
      advance();
      return AST.StringLiteral(tok.value, tok.value, tok.loc);
    }

    // Placeholder _name_
    if (at(T.PLACEHOLDER)) {
      advance();
      return AST.Placeholder(tok.value, tok.loc);
    }

    // Hole _
    if (at(T.HOLE)) {
      advance();
      return AST.Hole(tok.loc);
    }

    // Identifier (may be bool/constant/set)
    if (at(T.IDENT)) {
      advance();
      const { isBoolLiteral, isConstant, isSet } = require('./builtins');
      if (isBoolLiteral(tok.value)) return AST.BoolLiteral(tok.value === 'true', tok.loc);
      if (isConstant(tok.value)) return AST.ConstantRef(tok.value, tok.loc);
      if (isSet(tok.value)) return AST.SetRef(tok.value, tok.loc);
      return AST.Identifier(tok.value, tok.loc);
    }

    // Parenthesized expression
    if (at(T.LPAREN)) {
      advance(); // (
      const expr = parseExpr();
      expect(T.RPAREN);
      return expr;
    }

    // Array literal [a, b, c]
    if (at(T.LBRACKET)) {
      const lbracket = advance(); // [
      const elements = [];
      if (!at(T.RBRACKET)) {
        elements.push(parseExpr());
        while (at(T.COMMA)) {
          advance();
          if (at(T.RBRACKET)) break; // trailing comma
          elements.push(parseExpr());
        }
      }
      const rbracket = expect(T.RBRACKET);
      const endLoc = rbracket ? rbracket.loc : lbracket.loc;
      return AST.ArrayLiteral(elements,
        AST.loc(lbracket.loc.start.line, lbracket.loc.start.col, endLoc.end.line, endLoc.end.col));
    }

    // Unexpected token
    const bad = advance();
    diagnostics.push({
      severity: 'error',
      message: `Unexpected token '${bad.value}' (${bad.type})`,
      loc: bad.loc,
    });
    return AST.Identifier('__error__', bad.loc);
  }

  function parseArgList() {
    const args = [];
    if (at(T.RPAREN)) return args;

    args.push(parseArg());
    while (at(T.COMMA)) {
      advance();
      if (at(T.RPAREN)) break; // trailing comma
      args.push(parseArg());
    }
    return args;
  }

  function parseArg() {
    // Check for keyword argument: IDENT = expr (but not ==)
    if (at(T.IDENT) && pos + 1 < tokens.length
        && tokens[pos + 1].type === T.EQUALS) {
      const nameTok = advance();
      advance(); // =
      const value = parseExpr();
      return AST.KeywordArg(nameTok.value, value,
        AST.loc(nameTok.loc.start.line, nameTok.loc.start.col, value.loc.end.line, value.loc.end.col));
    }
    return parseExpr();
  }

  function parseIndexList() {
    const indices = [];
    indices.push(parseIndexArg());
    while (at(T.COMMA)) {
      advance();
      if (at(T.RBRACKET)) break;
      indices.push(parseIndexArg());
    }
    return indices;
  }

  function parseIndexArg() {
    // : means all (slice)
    if (at(T.COLON)) {
      const tok = advance();
      return AST.SliceAll(tok.loc);
    }
    return parseExpr();
  }

  // --- Statement parsing ---

  function parseStatement() {
    const startTok = peek();

    // Must start with an identifier (LHS of assignment)
    if (!at(T.IDENT)) {
      diagnostics.push({
        severity: 'error',
        message: `Expected variable name, got '${startTok.value}'`,
        loc: startTok.loc,
      });
      skipToNewline();
      return AST.ErrorStatement(startTok.value, startTok.loc);
    }

    // Collect LHS names: name1, name2, ... = rhs
    const names = [];
    names.push(AST.Identifier(peek().value, peek().loc));
    advance();

    while (at(T.COMMA)) {
      advance(); // ,
      if (at(T.IDENT)) {
        names.push(AST.Identifier(peek().value, peek().loc));
        advance();
      } else {
        diagnostics.push({
          severity: 'error',
          message: `Expected variable name after ','`,
          loc: peek().loc,
        });
        break;
      }
    }

    // Expect =
    if (!expect(T.EQUALS)) {
      skipToNewline();
      return AST.ErrorStatement(names.map(n => n.name).join(', '), startTok.loc);
    }

    // Parse RHS expression
    const value = parseExpr();

    const endLoc = value.loc;
    return AST.AssignStatement(names, value,
      AST.loc(startTok.loc.start.line, startTok.loc.start.col, endLoc.end.line, endLoc.end.col));
  }

  // --- Program ---

  function parseProgram() {
    const body = [];
    const comments = [];

    skipNewlines();
    while (!at(T.EOF)) {
      if (at(T.COMMENT)) {
        const c = advance();
        comments.push(AST.Comment(c.value, c.loc));
        skipNewlines();
        continue;
      }

      const stmt = parseStatement();
      body.push(stmt);

      // Expect newline or EOF after statement
      if (!at(T.NEWLINE) && !at(T.EOF) && !at(T.COMMENT)) {
        diagnostics.push({
          severity: 'error',
          message: `Expected end of line, got '${peek().value}'`,
          loc: peek().loc,
        });
        skipToNewline();
      }
      skipNewlines();
    }

    return AST.Program(body, comments);
  }

  const ast = parseProgram();
  return { ast, diagnostics };
}

module.exports = { parse };
