'use strict';

const { T } = require('./tokenizer.ts');
const AST = require('./ast.ts');

/**
 * Parse a token stream into a FlatPPL AST.
 * Returns { ast: Program, diagnostics: Diagnostic[] }.
 *
 * `variant` is the surface-syntax variant (see ./variants). The base
 * parser is variant-agnostic; per-variant grammar branches land in
 * later commits in this series. `variant` is accepted (and ignored)
 * starting now so call sites can pass it without churn.
 */
function parse(tokens: any[], variant: any) {
  // Fall back to canonical FlatPPL when no variant supplied — keeps
  // existing call sites that pass only `tokens` working.
  const v = variant || require('./variants.ts').FLATPPL;

  const diagnostics: any[] = [];
  let pos = 0;

  // Per spec §04: "Each occurrence of `_` on the left-hand side
  // (standalone or inside a decomposition) lowers to a distinct
  // auto-generated private name." We emit `__discard_N` for each
  // such occurrence so downstream passes see uniquely-named
  // bindings, preserving the discard semantics without tripping
  // the analyzer's duplicate-name check.
  let discardCounter = 0;
  function freshDiscardName() {
    return '__discard_' + (discardCounter++);
  }

  function peek() { return tokens[pos]; }
  function at(type: string) { return tokens[pos].type === type; }
  function atValue(type: string, val: any) { return tokens[pos].type === type && tokens[pos].value === val; }
  function advance() { return tokens[pos++]; }
  function expect(type: string) {
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

  // Skip newlines + plain comments while collecting any leading
  // doc-comments (the next binding's leading-attachment candidate).
  // Multiple leading doc-comments on the same binding are an error
  // (spec §04 §sec:documentation attachment rule "one per binding").
  function skipNewlinesCollectingDocs(): any | null {
    let leadingDoc: any | null = null;
    let leadingDocLoc: any | null = null;
    while (at(T.NEWLINE) || at(T.COMMENT)
           || at(T.DOC_LINE) || at(T.DOC_BLOCK)) {
      if (at(T.COMMENT) || at(T.NEWLINE)) {
        advance();
        continue;
      }
      // DOC_LINE / DOC_BLOCK — consume and remember as leading.
      const tok = peek();
      const isBlock = tok.type === T.DOC_BLOCK;
      const payload = tok.value;
      advance();
      if (leadingDoc) {
        diagnostics.push({
          severity: 'error',
          message: 'Only one doc-comment may precede a binding '
            + '(use a `%%%` block for multi-line content)',
          loc: tok.loc,
        });
        // Discard the new one; keep the first.
        continue;
      }
      leadingDoc = isBlock
        ? { markup: payload.markup, lines: payload.lines.slice() }
        : { markup: payload.markup, lines: [payload.content] };
      leadingDocLoc = tok.loc;
    }
    return leadingDoc;
  }

  function skipToNewline() {
    while (!at(T.NEWLINE) && !at(T.EOF)) advance();
  }

  // --- Expression parsing (precedence climbing) ---
  //
  // Grammar (spec §05), variant-parametric:
  //
  //   Expression  ::= Or
  //   Or          ::= And  (Or-sym  And)*
  //   And         ::= Cmp* (And-sym Cmp*)*       (Cmp* = Not for FlatPPY)
  //   Comparison  ::= Additive (CompOp Additive)?
  //   Additive    ::= Multiplicative ((+|-) Multiplicative)*
  //   Multiplicative ::= Unary ((*|/) Unary)*
  //   Unary       ::= ('-' | '!') Unary | Exponential   (! only when v.logicalSyms.not === '!')
  //   Exponential ::= Postfix ('^' Unary)?              (only when v.exponentOp)
  //   Postfix     ::= Primary (FieldAccess | Indexing | Call)*
  //
  // FlatPPY-only twist: `not` is a keyword whose precedence sits
  // above Comparison (so `not a < b` ≡ `not (a < b)`, per Python).
  // Logical ops lower to land / lor / lnot calls regardless of
  // variant.
  function parseExpr(): any {
    // Lambda sits at the top of Expression (lowest precedence) per spec
    // §05 Lambda syntax. Two surface forms:
    //   • Single-arg, bare: `name -> expr`            (no parens)
    //   • Multi-arg, parenthesised: `(n1, n2, ...) -> expr` (≥ 2 names)
    // The single-arg parenthesised form `(name) -> expr` is explicitly
    // NOT valid per spec §05 — paren-form lambdas require two or more
    // names. We peek without consuming; if neither pattern matches we
    // fall through to parseOr unchanged.
    if (at(T.IDENT) && tokens[pos + 1] && tokens[pos + 1].type === T.ARROW) {
      return parseSingleArgLambda();
    }
    if (at(T.LPAREN) && lookaheadIsLambdaParams()) {
      return parseLambda();
    }
    return parseOr();
  }

  // Look ahead from a `(` to determine if it opens a multi-arg lambda
  // parameter list. The parens must contain TWO OR MORE bare Names per
  // spec §05 (single-arg paren form is rejected); commas separate
  // names; trailing comma not permitted; the closing `)` must be
  // immediately followed by `->`. Returns false on any deviation so a
  // non-lambda parse path can proceed normally.
  function lookaheadIsLambdaParams() {
    if (!at(T.LPAREN)) return false;
    let i = pos + 1;  // first token after `(`
    if (tokens[i] && tokens[i].type !== T.IDENT) return false;
    let nameCount = 0;
    while (i < tokens.length) {
      if (!tokens[i] || tokens[i].type !== T.IDENT) return false;
      i++;
      nameCount++;
      if (tokens[i] && tokens[i].type === T.COMMA) {
        i++;
        continue;
      }
      if (tokens[i] && tokens[i].type === T.RPAREN) {
        // Spec §05: paren-form lambdas require ≥ 2 names. A bare
        // `(name) -> ...` is not a lambda; it's a parenthesised
        // identifier followed by a stray `->` (a parse error).
        if (nameCount < 2) return false;
        return tokens[i + 1] != null && tokens[i + 1].type === T.ARROW;
      }
      return false;
    }
    return false;
  }

  // Single-arg lambda: `name -> expr`. By the time we get here, the
  // top-level lookahead has already confirmed the next two tokens are
  // IDENT then ARROW. We delegate the heavy lifting (capture-avoiding
  // arg→placeholder rewrite, kwarg synthesis) to parseLambda by
  // building the parameter list inline.
  function parseSingleArgLambda(): any {
    const t = advance();  // IDENT
    advance();            // ARROW
    const body = parseExpr();
    const llocEnd = body.loc;
    const lloc = AST.loc(t.loc.start.line, t.loc.start.col,
                         llocEnd.end.line, llocEnd.end.col);
    const argSet = new Set([t.value]);
    const rewritten = rewriteFreeIdsToPlaceholders(body, argSet);
    return AST.CallExpr(
      AST.Identifier('functionof', lloc),
      [rewritten, AST.KeywordArg(t.value, AST.Placeholder(t.value, t.loc), t.loc)],
      lloc);
  }

  // Parse `(arg1, arg2, ...) -> expr`. By the time we get here,
  // lookaheadIsLambdaParams has confirmed the shape. We desugar at
  // parse time to `functionof(body', arg1 = _arg1_, ...)` where
  // body' is `expr` with every free occurrence of each `arg_i`
  // rewritten to Placeholder('arg_i'). The functionof+Placeholder
  // path is what `_lowerReification` already understands.
  function parseLambda(): any {
    const lparen = advance();  // (
    const argNames: string[] = [];
    const argLocs: any[] = [];
    while (at(T.IDENT)) {
      const t = advance();
      argNames.push(t.value);
      argLocs.push(t.loc);
      if (at(T.COMMA)) advance();
    }
    expect(T.RPAREN);
    expect(T.ARROW);
    const body = parseExpr();  // right-associative: body extends as far right as possible
    const llocEnd = body.loc;
    const lloc = AST.loc(lparen.loc.start.line, lparen.loc.start.col,
                         llocEnd.end.line, llocEnd.end.col);

    // Capture-avoiding rewrite of free identifier refs in `body`.
    // Skip into binders that re-bind the same name (inner lambdas
    // already desugared to functionof, plus user-written
    // functionof/kernelof).
    const argSet = new Set(argNames);
    const rewritten = rewriteFreeIdsToPlaceholders(body, argSet);

    const kwargs: any[] = [];
    for (let i = 0; i < argNames.length; i++) {
      kwargs.push(AST.KeywordArg(
        argNames[i],
        AST.Placeholder(argNames[i], argLocs[i]),
        argLocs[i]));
    }
    return AST.CallExpr(
      AST.Identifier('functionof', lloc),
      [rewritten, ...kwargs],
      lloc);
  }

  // Replace every free `Identifier(name)` whose name is in `argSet`
  // with `Placeholder(name)`. Skip into nested binders:
  //   - functionof/kernelof: the kwarg LHS names are local-scope
  //     binders for the body (args[0]); kwarg VALUES are outer-scope
  //     refs so we still rewrite them. The body shrinks the argSet
  //     by any kwarg name it shadows.
  //   - fn: doesn't bind names (only `_` holes), so we descend normally.
  function rewriteFreeIdsToPlaceholders(node: any, argSet: Set<string>): any {
    if (node == null || typeof node !== 'object') return node;
    if (Array.isArray(node)) {
      return node.map((x: any) => rewriteFreeIdsToPlaceholders(x, argSet));
    }
    if (node.type === 'Identifier' && argSet.has(node.name)) {
      return AST.Placeholder(node.name, node.loc);
    }
    if (node.type === 'CallExpr' && node.callee
        && node.callee.type === 'Identifier'
        && (node.callee.name === 'functionof' || node.callee.name === 'kernelof')
        && node.args && node.args.length > 0) {
      // Collect shadowing names from kwarg LHS (positions 1..n).
      const shadow = new Set<string>();
      for (let i = 1; i < node.args.length; i++) {
        const a = node.args[i];
        if (a && a.type === 'KeywordArg') shadow.add(a.name);
      }
      // Inner body gets the reduced argSet; kwarg values stay full.
      const bodyArg = node.args[0];
      const innerSet = new Set<string>();
      for (const n of argSet) if (!shadow.has(n)) innerSet.add(n);
      const newArgs: any[] = [];
      newArgs.push(rewriteFreeIdsToPlaceholders(bodyArg, innerSet));
      for (let i = 1; i < node.args.length; i++) {
        const a = node.args[i];
        if (a && a.type === 'KeywordArg') {
          newArgs.push({
            ...a,
            value: rewriteFreeIdsToPlaceholders(a.value, argSet),
          });
        } else {
          newArgs.push(rewriteFreeIdsToPlaceholders(a, argSet));
        }
      }
      return { ...node, args: newArgs };
    }
    const out: Record<string, any> = {};
    for (const k in node) out[k] = rewriteFreeIdsToPlaceholders(node[k], argSet);
    return out;
  }

  function logicalSym(kind: string) {
    return (v.logicalSyms && v.logicalSyms[kind]) || null;
  }

  // Dot-notation desugaring (spec §05 Broadcasting syntax). Build a
  // `broadcast(<callee>, <operands…>)` CallExpr — the literal lowering
  // the spec mandates ("`A .+ B` lowers to `broadcast(add, A, B)`",
  // "`f.(<args>)` lowers to `broadcast(f, <args>)`"). Used for the
  // dotted forms the parser already desugars by builtin name (`.&&`,
  // `.||`, `.!`, `.^`) and for dot-call. Dotted arithmetic &
  // comparisons instead tag their Binary/UnaryExpr with
  // `broadcast:true` (see below) and are wrapped in lower.js, which is
  // the single owner of the operator→builtin map — so the parser
  // never duplicates that table.
  function broadcastCall(calleeNode: any, operands: any[], loc: any) {
    return AST.CallExpr(AST.Identifier('broadcast', loc),
      [calleeNode, ...operands], loc);
  }

  // True when the next token matches the variant's logical operator
  // for `kind` (one of 'and' / 'or' / 'not'). FlatPPL/FlatPPJ use the
  // dedicated AMPAMP / PIPEPIPE / BANG tokens; FlatPPY uses IDENT
  // tokens with values 'and'/'or'/'not'.
  function atLogicalSym(kind: string) {
    const sym = logicalSym(kind);
    if (sym === '&&') return at(T.AMPAMP);
    if (sym === '||') return at(T.PIPEPIPE);
    if (sym === '!')  return at(T.BANG);
    if (sym) return atValue(T.IDENT, sym);
    return false;
  }

  function parseOr(): any {
    let left: any = parseAnd();
    while (atLogicalSym('or')) {
      const opTok = advance();
      const right = parseAnd();
      const mloc = AST.loc(left.loc.start.line, left.loc.start.col,
                           right.loc.end.line, right.loc.end.col);
      const callee = AST.Identifier('lor', opTok.loc);
      left = opTok.dotted
        ? broadcastCall(callee, [left, right], mloc)
        : AST.CallExpr(callee, [left, right], mloc);
    }
    return left;
  }

  function parseAnd(): any {
    // Canonical FlatPPL keeps `!` at the Unary level (lower than
    // Comparison), so `&&` sits directly above Comparison.
    let left: any = parseComparison();
    while (atLogicalSym('and')) {
      const opTok = advance();
      const right = parseComparison();
      const mloc = AST.loc(left.loc.start.line, left.loc.start.col,
                           right.loc.end.line, right.loc.end.col);
      const callee = AST.Identifier('land', opTok.loc);
      left = opTok.dotted
        ? broadcastCall(callee, [left, right], mloc)
        : AST.CallExpr(callee, [left, right], mloc);
    }
    return left;
  }

  function parseComparison(): any {
    const compOpTypes = [T.EQEQ, T.NEQ, T.LT, T.GT, T.LTE, T.GTE];
    function isCompOp() {
      return compOpTypes.includes(peek().type)
        || (v.membershipOp && atValue(T.IDENT, 'in'));
    }
    function mergeLoc(a: any, b: any) {
      return AST.loc(a.loc.start.line, a.loc.start.col,
                     b.loc.end.line,   b.loc.end.col);
    }

    let left: any = parseAddition();
    if (!isCompOp()) return left;

    // First comparison — always emitted as a plain BinaryExpr to keep
    // the simple `a == b` case identical to the pre-chain shape.
    let opTok = advance();
    let right = parseAddition();
    let chain = AST.BinaryExpr(opTok.value, left, right, mergeLoc(left, right));
    // A dotted comparison broadcasts elementwise; the `land` chain
    // combiner (below) stays scalar. Each comparison desugars
    // independently, so mixed chains like `a < b .< c` are permitted.
    if (opTok.dotted) chain.broadcast = true;
    let lastRight = right;

    // Without chained comparison, a second operator at this level is
    // a parse error caught downstream (a stray token after the
    // expression).
    if (!v.chainedComparison) return chain;

    // Chained: `a < b <= c` lowers to `land(a < b, b <= c)`. Each
    // additional comparison's left operand is the previous
    // comparison's right operand, and the cascade is left-
    // associative. Note: `b` appears twice in the source-form
    // lowering — for complex middle terms with stochastic content,
    // hoist to a binding before chaining (each occurrence is its own
    // DAG node).
    while (isCompOp()) {
      opTok = advance();
      right = parseAddition();
      const cmp = AST.BinaryExpr(opTok.value, lastRight, right,
                                 mergeLoc(lastRight, right));
      if (opTok.dotted) cmp.broadcast = true;
      const callee = AST.Identifier('land', opTok.loc);
      chain = AST.CallExpr(callee, [chain, cmp], mergeLoc(chain, cmp));
      lastRight = right;
    }
    return chain;
  }

  function parseAddition(): any {
    let left: any = parseMultiplication();
    while (at(T.PLUS) || at(T.MINUS)) {
      const opTok = advance();
      const right = parseMultiplication();
      left = AST.BinaryExpr(opTok.value, left, right,
        AST.loc(left.loc.start.line, left.loc.start.col, right.loc.end.line, right.loc.end.col));
      if (opTok.dotted) left.broadcast = true;
    }
    return left;
  }

  function parseMultiplication(): any {
    let left: any = parseUnary();
    while (at(T.STAR) || at(T.SLASH)) {
      const opTok = advance();
      const right = parseUnary();
      left = AST.BinaryExpr(opTok.value, left, right,
        AST.loc(left.loc.start.line, left.loc.start.col, right.loc.end.line, right.loc.end.col));
      if (opTok.dotted) left.broadcast = true;
    }
    return left;
  }

  function parseUnary(): any {
    if (at(T.MINUS)) {
      const opTok = advance();
      const operand: any = parseUnary();
      const un: any = AST.UnaryExpr('-', operand,
        AST.loc(opTok.loc.start.line, opTok.loc.start.col, operand.loc.end.line, operand.loc.end.col));
      if (opTok.dotted) un.broadcast = true;  // `.-x` → broadcast(neg, x)
      return un;
    }
    // FlatPPL/FlatPPJ: `!` lives at the Unary level (binds tighter
    // than Comparison, so `!a < b` ≡ `(!a) < b`). FlatPPY's `not`
    // sits above Comparison and is handled by parseNot.
    if (at(T.BANG) && logicalSym('not') === '!') {
      const opTok = advance();
      const operand: any = parseUnary();
      const uloc = AST.loc(opTok.loc.start.line, opTok.loc.start.col,
                           operand.loc.end.line, operand.loc.end.col);
      const callee = AST.Identifier('lnot', opTok.loc);
      return opTok.dotted
        ? broadcastCall(callee, [operand], uloc)        // `.! x` → broadcast(lnot, x)
        : AST.CallExpr(callee, [operand], uloc);
    }
    return parseExponential();
  }

  // Exponential (spec §05): `Postfix ('^' Unary)?` — right-
  // associative because the right operand is a Unary that recurses
  // back into parseExponential. Binds tighter than unary `-` (so
  // `-x ^ 2` parses as `-(x ^ 2)`). Lowers to `pow(base, exponent)`.
  // FlatPPL/FlatPPJ accept `^`; FlatPPY does not (use `pow()`).
  function parseExponential(): any {
    const base: any = parsePostfix();
    if (!at(T.CARET)) return base;
    const caretTok = peek();
    if (!v.exponentOp) {
      diagnostics.push({
        severity: 'error',
        message: `'^' is not an operator in ${v.id} `
          + '(use `pow(base, exponent)` instead)',
        loc: caretTok.loc,
      });
      advance();
      parseUnary();  // consume the would-be exponent to avoid cascades
      return base;
    }
    advance();  // ^
    const exponent: any = parseUnary();
    const eloc = AST.loc(base.loc.start.line, base.loc.start.col,
                         exponent.loc.end.line, exponent.loc.end.col);
    const callee = AST.Identifier('pow', caretTok.loc);
    return caretTok.dotted
      ? broadcastCall(callee, [base, exponent], eloc)   // `a .^ b` → broadcast(pow, a, b)
      : AST.CallExpr(callee, [base, exponent], eloc);
  }

  function parsePostfix(): any {
    let expr: any = parsePrimary();

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
        // Index: expr[indices]. The lowering target (`get` for 1-
        // based / `get0` for 0-based) is fixed at parse time from
        // the variant — FlatPPY uses get0; FlatPPL/FlatPPJ use get.
        advance(); // [
        const indices = parseIndexList();
        const rbracket = expect(T.RBRACKET);
        const endLoc = rbracket ? rbracket.loc : expr.loc;
        expr = AST.IndexExpr(expr, indices,
          AST.loc(expr.loc.start.line, expr.loc.start.col, endLoc.end.line, endLoc.end.col),
          v.indexingLowersTo);
      } else if (at(T.DOT) && tokens[pos + 1]
                 && tokens[pos + 1].type === T.LPAREN) {
        // Dot-call (spec §05): `f.(args)` lowers to
        // `broadcast(f, args)`. One-token lookahead after `.`
        // distinguishes this from field access (`.` + Name). The
        // callee `f` becomes broadcast's first argument; positional
        // and keyword args flow straight through (broadcast's own
        // kwarg handling in lower.js applies, e.g.
        // `g.(a = s, x = p)` ≡ `broadcast(g, a = s, x = p)`).
        advance(); // .
        advance(); // (
        const args = parseArgList();
        const rparen = expect(T.RPAREN);
        const endLoc = rparen ? rparen.loc
          : args.length > 0 ? args[args.length - 1].loc : expr.loc;
        expr = broadcastCall(expr, args,
          AST.loc(expr.loc.start.line, expr.loc.start.col,
                  endLoc.end.line, endLoc.end.col));
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

  function parsePrimary(): any {
    const tok = peek();

    // Number
    if (at(T.NUMBER)) {
      advance();
      // Use Number() (handles hex 0x… correctly) rather than parseFloat
      // (which stops at the first non-decimal digit and returns 0 for
      // "0xb2"). Strip spec-allowed underscore digit separators first.
      const clean = tok.value.replace(/_/g, '');
      return AST.NumberLiteral(Number(clean), tok.value, tok.loc);
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

    // Identifier (may be bool/constant/set). Boolean spelling is
    // variant-specific (`true`/`false` in FlatPPL/FlatPPJ;
    // `True`/`False` in FlatPPY); the AST normalizes to a
    // BoolLiteral with a JS boolean value regardless of spelling.
    if (at(T.IDENT)) {
      advance();
      const { isConstant, isSet } = require('./builtins.ts');
      if (v.booleanLiterals && v.booleanLiterals[0] === tok.value) {
        return AST.BoolLiteral(true, tok.loc);
      }
      if (v.booleanLiterals && v.booleanLiterals[1] === tok.value) {
        return AST.BoolLiteral(false, tok.loc);
      }
      if (isConstant(tok.value)) return AST.ConstantRef(tok.value, tok.loc);
      if (isSet(tok.value)) return AST.SetRef(tok.value, tok.loc);
      return AST.Identifier(tok.value, tok.loc);
    }

    // AxisRef `.name` at the START of a Primary (spec §05 Axis names).
    // Distinguished from FieldAccess `expr.name` (handled in parsePostfix)
    // by position: a `.name` token sequence appearing where a Primary is
    // expected is an axis label; the same sequence appearing after a
    // postfix-able expression is field access. Per spec §05 closing
    // notes: "A `.Name` token is `FieldAccess` when it follows a
    // `Postfix`-able expression, and `Axis` otherwise (at the start of
    // a `Primary`)."
    if (at(T.DOT) && tokens[pos + 1] && tokens[pos + 1].type === T.IDENT) {
      const dotTok = advance();
      const nameTok = advance();
      return AST.AxisRef(nameTok.value,
        AST.loc(dotTok.loc.start.line, dotTok.loc.start.col,
                nameTok.loc.end.line, nameTok.loc.end.col));
    }

    // Parenthesized expression or tuple literal: (expr) | (expr, expr [, expr...])
    if (at(T.LPAREN)) {
      const lparen = advance(); // (
      const first: any = parseExpr();
      // Tuple? Requires at least one comma.
      if (at(T.COMMA)) {
        const elements: any[] = [first];
        while (at(T.COMMA)) {
          advance();
          if (at(T.RPAREN)) break; // trailing comma
          elements.push(parseExpr());
        }
        const rparen = expect(T.RPAREN);
        const endLoc: any = rparen ? rparen.loc : (elements[elements.length - 1] || first).loc;
        const tupleLoc: any = AST.loc(lparen.loc.start.line, lparen.loc.start.col, endLoc.end.line, endLoc.end.col);
        if (elements.length < 2) {
          diagnostics.push({
            severity: 'error',
            message: 'Tuples must have at least two elements; single-element tuples are not supported',
            loc: tupleLoc,
          });
        }
        return AST.TupleLiteral(elements, tupleLoc);
      }
      const rparen = expect(T.RPAREN);
      // Extend the inner expression's loc to include the surrounding
      // parens so source-slice display (e.g. analyzer's binding.rhs
      // for `E = (C - D) .^ 2`) shows the full parenthesised form
      // rather than chopping the leading `(`.
      if (rparen && first && first.loc) {
        first.loc = AST.loc(
          lparen.loc.start.line, lparen.loc.start.col,
          rparen.loc.end.line, rparen.loc.end.col);
      }
      return first;
    }

    // Array literal [a, b, c]
    if (at(T.LBRACKET)) {
      const lbracket = advance(); // [
      const elements: any[] = [];
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

    // Unexpected token: report and return an error placeholder.
    // Don't consume NEWLINE/EOF — those are needed to terminate the statement.
    const bad = peek();
    diagnostics.push({
      severity: 'error',
      message: `Unexpected token '${bad.value}' (${bad.type})`,
      loc: bad.loc,
    });
    if (!at(T.NEWLINE) && !at(T.EOF)) advance();
    return AST.Identifier('__error__', bad.loc);
  }

  function parseArgList() {
    const args: any[] = [];
    if (at(T.RPAREN)) return args;

    // FlatPPJ allows a Julia-style semicolon separator that switches
    // the arg list from positional to keyword: `f(x, y; a = 1, b = 2)`
    // is equivalent to `f(x, y, a = 1, b = 2)`. Outside FlatPPJ the
    // semicolon is a parse error (caught below).

    // A leading `;` (e.g. `f(; a = 1)`) is allowed in FlatPPJ for
    // pure-kwargs calls. In FlatPPL/FlatPPY we fall through to the
    // semicolon-rejection branch.
    if (!at(T.SEMI)) {
      args.push(parseArg());
      while (at(T.COMMA)) {
        advance();
        if (at(T.RPAREN) || at(T.SEMI)) break; // trailing comma
        args.push(parseArg());
      }
    }
    if (at(T.SEMI)) {
      const semiTok = peek();
      if (!v.semiKwargs) {
        diagnostics.push({
          severity: 'error',
          message: `';' is not allowed in ${v.id} call argument lists `
            + '(use `,` to separate kwargs)',
          loc: semiTok.loc,
        });
        // Recover by consuming the semicolon and continuing as if it
        // were a comma.
      }
      advance();  // consume ;
      // Everything after the semicolon must be a keyword arg.
      if (!at(T.RPAREN)) {
        const firstKw = parseArg();
        if (firstKw.type !== 'KeywordArg') {
          diagnostics.push({
            severity: 'error',
            message: 'Expected keyword argument after `;`',
            loc: firstKw.loc,
          });
        }
        args.push(firstKw);
        while (at(T.COMMA)) {
          advance();
          if (at(T.RPAREN)) break;
          const kw = parseArg();
          if (kw.type !== 'KeywordArg') {
            diagnostics.push({
              severity: 'error',
              message: 'Expected keyword argument after `;`',
              loc: kw.loc,
            });
          }
          args.push(kw);
        }
      }
    }

    // Enforce: positional args must come before keyword args
    let seenKwarg = false;
    for (const a of args) {
      if (a.type === 'KeywordArg') {
        seenKwarg = true;
      } else if (seenKwarg) {
        diagnostics.push({
          severity: 'error',
          message: 'Positional argument cannot follow keyword argument',
          loc: a.loc,
        });
        break;
      }
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
    const indices: any[] = [];
    indices.push(parseIndexArg());
    while (at(T.COMMA)) {
      advance();
      if (at(T.RBRACKET)) break;
      indices.push(parseIndexArg());
    }
    return indices;
  }

  function parseIndexArg() {
    // : means all (slice). Surface form for the `all` axis selector
    // (spec §07).
    if (at(T.COLON)) {
      const tok = advance();
      return AST.SliceAll(tok.loc);
    }
    // ! means only (singleton-axis selector; spec §07). Inside `[...]`
    // a BANG token followed immediately by `,` or `]` is the `only`
    // keyword; in any other position BANG is the unary logical-not
    // operator (handled by parseUnary). One-token lookahead — same
    // shape as the `.name` FieldAccess/Axis disambiguation.
    if (at(T.BANG)
        && tokens[pos + 1]
        && (tokens[pos + 1].type === T.COMMA
            || tokens[pos + 1].type === T.RBRACKET)) {
      const tok = advance();
      return AST.SliceOnly(tok.loc);
    }
    return parseExpr();
  }

  // Look ahead past `Name [` for the AggregateBinding shape:
  //   IDENT LBRACKET DOT IDENT (COMMA DOT IDENT)* RBRACKET COLON_EQ
  // Returns true only if that exact sequence matches; otherwise false
  // so the regular Binding/Decomposition path proceeds.
  function lookaheadIsAggregateBinding() {
    let i = pos + 2;  // skip IDENT then LBRACKET
    if (!tokens[i] || tokens[i].type !== T.DOT) return false;
    while (i < tokens.length) {
      if (!tokens[i] || tokens[i].type !== T.DOT) return false;
      i++;
      if (!tokens[i] || tokens[i].type !== T.IDENT) return false;
      i++;
      if (tokens[i] && tokens[i].type === T.COMMA) {
        i++;
        continue;
      }
      if (tokens[i] && tokens[i].type === T.RBRACKET) {
        return tokens[i + 1] != null && tokens[i + 1].type === T.COLON_EQ;
      }
      return false;
    }
    return false;
  }

  // Parse `Name [ .axis (, .axis)* ] := Expression`. The lookahead has
  // already confirmed the shape; we consume tokens and desugar
  // at parse time to `Name = aggregate(sum, [Axis, ...], Expression)`
  // so downstream passes see only the canonical CallExpr form.
  function parseAggregateBinding() {
    const nameTok = advance();              // IDENT
    advance();                              // [
    const axes: any[] = [];
    while (at(T.DOT)) {
      const dotTok = advance();
      const axisNameTok = advance();
      axes.push(AST.AxisRef(axisNameTok.value,
        AST.loc(dotTok.loc.start.line, dotTok.loc.start.col,
                axisNameTok.loc.end.line, axisNameTok.loc.end.col)));
      if (at(T.COMMA)) advance();
    }
    expect(T.RBRACKET);
    expect(T.COLON_EQ);
    const body = parseExpr();
    const stmtLoc = AST.loc(
      nameTok.loc.start.line, nameTok.loc.start.col,
      body.loc.end.line, body.loc.end.col);

    // Build the canonical desugar:
    //   <name> = aggregate(sum, [ax1, ax2, ...], <body>)
    const aggCall = AST.CallExpr(
      AST.Identifier('aggregate', stmtLoc),
      [
        AST.Identifier('sum', stmtLoc),
        AST.ArrayLiteral(axes, stmtLoc),
        body,
      ],
      stmtLoc);
    return AST.AssignStatement(
      [AST.Identifier(nameTok.value, nameTok.loc)],
      aggCall,
      stmtLoc);
  }

  // --- Statement parsing ---

  function parseStatement() {
    const startTok = peek();

    // Bare `_` (HOLE token) is a valid LHS name (discards the value).
    // The grammar's `Name` allows `_` lexically; we accept either IDENT or HOLE here.
    function isLhsName() { return at(T.IDENT) || at(T.HOLE); }

    if (!isLhsName()) {
      diagnostics.push({
        severity: 'error',
        message: `Expected variable name, got '${startTok.value}'`,
        loc: startTok.loc,
      });
      skipToNewline();
      return AST.ErrorStatement(startTok.value, startTok.loc);
    }

    // AggregateBinding (spec §05 Axis names + §04 §sec:aggregate):
    //   Name "[" Axis ("," Axis)* "]" ":=" Expression
    // Lowers at parse time to:
    //   Name "=" aggregate(sum, [Axis, ...], Expression)
    // Recognised by lookahead: after the leading Name, peek for `[`
    // then a sequence of `.IDENT` (axis) tokens separated by commas,
    // then `]` `:=`. If the pattern doesn't match we fall through to
    // the normal Binding/Tilde/Decomposition parse. The lookahead is
    // bounded by the closing bracket, so it stays cheap.
    if (at(T.IDENT) && tokens[pos + 1] && tokens[pos + 1].type === T.LBRACKET
        && lookaheadIsAggregateBinding()) {
      return parseAggregateBinding();
    }

    // Collect LHS names: name1, name2, ... = rhs. Reserved names
    // (variant-specific: spec §04 lists `and`/`or`/`not`/`True`/`False`
    // as reserved at the module level; FlatPPY additionally treats
    // `true`/`false` as reserved since they're shadowable but
    // round-tripping to FlatPPL would surprise readers) are rejected
    // with a clear diagnostic. Bare `_` is always allowed (discard
    // pattern; lowers to a private anon name).
    function checkReservedName(nameTok: any) {
      const isHole = nameTok.type === T.HOLE;
      if (isHole) return;  // `_` is always OK
      if (v.reservedAtBinding && v.reservedAtBinding.has(nameTok.value)) {
        diagnostics.push({
          severity: 'error',
          message: `'${nameTok.value}' is a reserved name in ${v.id} `
            + 'and cannot be bound',
          loc: nameTok.loc,
        });
      }
    }

    // Push one LHS name, renaming bare `_` (HOLE) to a unique
    // `__discard_N` per spec §04.
    function pushLhsName() {
      const tok = peek();
      checkReservedName(tok);
      const name = (tok.type === T.HOLE) ? freshDiscardName() : tok.value;
      names.push(AST.Identifier(name, tok.loc));
      advance();
    }

    const names: any[] = [];
    pushLhsName();

    while (at(T.COMMA)) {
      advance(); // ,
      if (isLhsName()) {
        pushLhsName();
      } else {
        diagnostics.push({
          severity: 'error',
          message: `Expected variable name after ','`,
          loc: peek().loc,
        });
        break;
      }
    }

    // Expect = or ~ (tilde for FlatPPL/FlatPPJ; spec §05). `x ~ M`
    // and `a, b ~ M` lower transparently to `x = draw(M)` and
    // `a, b = draw(M)` at parse time, so downstream passes see only
    // AssignStatement nodes.
    let isTilde = false;
    if (at(T.TILDE)) {
      if (!v.tildeBindings) {
        diagnostics.push({
          severity: 'error',
          message: `Tilde binding '~' is not allowed in ${v.id} `
            + '(use `name = draw(M)` instead)',
          loc: peek().loc,
        });
        skipToNewline();
        return AST.ErrorStatement(names.map(n => n.name).join(', '), startTok.loc);
      }
      advance(); // ~
      isTilde = true;
    } else if (!expect(T.EQUALS)) {
      skipToNewline();
      return AST.ErrorStatement(names.map(n => n.name).join(', '), startTok.loc);
    }

    // Parse RHS expression
    let value = parseExpr();

    if (isTilde) {
      // Wrap in draw(...). The synthetic CallExpr inherits the RHS's
      // location so error messages and source ranges point back at
      // the original expression.
      const callee = AST.Identifier('draw', value.loc);
      value = AST.CallExpr(callee, [value], value.loc);
    }

    const endLoc = value.loc;
    return AST.AssignStatement(names, value,
      AST.loc(startTok.loc.start.line, startTok.loc.start.col, endLoc.end.line, endLoc.end.col));
  }

  // --- Program ---

  function parseProgram() {
    const body: any[] = [];
    const comments: any[] = [];

    // Collect a doc-comment that PRECEDES the first statement (leading
    // form), if any.
    let leadingDoc = skipNewlinesCollectingDocs();
    while (!at(T.EOF)) {
      if (at(T.COMMENT)) {
        const c = advance();
        comments.push(AST.Comment(c.value, c.loc));
        leadingDoc = skipNewlinesCollectingDocs() || leadingDoc;
        continue;
      }

      const stmt = parseStatement();

      // Trailing-doc check: a single-line `%` doc-comment right after
      // the statement's RHS, before the next NEWLINE / SEMI / EOF /
      // COMMENT, attaches as trailing per spec §04. Block-form
      // doc-comments cannot be trailing (the fence requires its own
      // line). XOR rule: a binding has at most one doc; leading
      // beats trailing only if both present, which is itself an
      // error.
      let trailingDoc: any | null = null;
      let trailingDocLoc: any | null = null;
      if (at(T.DOC_LINE)) {
        const tok = peek();
        trailingDoc = { markup: tok.value.markup, lines: [tok.value.content] };
        trailingDocLoc = tok.loc;
        advance();
      } else if (at(T.DOC_BLOCK)) {
        // Block form in trailing position is a spec error.
        const tok = peek();
        diagnostics.push({
          severity: 'error',
          message: 'Block doc-comments (`%%%`) cannot be in trailing '
            + 'position; place it on its own line BEFORE the binding',
          loc: tok.loc,
        });
        advance();
      }

      // Resolve leading/trailing XOR.
      let attached: any | null = null;
      if (leadingDoc && trailingDoc) {
        diagnostics.push({
          severity: 'error',
          message: 'A binding may have a leading OR trailing doc-comment, '
            + 'not both',
          loc: trailingDocLoc,
        });
        attached = leadingDoc;
      } else {
        attached = leadingDoc || trailingDoc;
      }
      if (attached) stmt.doc = attached;
      leadingDoc = null;
      body.push(stmt);

      // Expect newline / SEMI(already-NEWLINE) / EOF / COMMENT after
      // statement (and now also DOC_*, which would have been the
      // trailing form handled above; defensively swallow any here).
      if (!at(T.NEWLINE) && !at(T.EOF) && !at(T.COMMENT)) {
        diagnostics.push({
          severity: 'error',
          message: `Expected end of line, got '${peek().value}'`,
          loc: peek().loc,
        });
        skipToNewline();
      }
      leadingDoc = skipNewlinesCollectingDocs();
    }

    // Orphan check: if leadingDoc is set at EOF, no binding followed.
    if (leadingDoc) {
      diagnostics.push({
        severity: 'warning',
        message: 'Doc-comment is not attached to any following binding',
        loc: { start: { line: 0, col: 0 }, end: { line: 0, col: 0 } },
      });
    }

    return AST.Program(body, comments);
  }

  const ast = parseProgram();
  return { ast, diagnostics };
}

module.exports = { parse };
