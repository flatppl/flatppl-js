'use strict';

// Dot-notation broadcasting (spec §05 Broadcasting syntax / §04
// higher-order): `f.(args)` ≡ `broadcast(f, args)`, `a .op b` ≡
// `broadcast(opfn, a, b)`, `.op x` ≡ `broadcast(opfn, x)`. Dotted
// notation is pure syntactic sugar — it must lower to the *identical*
// IR an explicit `broadcast(...)` produces, so these tests drive the
// full tokenizer→parser→lower→value-evaluator chain and also pin the
// AST/IR shape and the lexer corner cases (Julia trailing-dot rule,
// dot-call vs field-access, no dotted `in`).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { tokenize, parse, lower } = require('..');
const sampler = require('../sampler.ts');
const V = require('../variants.ts').FLATPPL;

// Evaluate a single FlatPPL expression through the real pipeline.
function ev(expr) {
  const { tokens } = tokenize('R = ' + expr, { variant: 'flatppl' });
  const parsed = parse(tokens, V);
  assert.deepEqual(parsed.diagnostics || [], [],
    'unexpected parse diagnostics for: ' + expr);
  const ir = lower.lowerBinding(parsed.ast.body[0], {});
  return sampler.evaluateExpr(ir, {});
}

// Parse a statement; return { node, diagnostics }.
function parseStmt(src) {
  const { tokens } = tokenize(src, { variant: 'flatppl' });
  const parsed = parse(tokens, V);
  return {
    node: parsed.ast.body[0],
    diags: (parsed.diagnostics || []).map((d) => d.message),
  };
}

function lowerExpr(expr) {
  const { tokens } = tokenize('R = ' + expr, { variant: 'flatppl' });
  return lower.lowerBinding(parse(tokens, V).ast.body[0], {});
}

// Structural IR equality, ignoring source `loc` (differing column
// offsets between sugar and its explicit form are expected and
// irrelevant — only the lowered structure must match).
function stripLoc(x) {
  if (Array.isArray(x)) return x.map(stripLoc);
  if (x && typeof x === 'object') {
    const o = {};
    for (const k of Object.keys(x)) if (k !== 'loc') o[k] = stripLoc(x[k]);
    return o;
  }
  return x;
}

// --- dotted binary arithmetic -----------------------------------------

test('dotted +,-,*,/ over equal-length vectors', () => {
  assert.deepEqual(ev('[1.0,2.0,3.0] .+ [10.0,20.0,30.0]'), [11, 22, 33]);
  assert.deepEqual(ev('[10.0,20.0,30.0] .- [1.0,2.0,3.0]'), [9, 18, 27]);
  assert.deepEqual(ev('[1.0,2.0,3.0] .* [10.0,20.0,30.0]'), [10, 40, 90]);
  assert.deepEqual(ev('[10.0,20.0,30.0] ./ [10.0,10.0,10.0]'), [1, 2, 3]);
});

test('dotted op: singleton expansion and scalar loop-invariance', () => {
  assert.deepEqual(ev('[1.0,2.0,3.0] .+ [100.0]'), [101, 102, 103]);
  assert.deepEqual(ev('[100.0] .+ [1.0,2.0,3.0]'), [101, 102, 103]);
  assert.deepEqual(ev('[1.0,2.0,3.0] .* 2.0'), [2, 4, 6]);   // scalar held
  assert.deepEqual(ev('10.0 .- [1.0,2.0,3.0]'), [9, 8, 7]);
});

// --- dotted unary ------------------------------------------------------

test('dotted unary minus → broadcast(neg, …)', () => {
  assert.deepEqual(ev('.- [1.0,2.0,3.0]'), [-1, -2, -3]);
});

test('dotted unary not → broadcast(lnot, …)', () => {
  assert.deepEqual(ev('.! [true, false, true]'), [false, true, false]);
});

// --- dotted exponent / comparisons / logical --------------------------

test('dotted ^ → broadcast(pow, …) (right-assoc preserved)', () => {
  assert.deepEqual(ev('[2.0,3.0,4.0] .^ 2.0'), [4, 9, 16]);
  // `.^` is right-associative like `^`: 2 .^ (1 .^ 3) == 2
  assert.deepEqual(ev('2.0 .^ 1.0 .^ 3.0'), 2);
});

test('dotted comparisons produce elementwise booleans', () => {
  assert.deepEqual(ev('[1.0,2.0,3.0] .< [3.0,2.0,9.0]'), [true, false, true]);
  assert.deepEqual(ev('[1.0,2.0,3.0] .>= [3.0,2.0,1.0]'), [false, true, true]);
  assert.deepEqual(ev('[1.0,2.0,3.0] .== [1.0,9.0,3.0]'), [true, false, true]);
  assert.deepEqual(ev('[1.0,2.0,3.0] .!= [1.0,9.0,3.0]'), [false, true, false]);
});

test('dotted logical &&/|| (numeric land/lor, like plain)', () => {
  // `land`/`lor` return 1/0 — identical to the non-dotted lowering.
  assert.deepEqual(ev('[true,true,false] .&& [true,false,false]'), [1, 0, 0]);
  assert.deepEqual(ev('[true,false,false] .|| [false,true,false]'), [1, 1, 0]);
});

// --- precedence, chains, nesting --------------------------------------

test('dotted operators keep plain precedence', () => {
  // `.*` binds tighter than `.+`: 2 + 3*4 = 14
  assert.deepEqual(ev('[2.0] .+ [3.0] .* [4.0]'), [14]);
  // unary `.-` binds tighter than `.+`
  assert.deepEqual(ev('.- [1.0,2.0] .+ [10.0,10.0]'), [9, 8]);
});

test('nested / chained broadcast expressions', () => {
  assert.deepEqual(
    ev('([1.0,2.0,3.0] .+ [1.0,1.0,1.0]) .* [2.0,2.0,2.0]'), [4, 6, 8]);
  assert.deepEqual(
    ev('([4.0,9.0,16.0] .^ 0.5) .- [1.0,1.0,1.0]'), [1, 2, 3]);
  // comparison chain: a .< b .<= c  →  land(a .< b, b .<= c); each
  // comparison broadcasts (→ length-1 arrays), the land combiner is
  // the plain (scalar-callee) land applied elementwise.
  assert.deepEqual(ev('[1.0] .< [2.0] .<= [2.0]'), [true]);
  assert.deepEqual(ev('[1.0] .< [2.0] .<= [0.0]'), [false]);
});

test('mixed plain + dotted in one expression', () => {
  // plain scalar `*` then dotted vector `.+`
  assert.deepEqual(ev('([1.0,2.0,3.0] .* 2.0) .+ [1.0,1.0,1.0]'), [3, 5, 7]);
});

// --- dot-call ----------------------------------------------------------

test('dot-call f.(args): positional, anon, multi-arg', () => {
  assert.deepEqual(ev('fn(2.0 * _ + 1.0).([1.0,2.0,3.0])'), [3, 5, 7]);
  assert.deepEqual(
    ev('fn(_ + _).([1.0,2.0,3.0], [10.0,20.0,30.0])'), [11, 22, 33]);
});

test('dot-call result feeds into a dotted operator', () => {
  assert.deepEqual(
    ev('fn(_ * 2.0).([1.0,2.0,3.0]) .+ [1.0,1.0,1.0]'), [3, 5, 7]);
});

// --- spec-mandated builtin-name broadcast (the literal lowering) -------

test('broadcast(<builtin>, …) works (the form dotted ops lower to)', () => {
  assert.deepEqual(ev('broadcast(add, [1.0,2.0], [3.0,4.0])'), [4, 6]);
  assert.deepEqual(ev('broadcast(neg, [1.0,2.0,3.0])'), [-1, -2, -3]);
  assert.deepEqual(ev('broadcast(pow, [2.0,3.0], [2.0,3.0])'), [4, 27]);
});

// --- IR shape: dotted sugar ≡ explicit broadcast ----------------------

test('dotted op lowers identically to explicit broadcast(builtin,…)', () => {
  const dotted = lowerExpr('[1.0] .+ [2.0]');
  const explicit = lowerExpr('broadcast(add, [1.0], [2.0])');
  assert.deepEqual(stripLoc(dotted), stripLoc(explicit));
});

test('dot-call lowers identically to explicit broadcast(f,…)', () => {
  const dotted = lowerExpr('fn(_ * 2.0).([1.0,2.0])');
  const explicit = lowerExpr('broadcast(fn(_ * 2.0), [1.0,2.0])');
  assert.deepEqual(stripLoc(dotted), stripLoc(explicit));
});

test('synthesized callable is a functionof wrapping the op', () => {
  const ir = lowerExpr('[1.0,2.0] .* [3.0,4.0]');
  assert.equal(ir.op, 'broadcast');
  assert.equal(ir.args[0].op, 'functionof');
  assert.equal(ir.args[0].body.op, 'mul');
  assert.deepEqual(ir.args[0].params, ['_arg1_', '_arg2_']);
});

// --- lexer corner cases -----------------------------------------------

test('Julia trailing-dot rule: 1./y is float-divide, 1 ./ y broadcasts', () => {
  // `1./y` → `1.` munches as the real literal 1.0; plain `/`.
  const a = lowerExpr('1.0 / y');
  const plain = lowerExpr('1./y');
  assert.equal(plain.op, 'div');
  assert.deepEqual(stripLoc(plain), stripLoc(a));
  // `1 ./ y` → dotted divide (broadcast).
  const bc = lowerExpr('1 ./ y');
  assert.equal(bc.op, 'broadcast');
  assert.equal(bc.args[0].body.op, 'div');
});

test('dotted op needs no space when LHS is an identifier (A.+B)', () => {
  const noSpace = lowerExpr('A.+B');
  const spaced = lowerExpr('A .+ B');
  assert.deepEqual(stripLoc(noSpace), stripLoc(spaced));
});

test('`.(` is dot-call; `.name` stays field access', () => {
  assert.equal(parseStmt('y = r.field').node.value.type, 'FieldAccess');
  const dc = parseStmt('y = f.(x)').node.value;
  assert.equal(dc.type, 'CallExpr');
  assert.equal(dc.callee.name, 'broadcast');
});

test('`in` has no dotted form (.in is not a dotted operator)', () => {
  // `.in` must NOT tokenize as a dotted membership op; `.i` is not an
  // operator lexeme, so `.` is a bare DOT → parsed as field access,
  // leaving `b` stranded (a diagnostic), never a broadcast.
  const { node, diags } = parseStmt('y = a .in b');
  assert.ok(diags.length > 0, 'expected a diagnostic for `.in`');
  assert.notEqual(
    node.value && node.value.callee && node.value.callee.name, 'broadcast');
});

test('non-dotted operators are completely unchanged', () => {
  const plus = parseStmt('y = a + b').node.value;
  assert.equal(plus.type, 'BinaryExpr');
  assert.ok(!plus.broadcast);
  const and = parseStmt('y = a && b').node.value;
  assert.equal(and.callee.name, 'land');           // not broadcast-wrapped
  const pw = parseStmt('y = a ^ b').node.value;
  assert.equal(pw.callee.name, 'pow');
});

// --- tokenizer: every dotted lexeme + maximal munch -------------------

test('all dotted operator lexemes tokenize with the dotted marker', () => {
  const cases = {
    '.+': 'PLUS', '.-': 'MINUS', '.*': 'STAR', './': 'SLASH',
    '.^': 'CARET', '.<': 'LT', '.>': 'GT', '.<=': 'LTE', '.>=': 'GTE',
    '.==': 'EQEQ', '.!=': 'NEQ', '.&&': 'AMPAMP', '.||': 'PIPEPIPE',
    '.!': 'BANG',
  };
  for (const [lex, type] of Object.entries(cases)) {
    const { tokens } = tokenize('a ' + lex + ' b', { variant: 'flatppl' });
    const op = tokens[1];
    assert.equal(op.type, type, lex);
    assert.equal(op.dotted, true, lex + ' must carry dotted marker');
  }
  // maximal munch: `.<=` is one token, not `.<` then `=`
  const { tokens } = tokenize('a .<= b', { variant: 'flatppl' });
  assert.equal(tokens[1].type, 'LTE');
  assert.equal(tokens.filter((t) => t.type !== 'EOF').length, 3);
});
