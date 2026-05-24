'use strict';

// Spec §05 Lambda syntax + §04 Lambda notation.
//
// Two surface forms (spec `2eb595d`):
//   • Single-arg, bare:        `arg -> expr`
//   • Multi-arg, parenthesised: `(arg1, arg2, ...) -> expr`  (≥ 2 names)
// The single-arg parenthesised form `(arg) -> expr` is NOT legal —
// spec §05 explicitly rejects it.
//
// Both forms lower to `functionof(expr', arg = _arg_, ...)` with every
// free occurrence of each `arg_i` in `expr` rewritten to the placeholder
// `_arg_i_`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

test('lambda: single-arg bare form parses cleanly', () => {
  assert.equal(errors('f = x -> x + 1\n').length, 0);
});

test('lambda: multi-arg parses cleanly', () => {
  assert.equal(errors('f = (a, b) -> a * b + 1\n').length, 0);
});

test('lambda: single-arg lowers to functionof + placeholder kwarg', () => {
  const ctx = processSource('f = x -> 2 * x\n');
  const f = ctx.bindings.get('f');
  assert.ok(f, 'binding f exists');
  const v = f.node.value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'functionof');
  const kwargs = v.args.filter((a: any) => a.type === 'KeywordArg');
  assert.equal(kwargs.length, 1);
  assert.equal(kwargs[0].name, 'x');
  assert.equal(kwargs[0].value.type, 'Placeholder');
});

test('lambda: multi-arg lowers to functionof + placeholder kwargs', () => {
  const ctx = processSource('f = (a, b) -> a * b + 1\n');
  const f = ctx.bindings.get('f');
  const v = f.node.value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'functionof');
  const kwargs = v.args.filter((a: any) => a.type === 'KeywordArg');
  assert.equal(kwargs.length, 2);
  assert.equal(kwargs[0].name, 'a');
  assert.equal(kwargs[1].name, 'b');
});

test('lambda: free args in body rewritten to placeholders', () => {
  const ctx = processSource('f = x -> x * x\n');
  const f = ctx.bindings.get('f');
  const v = f.node.value;
  const body = v.args[0];  // x * x
  assert.equal(body.type, 'BinaryExpr');
  assert.equal(body.left.type, 'Placeholder');
  assert.equal(body.left.name, 'x');
  assert.equal(body.right.type, 'Placeholder');
});

test('lambda: nested single-arg lambdas are right-associative', () => {
  // `x -> y -> x + y` parses as `x -> (y -> x + y)`. The body of the
  // outer is itself a lambda — and its body `x + y` references both
  // arg names. The outer rewrite handles `x`; the inner already
  // handled `y` during its own parse.
  assert.equal(errors('f = x -> y -> x + y\n').length, 0);
  const ctx = processSource('f = x -> y -> x + y\n');
  const f = ctx.bindings.get('f');
  const outerFn = f.node.value;                  // functionof outer
  assert.equal(outerFn.callee.name, 'functionof');
  const outerBody = outerFn.args[0];             // inner functionof CallExpr
  assert.equal(outerBody.type, 'CallExpr');
  assert.equal(outerBody.callee.name, 'functionof');
});

test('lambda: nested lambda — inner shadows outer arg', () => {
  // Outer binds `a`; inner re-binds `a`. The inner body's `a`
  // references the inner binding, not the outer one.
  const src = 'f = a -> (a -> a + 1)\n';
  assert.equal(errors(src).length, 0);
});

test('lambda: body extends right (lower precedence than every other form)', () => {
  // `(a, b) -> a^2 + b^2` parses with body `a^2 + b^2`.
  const ctx = processSource('f = (a, b) -> a^2 + b^2\n');
  const f = ctx.bindings.get('f');
  const v = f.node.value;
  const body = v.args[0];
  assert.equal(body.type, 'BinaryExpr');
  assert.equal(body.op, '+');
});

test('lambda: bare and parenthesised multi-arg are equivalent to functionof(...)', () => {
  // x -> 2 * x  ≡  functionof(2 * _x_, x = _x_)
  const ctxA = processSource('f = x -> 2 * x\n');
  const ctxB = processSource('f = functionof(2 * _x_, x = _x_)\n');
  function stripLocs(n: any): any {
    if (n == null) return n;
    if (Array.isArray(n)) return n.map(stripLocs);
    if (typeof n !== 'object') return n;
    const out: any = {};
    for (const k of Object.keys(n)) {
      if (k === 'loc') continue;
      out[k] = stripLocs(n[k]);
    }
    return out;
  }
  assert.deepEqual(stripLocs(ctxA.bindings.get('f').ir),
                   stripLocs(ctxB.bindings.get('f').ir));
});

test('lambda: usable as pushfwd transform (spec §02 example)', () => {
  // Spec §02-overview example after `2eb595d` uses bare-arg lambda.
  const src = `
mu = Normal(mu = 0, sigma = 1)
ln = pushfwd(x -> exp(x), mu)
`;
  assert.equal(errors(src).length, 0);
});

test('lambda: usable inside broadcast', () => {
  const src = `
A = [1.0, 2.0, 3.0]
B = broadcast(x -> x * x, A)
`;
  assert.equal(errors(src).length, 0);
});

test('lambda: NULLARY `() -> expr` is rejected', () => {
  // Spec §05: "no nullary lambdas".
  const errs = errors('f = () -> 1\n');
  assert.ok(errs.length > 0, 'expect a diagnostic');
});

test('lambda: SINGLE-ARG PAREN `(x) -> expr` is rejected per spec §05', () => {
  // Spec `2eb595d` (2026-05-24): paren-form lambdas require ≥ 2 names.
  // `(x) -> expr` is a parenthesised expression `(x)` followed by a
  // stray `->`. Stays a parse error.
  const errs = errors('f = (x) -> x + 1\n');
  assert.ok(errs.length > 0,
    '(x) -> expr should be rejected as not-a-lambda');
});

test('lambda: not confused with paren-expression `(x)`', () => {
  // `(x)` followed by something other than `->` is a parenthesised
  // expression — the parser must NOT consume `(x)` as a lambda.
  assert.equal(errors('a = elementof(reals)\nb = (a) + 1\n').length, 0);
});

test('lambda: not confused with tuple `(a, b)`', () => {
  assert.equal(errors('a = 1.0\nb = 2.0\nt = (a, b)\nx = t[1]\n').length, 0);
});
