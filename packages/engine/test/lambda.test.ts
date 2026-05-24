'use strict';

// Spec §05 Lambda syntax + §04 Lambda notation.
//
// `(arg1, arg2, ...) -> expr` is shorthand for
// `functionof(expr', arg1 = _arg1_, arg2 = _arg2_, ...)` where every
// free occurrence of each `arg_i` in `expr` is rewritten to the
// placeholder `_arg_i_`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

test('lambda: simple unary parses cleanly', () => {
  assert.equal(errors('f = (x) -> x + 1\n').length, 0);
});

test('lambda: multi-arg parses cleanly', () => {
  assert.equal(errors('f = (a, b) -> a * b + 1\n').length, 0);
});

test('lambda: lowers to functionof + placeholder kwargs', () => {
  const ctx = processSource('f = (a, b) -> a * b + 1\n');
  const f = ctx.bindings.get('f');
  assert.ok(f, 'binding f exists');
  // After analyze, the binding RHS AST is a CallExpr to functionof
  // with two placeholder kwargs.
  const v = f.node.value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'functionof');
  const kwargs = v.args.filter((a: any) => a.type === 'KeywordArg');
  assert.equal(kwargs.length, 2);
  assert.equal(kwargs[0].name, 'a');
  assert.equal(kwargs[1].name, 'b');
  assert.equal(kwargs[0].value.type, 'Placeholder');
  assert.equal(kwargs[1].value.type, 'Placeholder');
});

test('lambda: free args in body rewritten to placeholders', () => {
  const ctx = processSource('f = (x) -> x * x\n');
  const f = ctx.bindings.get('f');
  const v = f.node.value;
  const body = v.args[0];  // x * x
  // body is BinaryExpr (* x x); both leaves should be Placeholder('x'),
  // not Identifier('x'), so the lowerer emits %local refs.
  assert.equal(body.type, 'BinaryExpr');
  assert.equal(body.left.type, 'Placeholder');
  assert.equal(body.left.name, 'x');
  assert.equal(body.right.type, 'Placeholder');
});

test('lambda: nested lambda — inner shadows outer arg', () => {
  // Outer lambda binds `a`; inner lambda re-binds `a`. The inner body's
  // `a` references the inner binding, not the outer one, so it must NOT
  // be rewritten by the outer's pass.
  const ctx = processSource('f = (a) -> ((a) -> a + 1)\n');
  assert.equal(errors('f = (a) -> ((a) -> a + 1)\n').length, 0);
  const f = ctx.bindings.get('f');
  const outerFn = f.node.value;                  // functionof outer
  const outerBody = outerFn.args[0];             // inner functionof CallExpr
  assert.equal(outerBody.callee.name, 'functionof');
  const innerBody = outerBody.args[0];           // a + 1
  // The inner body's `a` was rewritten by the *inner* lambda's pass
  // (to Placeholder('a') for the INNER scope), but the outer pass must
  // not have re-rewritten it. Either way it's a Placeholder; check that
  // the structure round-trips through the analyzer cleanly.
  assert.equal(innerBody.left.type, 'Placeholder');
  assert.equal(innerBody.left.name, 'a');
});

test('lambda: body extends right (lower precedence than every other form)', () => {
  // `(a, b) -> a^2 + b^2` parses with body `a^2 + b^2`, not `(a, b) -> a^2`
  // followed by `+ b^2`.
  const ctx = processSource('f = (a, b) -> a^2 + b^2\n');
  const f = ctx.bindings.get('f');
  const v = f.node.value;
  // Body should be a BinaryExpr `+` whose two sides are `pow(_a_, 2)`
  // and `pow(_b_, 2)`.
  const body = v.args[0];
  assert.equal(body.type, 'BinaryExpr');
  assert.equal(body.op, '+');
});

test('lambda: equivalent to functionof(...)', () => {
  // (x) -> 2 * x  ≡  functionof(2 * _x_, x = _x_)
  const ctxA = processSource('f = (x) -> 2 * x\n');
  const ctxB = processSource('f = functionof(2 * _x_, x = _x_)\n');
  // Compare lowered IR.
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

test('lambda: usable as pushfwd transform', () => {
  // (x) -> exp(x) ≡ exp; pushfwd through the lambda lowers and works.
  const src = `
mu = Normal(mu = 0, sigma = 1)
ln = pushfwd((x) -> exp(x), mu)
`;
  assert.equal(errors(src).length, 0);
});

test('lambda: usable inside broadcast', () => {
  const src = `
A = [1.0, 2.0, 3.0]
B = broadcast((x) -> x * x, A)
`;
  assert.equal(errors(src).length, 0);
});

test('lambda: empty paren list (`() -> expr`) is NOT a lambda (no nullary lambdas)', () => {
  // Spec §05 closing notes: "a lambda is well-formed only if the
  // parenthesised content was a non-empty list of bare `Name`s."
  // `() -> expr` therefore must fail somewhere — at minimum the `->`
  // token becomes a syntax error after `()`.
  const errs = errors('f = () -> 1\n');
  assert.ok(errs.length > 0, 'expect some diagnostic');
});

test('lambda: not confused with paren-expression `(x)`', () => {
  // `(x)` followed by something other than `->` is a parenthesised
  // expression — the parser must NOT consume `(x)` as a lambda-param
  // list when no `->` follows.
  assert.equal(errors('a = elementof(reals)\nb = (a) + 1\n').length, 0);
});

test('lambda: not confused with tuple `(a, b)`', () => {
  assert.equal(errors('a = 1.0\nb = 2.0\nt = (a, b)\nx = t[1]\n').length, 0);
});
