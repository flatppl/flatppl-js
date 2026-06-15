'use strict';

// Spec §05 "Function definition syntax".
//
// `f(arg1, ...) = expr` is sugar for the lambda binding
// `f = (arg1, ...) -> expr` (single-arg: `f = arg1 -> expr`), which already
// lowers to `functionof` with placeholders. It introduces no new node and
// must produce output identical to the equivalent lambda binding.
//
// The core contract is EQUIVALENCE: the lowered FlatPIR of `f(args) = expr`
// is byte-identical to that of `f = (args) -> expr`.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const { toSexpr } = require('../pir-sexpr.ts');

function errors(src: string) {
  return processSource(src).diagnostics.filter(
    (d: any) => d.severity === 'error');
}

// FlatPIR projection of a source's lowered module — loc-independent, so it is
// the right granularity for an equivalence check.
function pir(src: string) {
  const ctx = processSource(src);
  assert.equal(
    ctx.diagnostics.filter((d: any) => d.severity === 'error').length, 0,
    `unexpected errors parsing:\n${src}`);
  return toSexpr(ctx.loweredModule);
}

// Each function definition lowers identically to its lambda binding.
const EQUIV: Array<[string, string]> = [
  // Multi-argument.
  ['f(x, y) = x * y + 1\n', 'f = (x, y) -> x * y + 1\n'],
  // Single-argument (desugars to the bare `arg -> expr` lambda).
  ['g(x) = 2 * x + 1\n', 'g = x -> 2 * x + 1\n'],
  // Record body — the body type is the output type, so this is multi-output.
  ['h(x, y) = record(p = x + y, q = x * y)\n',
    'h = (x, y) -> record(p = x + y, q = x * y)\n'],
  // Array body — likewise multi-output.
  ['k(x, y) = [x / y, x * y]\n', 'k = (x, y) -> [x / y, x * y]\n'],
  // Args shadow module bindings of the same name, exactly as lambda args do.
  ['x = elementof(reals)\nf(x) = x + x\n',
    'x = elementof(reals)\nf = x -> x + x\n'],
];

for (const [def, lam] of EQUIV) {
  test(`function def ≡ lambda: ${def.trim()}`, () => {
    assert.equal(pir(def), pir(lam),
      `function definition should lower identically to the lambda form`);
  });
}

test('function def: parses cleanly and lowers to functionof + placeholder kwargs', () => {
  const ctx = processSource('f(x, y) = x * y\n');
  assert.equal(errors('f(x, y) = x * y\n').length, 0);
  const f = ctx.bindings.get('f');
  assert.ok(f, 'binding f exists');
  const v = f.node.value;
  assert.equal(v.type, 'CallExpr');
  assert.equal(v.callee.name, 'functionof');
  const kwargs = v.args.filter((a: any) => a.type === 'KeywordArg');
  assert.equal(kwargs.length, 2);
  assert.equal(kwargs[0].name, 'x');
  assert.equal(kwargs[1].name, 'y');
  assert.equal(kwargs[0].value.type, 'Placeholder');
});

test('function def: free args in body rewritten to placeholders', () => {
  const ctx = processSource('f(x) = x * x\n');
  const body = ctx.bindings.get('f').node.value.args[0];
  assert.equal(body.type, 'BinaryExpr');
  assert.equal(body.left.type, 'Placeholder');
  assert.equal(body.right.type, 'Placeholder');
});

test('function def: at least one argument is required (no nullary)', () => {
  assert.ok(errors('f() = 1\n').length > 0, '`f() = 1` must be rejected');
});

test('function def: params are bare names, not keyword args', () => {
  // `f(x = a) = 1` has a `=` inside the parens — that is a keyword call shape,
  // not a definition parameter list, so it is not a function definition.
  assert.ok(errors('f(x = a) = 1\n').length > 0, '`f(x = a) = 1` must be rejected');
});

test('function def: a bare call is still not a statement', () => {
  assert.ok(errors('f(a, b)\n').length > 0, '`f(a, b)` must be rejected');
});
