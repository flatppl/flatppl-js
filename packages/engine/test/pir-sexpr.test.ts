'use strict';

// Tests for pir-sexpr.ts — FlatPIR S-expression printer + reader.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');
const pirSexpr = require('../pir-sexpr.ts');

function buildModule(src: any) {
  return processSource(src).loweredModule;
}

// ---------------------------------------------------------------------
// Printer
// ---------------------------------------------------------------------

test('toSexpr: trivial scalar binding', () => {
  const mod = buildModule(`x = 3.14`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%module/);
  assert.match(out, /\(%bind x 3\.14\)/);
});

test('toSexpr: %public list includes user names', () => {
  const mod = buildModule(`x = 1\ny = 2`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%public x y\)/);
});

test('toSexpr: identifier ref → (%ref self <name>)', () => {
  const mod = buildModule(`x = 1\ny = x`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%bind y \(%ref self x\)\)/);
});

test('toSexpr: binary op call', () => {
  const mod = buildModule(`x = 1\ny = 2\nz = x + y`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%bind z \(add \(%ref self x\) \(%ref self y\)\)\)/);
});

test('toSexpr: kwarg call form', () => {
  const mod = buildModule(`m = Normal(mu = 0, sigma = 1)`);
  const out = pirSexpr.toSexpr(mod);
  // both kwargs in some order
  assert.match(out, /\(Normal /);
  assert.match(out, /\(%kwarg mu 0\)/);
  assert.match(out, /\(%kwarg sigma 1\)/);
});

test('toSexpr: array literal → vector', () => {
  const mod = buildModule(`v = [1.0, 2.0, 3.0]`);
  const out = pirSexpr.toSexpr(mod);
  // Real lits keep their lexical form on the wire (spec §11: the form
  // IS the type) — matches the spec's own `(vector 1.0 2.0 3.0)` example.
  assert.match(out, /\(vector 1\.0 2\.0 3\.0\)/);
});

test('toSexpr: record literal → record with %field entries', () => {
  const mod = buildModule(`r = record(mu = 0.0, sigma = 1.0)`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(record /);
  // Fields are positional-ordered structural entries
  assert.match(out, /\(%field mu 0\.0\)/);
  assert.match(out, /\(%field sigma 1\.0\)/);
});

// ---------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------

test('fromSexpr: simple module round-trip preserves bindings', () => {
  const src = `
(%module
  (%public x y)
  (%bind x 3.14)
  (%bind y (add (%ref self x) 1.0)))
`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.ok(module, 'should produce a module');
  assert.deepEqual(Array.from(module.publicSet), ['x', 'y']);
  assert.equal(module.bindings.size, 2);
  const yBind = module.bindings.get('y');
  assert.equal(yBind.rhs.kind, 'call');
  assert.equal(yBind.rhs.op, 'add');
  assert.equal(yBind.rhs.args[0].kind, 'ref');
  assert.equal(yBind.rhs.args[0].ns, 'self');
  assert.equal(yBind.rhs.args[0].name, 'x');
});

test('fromSexpr: comments and whitespace are tolerated', () => {
  const src = `
; A comment up top
(%module
  (%bind x 1)   ; inline
  ; comment between
  (%bind y (add (%ref self x) 1)))
`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal(module.bindings.size, 2);
});

test('fromSexpr: kwargs reconstruct call kwargs', () => {
  const src = `
(%module
  (%bind m (Normal (%kwarg mu 0) (%kwarg sigma 1))))
`;
  const { module } = pirSexpr.fromSexpr(src);
  const mBind = module.bindings.get('m');
  assert.equal(mBind.rhs.kind, 'call');
  assert.equal(mBind.rhs.op, 'Normal');
  assert.equal(mBind.rhs.kwargs.mu.value, 0);
  assert.equal(mBind.rhs.kwargs.sigma.value, 1);
});

test('fromSexpr: %field entries reconstruct record fields', () => {
  const src = `
(%module
  (%bind r (record (%field mu 0) (%field sigma 1))))
`;
  const { module } = pirSexpr.fromSexpr(src);
  const rBind = module.bindings.get('r');
  assert.equal(rBind.rhs.op, 'record');
  assert.equal(rBind.rhs.fields.length, 2);
  assert.equal(rBind.rhs.fields[0].name, 'mu');
  assert.equal(rBind.rhs.fields[1].name, 'sigma');
});

// ---------------------------------------------------------------------
// Round-trip
// ---------------------------------------------------------------------

test('round-trip: simple program → S-expr → module → S-expr is stable', () => {
  const src = `
x = 1
y = 2
z = x + y
m = Normal(mu = x, sigma = 1.0)
`;
  const mod = buildModule(src);
  const sexp1 = pirSexpr.toSexpr(mod);
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(sexp1);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const sexp2 = pirSexpr.toSexpr(mod2);
  // We don't insist on byte-identical output (kwarg ordering, blank
  // lines may shift), but the binding names should match and the
  // re-emitted text should be parseable and produce the same bindings.
  assert.equal(mod2.bindings.size, mod.bindings.size);
  for (const name of mod.bindings.keys()) {
    assert.ok(mod2.bindings.has(name), `binding ${name} missing after round-trip`);
  }
  // sanity: second pass produces a valid module too
  const { diagnostics: d2 } = pirSexpr.fromSexpr(sexp2);
  assert.deepEqual(d2.filter((d: any) => d.severity === 'error'), []);
});

// ---------------------------------------------------------------------
// User-defined-callable calls — spec §11 `(%call (%ref …) …)`
//
// lower.ts emits a user call as `{kind:'call', target:{ns,name}}` (no `op`).
// The emitter must head it with `(%call (%ref <ns> <name>) …)`, not with the
// (undefined) `op` — the old code printed a headless `( (%kwarg a 5))`, losing
// the callee name entirely.
// ---------------------------------------------------------------------

test('toSexpr: user-defined-callable call → (%call (%ref self g) …)', () => {
  const mod = buildModule(
    `a = elementof(reals)\ng = functionof(2 * a + 1, a = a)\ny = g(a = 5.0)`);
  const out = pirSexpr.toSexpr(mod);
  assert.match(out, /\(%bind y \(%call \(%ref self g\) \(%kwarg a 5\.0\)\)\)/);
});

test('fromSexpr: (%call (%ref …) …) reconstructs target + kwargs', () => {
  const src = `
(%module
  (%bind y (%call (%ref self g) (%kwarg a 5))))
`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const y = module.bindings.get('y');
  assert.equal(y.rhs.kind, 'call');
  assert.equal(y.rhs.op, undefined, 'a user call must not carry a bare op head');
  assert.deepEqual(y.rhs.target, { ns: 'self', name: 'g' });
  assert.equal(y.rhs.kwargs.a.value, 5);
});

test('fromSexpr: the ref-head is the target; a following ref is an arg', () => {
  // Disambiguation: the FIRST ref after %call is the callee head; any
  // further ref is an ordinary positional argument.
  const src = `
(%module
  (%bind z (%call (%ref self h) (%ref self b))))
`;
  const { module, diagnostics } = pirSexpr.fromSexpr(src);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  const z = module.bindings.get('z');
  assert.deepEqual(z.rhs.target, { ns: 'self', name: 'h' });
  assert.equal(z.rhs.args.length, 1);
  assert.equal(z.rhs.args[0].kind, 'ref');
  assert.equal(z.rhs.args[0].name, 'b');
});

test('round-trip: user-call survives module → S-expr → module (kwarg + positional)', () => {
  const src = `
a = elementof(reals)
g = functionof(2 * a + 1, a = a)
b = 4.0
y = g(a = 5.0)
z = g(b)
`;
  const mod = buildModule(src);
  const sexp = pirSexpr.toSexpr(mod);
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(sexp);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  for (const name of ['y', 'z']) {
    const r = mod2.bindings.get(name).rhs;
    assert.equal(r.kind, 'call');
    assert.deepEqual(r.target, { ns: 'self', name: 'g' },
      `${name}: callee target lost on round-trip`);
  }
  assert.equal(mod2.bindings.get('y').rhs.kwargs.a.value, 5);  // kwarg preserved
  assert.equal(mod2.bindings.get('z').rhs.args[0].name, 'b');  // positional ref preserved
});

test('numType wire fidelity: lit type IS the lexical form (spec §11) and round-trips', () => {
  // Spec §11: `3` is an integer, `1.0` a real — the lexical form carries
  // the type. The writer must therefore emit an integer-VALUED real lit
  // with a decimal point (previously `x = 3.0` emitted as `3`, silently
  // changing its type on the wire); the reader reconstructs `numType`
  // from the same lexical rule, so the tag survives the round-trip.
  const mod = buildModule(`
n = 3
x = 3.0
y = 1.5
`);
  const sexp = pirSexpr.toSexpr(mod);
  const flat = sexp.replace(/\s+/g, ' ');
  assert.ok(flat.includes('(%bind n 3)'),   'integer lit emits bare: ' + flat);
  assert.ok(flat.includes('(%bind x 3.0)'), 'integer-valued REAL lit keeps the decimal point: ' + flat);
  assert.ok(flat.includes('(%bind y 1.5)'), 'fractional real unchanged: ' + flat);
  const { module: mod2, diagnostics } = pirSexpr.fromSexpr(sexp);
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal(mod2.bindings.get('n').rhs.numType, 'integer');
  assert.equal(mod2.bindings.get('x').rhs.numType, 'real');
  assert.equal(mod2.bindings.get('y').rhs.numType, 'real');
  assert.equal(mod2.bindings.get('x').rhs.value, 3);
});

test('%meta emission (opt-in): annotated output matches the spec §11 annotated examples', () => {
  // Spec §11 "Annotated FlatPIR" (helpers/model worked examples). Emission
  // is OPT-IN — bare FlatPIR stays the canonical default shape. The
  // outermost call of each binding's RHS carries the binding-level phase;
  // inner calls carry type-only slots (phase %deferred — the spec allows
  // annotating only the outermost call). Unrepresentable types downgrade
  // to %deferred (≡ omitted), and a call where both slots would be
  // %deferred emits no %meta at all.
  const mod = buildModule(`
center = elementof(reals)
spread = elementof(posreals)
obs_kernel = functionof(
    Normal(mu = center + _x_, sigma = spread),
    center = center, spread = spread, x = _x_)
shifted_value = center + 1.0
b ~ Normal(mu = 0.0, sigma = 2.0)
`);
  const bare = pirSexpr.toSexpr(mod);
  assert.ok(!bare.includes('%meta'), 'default emission stays bare (canonical)');
  const out = pirSexpr.toSexpr(mod, { meta: true }).replace(/\s+/g, ' ');
  // The spec's own annotated lines, byte-for-byte:
  assert.ok(out.includes('(%bind center (elementof (%meta (%scalar real) %parameterized) reals))'),
    'elementof annotation matches the spec example: ' + out);
  assert.ok(out.includes('(%bind shifted_value (add (%meta (%scalar real) %parameterized) (%ref self center) 1.0))'),
    'outermost-call annotation matches the spec example: ' + out);
  assert.ok(out.includes('(functionof (%meta (%kernel (%inputs center spread x)) %fixed)'),
    'reification head annotation: kernel type with CALL-names, %fixed phase: ' + out);
  assert.ok(out.includes('(draw (%meta (%scalar real) %stochastic)'),
    'draw binding carries %stochastic: ' + out);
  // Inner calls: type-only (phase %deferred) — valid per spec.
  assert.ok(out.includes('(Normal (%meta (%measure (%domain (%scalar real))) %deferred)'),
    'inner call annotation is type-only: ' + out);
  // The annotated text reads back cleanly (the reader tolerates %meta).
  const { diagnostics } = pirSexpr.fromSexpr(pirSexpr.toSexpr(mod, { meta: true }));
  assert.deepEqual(diagnostics.filter((d: any) => d.severity === 'error'), []);
});
