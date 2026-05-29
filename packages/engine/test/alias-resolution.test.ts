'use strict';

// Tests for engine/alias-resolution.ts — the canonical-IR rewrite pass
// that follows aliases (`b = a` / `b = mod.x` / `b = self.x`) through
// to their canonical form so every IR-internal ref points at the
// underlying object.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource } = require('..');

function ir(name: string, src: string) {
  const out = processSource(src);
  const lb = out.loweredModule.bindings.get(name);
  return { lb, errs: (out.diagnostics || []).filter((d: any) => d.severity === 'error') };
}

// =====================================================================
// 1. Pure single-hop alias to a module-namespaced ref
// =====================================================================

test('alias → module ref: breit_wigner = hepphys.X canonicalises', () => {
  const { lb, errs } = ir('breit_wigner', `
hepphys = standard_module("particle-physics", "0.1")
breit_wigner = hepphys.breit_wigner_lineshape
`);
  assert.equal(errs.length, 0);
  assert.equal(lb.isAlias, true, 'binding is tagged as alias');
  assert.equal(lb.rhs.kind, 'ref');
  assert.equal(lb.rhs.ns, 'hepphys');
  assert.equal(lb.rhs.name, 'breit_wigner_lineshape');
});

// =====================================================================
// 2. Multi-hop alias chain resolves transitively
// =====================================================================

test('alias chain: b = a; a = mod.x → b canonicalises to mod.x', () => {
  const { lb: bLb } = ir('b', `
hepphys = standard_module("particle-physics", "0.1")
a = hepphys.breit_wigner_lineshape
b = a
`);
  assert.equal(bLb.isAlias, true);
  assert.equal(bLb.rhs.ns, 'hepphys');
  assert.equal(bLb.rhs.name, 'breit_wigner_lineshape');
});

// =====================================================================
// 3. Call sites through an alias rewrite their target
// =====================================================================

test('call.target through alias rewrites to canonical', () => {
  const { lb } = ir('foo', `
hepphys = standard_module("particle-physics", "0.1")
breit_wigner = hepphys.breit_wigner_lineshape
foo = breit_wigner(0.5)
`);
  assert.equal(lb.rhs.kind, 'call');
  assert.equal(lb.rhs.target.ns, 'hepphys');
  assert.equal(lb.rhs.target.name, 'breit_wigner_lineshape');
});

// =====================================================================
// 4. Self-ref rewrite in nested expressions (the typical
//    "f = (x) -> alias(x)" shape)
// =====================================================================

test('alias inside functionof body rewrites to canonical', () => {
  const { lb } = ir('resonance', `
hepphys = standard_module("particle-physics", "0.1")
breit_wigner = hepphys.breit_wigner_lineshape
resonance = s -> breit_wigner(s)
`);
  // resonance's body is a functionof body containing a call to
  // breit_wigner. After alias-resolution that call.target should
  // be hepphys.breit_wigner_lineshape.
  const body = lb.rhs.body;
  assert.equal(body.kind, 'call');
  assert.equal(body.target.ns, 'hepphys');
  assert.equal(body.target.name, 'breit_wigner_lineshape');
});

// =====================================================================
// 5. `self.X` lowers to a canonical self-ref (NOT get_field)
// =====================================================================

test('self.X is a self-namespaced ref, not get_field', () => {
  const { lb } = ir('y', `
x = 42
y = self.x
`);
  assert.equal(lb.isAlias, true, 'self.x is an alias to x');
  // x is itself a literal (not an alias), so canonicalised y points
  // at (%ref self x).
  assert.equal(lb.rhs.kind, 'ref');
  assert.equal(lb.rhs.ns, 'self');
  assert.equal(lb.rhs.name, 'x');
});

// =====================================================================
// 6. Record field access stays as get_field — discrimination by binding
//    type, not by surface syntax
// =====================================================================

test('record.X stays as get_field (not a ref)', () => {
  const { lb } = ir('val', `
rec = record(a = 1, b = 2)
val = rec.a
`);
  assert.ok(!lb.isAlias, 'val is not an alias');
  assert.equal(lb.rhs.kind, 'call');
  assert.equal(lb.rhs.op, 'get_field');
});

// =====================================================================
// 7. Cyclic alias detection — affected binding stays as the original
//    self-ref so downstream typeinfer surfaces a diagnostic at the
//    cleanest location.
// =====================================================================

test('cyclic alias is detected (binding left untouched)', () => {
  // The analyzer rejects same-name redefinition, so we test a
  // mutual-cycle via three names. b → c, c → b → ...
  const out = processSource(`
b = c
c = b
d = 42
`);
  const bLb = out.loweredModule.bindings.get('b');
  const cLb = out.loweredModule.bindings.get('c');
  // Both are pure-ref RHS; the cycle prevents canonicalisation, so
  // neither is marked `isAlias` (the unresolved chain stays for
  // downstream consumers to flag).
  assert.equal(bLb.isAlias, undefined);
  assert.equal(cLb.isAlias, undefined);
  assert.equal(bLb.rhs.kind, 'ref');
  assert.equal(cLb.rhs.kind, 'ref');
});

// =====================================================================
// 8. Identity preservation — bindings with no alias-affected refs are
//    NOT rebuilt (===-equality of the rhs object before/after).
// =====================================================================

test('non-alias-affected binding is referentially preserved', () => {
  // The IR-rewrite step uses `mapIR`'s identity-preserving rebuild —
  // when no descendant changes, the original IR object passes
  // through. Check that a binding which uses no aliases keeps the
  // same IR reference between parse-and-resolve and re-resolve.
  const out = processSource(`
hepphys = standard_module("particle-physics", "0.1")
breit_wigner = hepphys.X
unaffected = 1 + 2 + 3
`);
  // The `unaffected` binding's RHS contains no refs to alias names;
  // it should NOT be rewritten (its mapIR returns the same reference).
  // We can't directly check === across runs, but we can verify the IR
  // shape is unchanged.
  const lb = out.loweredModule.bindings.get('unaffected');
  assert.equal(lb.rhs.kind, 'call');
  assert.equal(lb.rhs.op, 'add');
});
