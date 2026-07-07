'use strict';

// =====================================================================
// self-referential / cyclic deterministic bindings
// =====================================================================
//
// FlatPPL modules are DAGs (spec §04) — a binding may not depend on
// itself. buildDerivations must not emit a materialisable derivation
// for a cyclic deterministic binding; doing so makes the materialiser
// recurse forever (alias → getMeasure(self) is synchronous; evaluate →
// an opaque promise-chaining reject). Instead: drop the derivation and
// emit an actionable diagnostic.
//
// Cases:
//   f = f          alias-to-self  (the reported infinite recursion)
//   f = f + 1      self-ref evaluate
//   a = b; b = a   mutual cycle

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');

function build(src: string) {
  return orchestrator.buildDerivations(processSource(src).bindings);
}

function hasCycleDiag(ds: any, name: string): boolean {
  return (ds.diagnostics || []).some((d: any) =>
    d.severity === 'error' && /cycl|refers to itself|itself/i.test(d.message)
    && (d.message.indexOf("'" + name + "'") >= 0));
}

test('f = f (alias-to-self) is dropped with a cycle diagnostic', () => {
  const ds = build('f = f');
  assert.ok(!ds.derivations['f'], 'no materialisable derivation for f = f');
  assert.ok(hasCycleDiag(ds, 'f'), 'emits a cycle diagnostic naming f');
  // A cycle is a user error, not an "engine gap" — the fixed-phase dead-end
  // diagnostic must not ALSO fire for a binding already flagged as cyclic.
  assert.ok(!(ds.diagnostics || []).some((d: any) => /produced no value|engine gap/i.test(d.message)),
    'no redundant "engine gap" diagnostic for a cyclic binding');
});

test('f = f + 1 (self-ref evaluate) is dropped with a cycle diagnostic', () => {
  const ds = build('f = f + 1');
  assert.ok(!ds.derivations['f'], 'no materialisable derivation for f = f + 1');
  assert.ok(hasCycleDiag(ds, 'f'), 'emits a cycle diagnostic naming f');
});

test('a = b; b = a (mutual cycle) drops both with a diagnostic', () => {
  const ds = build('a = b\nb = a');
  assert.ok(!ds.derivations['a'] && !ds.derivations['b'],
    'no materialisable derivation for either side of a mutual cycle');
  assert.ok(hasCycleDiag(ds, 'a') || hasCycleDiag(ds, 'b'),
    'emits a cycle diagnostic naming a or b');
});

test('a = b; b = 2 (acyclic chain) is unaffected', () => {
  const ds = build('a = b\nb = 2');
  assert.ok(ds.fixedValues.has('a') || ds.derivations['a'],
    'a still resolves (b = 2 is a plain value)');
  assert.equal((ds.diagnostics || []).filter((d: any) => /cycl/i.test(d.message)).length, 0,
    'no spurious cycle diagnostic on an acyclic chain');
});
