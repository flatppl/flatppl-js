'use strict';

// =====================================================================
// measure-shape.test.ts — P2 sample/batch/event shape on measure types
// =====================================================================
//
// Pins the three-shape decomposition added to the measure type by P2
// (engine-concepts §18.11 / §20.10.5 item 2):
//
//   variate.shape == sampleShape ++ batchShape ++ eventShape
//
// today populated for:
//   - scalar leaf distributions  → eventShape = []
//   - `iid(M, n…)`               → prepends n… to sampleShape, carries
//                                  M's batch / event shapes through
//
// kernel-broadcast batchShape population is a future P2 follow-up.
// The fields are PURELY ADDITIVE — existing typeinfer / dispatch
// consumers must tolerate their absence; only the populated cases
// carry them.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const T = require('../types.ts');
const { processSource } = require('../index.ts');

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

// Pull a binding's inferred type out of the loweredModule.
function inferredTypeOf(src: string, name: string): any {
  const proc = processSource(src);
  const b = proc.loweredModule.bindings.get(name);
  if (!b) throw new Error('no binding ' + name);
  return b.inferredType;
}

// =====================================================================
// 1. Scalar leaf distributions — eventShape: []
// =====================================================================

test('measure type: Normal scalar dist has eventShape=[]', () => {
  const sig = T.signatureOf('Normal');
  assert.ok(sig, 'Normal signature exists');
  assert.equal(sig.result.kind, 'measure');
  assert.deepEqual(sig.result.eventShape, []);
  // sampleShape / batchShape unset on the bare leaf signature.
  assert.equal(sig.result.sampleShape, undefined);
  assert.equal(sig.result.batchShape,  undefined);
});

test('measure type: Poisson (integer scalar dist) has eventShape=[]', () => {
  const sig = T.signatureOf('Poisson');
  assert.deepEqual(sig.result.eventShape, []);
});

test('measure type: Bernoulli (boolean scalar dist) has eventShape=[]', () => {
  const sig = T.signatureOf('Bernoulli');
  assert.deepEqual(sig.result.eventShape, []);
});

// =====================================================================
// 2. iid populates sampleShape
// =====================================================================

test('measure type: iid(Normal, 10) carries sampleShape=[10] eventShape=[]', () => {
  const t = inferredTypeOf(`
flatppl_compat = "0.1"
mu = elementof(reals)
M = Normal(mu = mu, sigma = 1.0)
X = iid(M, 10)
`, 'X');
  assert.ok(t, 'X has inferredType');
  assert.equal(t.kind, 'measure');
  assert.deepEqual(t.sampleShape, [10]);
  // iid carries inner batch/event through.
  assert.deepEqual(t.batchShape, []);
  assert.deepEqual(t.eventShape, []);
});

test('measure type: nested iid stacks sampleShape outer-to-inner', () => {
  const t = inferredTypeOf(`
flatppl_compat = "0.1"
mu = elementof(reals)
M = Normal(mu = mu, sigma = 1.0)
Y = iid(iid(M, 5), 3)
`, 'Y');
  assert.equal(t.kind, 'measure');
  // Outer iid(_, 3) prepends; inner iid(M, 5) was already there.
  // Result: sampleShape = [3, 5].
  assert.deepEqual(t.sampleShape, [3, 5]);
  assert.deepEqual(t.eventShape, []);
});

test('measure type: iid with dynamic size carries %dynamic', () => {
  const t = inferredTypeOf(`
flatppl_compat = "0.1"
n = external(posintegers)
mu = elementof(reals)
M = Normal(mu = mu, sigma = 1.0)
X = iid(M, n)
`, 'X');
  assert.equal(t.kind, 'measure');
  assert.deepEqual(t.sampleShape, ['%dynamic']);
});

test('measure type: nested iid keeps domain as nested array (spec §03), but sampleShape is flat', () => {
  // FlatPPL's spec §03 distinguishes vectors-of-vectors from
  // matrices: `iid(iid(M, 7), 4)` produces a NESTED array
  // (`array<1, [4], array<1, [7], real>>`), not a flat matrix
  // `[4, 7]`. The sampleShape decomposition, by contrast, follows
  // the Pyro/TFP convention and flat-concats all iid axes —
  // sampleShape = [4, 7] regardless of how nesting expresses them
  // in the value-type tree.
  //
  // So the decomposition is implementer-side METADATA (a
  // correspondence to Pyro/TFP's flat-shape model) rather than a
  // strict structural invariant of the engine's nested type AST.
  const t = inferredTypeOf(`
flatppl_compat = "0.1"
mu = elementof(reals)
M = Normal(mu = mu, sigma = 1.0)
X = iid(iid(M, 7), 4)
`, 'X');
  assert.equal(t.kind, 'measure');
  assert.deepEqual(t.sampleShape, [4, 7]);
  // Domain: outer iid's own shape is [4]; inner [7] lives inside
  // the elem (a NESTED array, NOT a flat [4, 7] matrix).
  assert.equal(t.domain.kind, 'array');
  assert.deepEqual(t.domain.shape, [4]);
  assert.equal(t.domain.elem.kind, 'array');
  assert.deepEqual(t.domain.elem.shape, [7]);
});

// =====================================================================
// 3. Tolerance for missing fields
// =====================================================================

test('measure(domain) without opts produces no shape fields', () => {
  const m = T.measure({ kind: 'scalar', prim: 'real' });
  assert.equal(m.sampleShape, undefined);
  assert.equal(m.batchShape,  undefined);
  assert.equal(m.eventShape,  undefined);
});

test('measure(domain, opts) preserves only specified fields', () => {
  const m = T.measure({ kind: 'scalar', prim: 'real' }, { eventShape: [] });
  assert.deepEqual(m.eventShape, []);
  assert.equal(m.sampleShape, undefined);
  assert.equal(m.batchShape,  undefined);
});

// =====================================================================
// 4. Kernel-broadcast populates batchShape
// =====================================================================

test('measure type: broadcast(K, vec) populates batchShape', () => {
  // K is a kernel over scalar reals; broadcasting over a length-N
  // vector produces a measure with batchShape = [N] and the inner
  // kernel's eventShape = [].
  const t = inferredTypeOf(`
flatppl_compat = "0.1"
mu = elementof(reals)
K = fn(Normal(mu = mu + _, sigma = 1.0))
xs = [0.5, 1.0, 1.5, 2.0]
D = broadcast(K, xs)
`, 'D');
  assert.equal(t.kind, 'measure');
  assert.deepEqual(t.batchShape, [4]);
  assert.deepEqual(t.eventShape, []);
  assert.deepEqual(t.sampleShape, []);
});

test('measure type: kernel-broadcast batchShape distinct from iid sampleShape', () => {
  // Same outer length 4 but different decomposition: iid → sample;
  // kernel-broadcast → batch. The distinction matters for fusion
  // (b) and for density-walk axis-reduction (event axes sum out,
  // batch axes preserve, sample axes per-replicate).
  const iidT = inferredTypeOf(`
flatppl_compat = "0.1"
mu = elementof(reals)
M = Normal(mu = mu, sigma = 1.0)
X = iid(M, 4)
`, 'X');
  const bcastT = inferredTypeOf(`
flatppl_compat = "0.1"
mu = elementof(reals)
K = fn(Normal(mu = mu + _, sigma = 1.0))
xs = [0.5, 1.0, 1.5, 2.0]
D = broadcast(K, xs)
`, 'D');
  assert.deepEqual(iidT.sampleShape, [4]);
  assert.deepEqual(iidT.batchShape, []);
  assert.deepEqual(bcastT.sampleShape, []);
  assert.deepEqual(bcastT.batchShape, [4]);
});

// =====================================================================
// 5. substitute() preserves shape metadata
// =====================================================================

test('substitute() preserves sampleShape / batchShape / eventShape', () => {
  const tvar = { kind: 'var', id: 'T' };
  const m = T.measure(tvar, { sampleShape: [4], batchShape: [], eventShape: [3] });
  const subst = new Map([['T', { kind: 'scalar', prim: 'real' }]]);
  const out = T.substitute(m, subst);
  assert.equal(out.kind, 'measure');
  assert.deepEqual(out.sampleShape, [4]);
  assert.deepEqual(out.batchShape, []);
  assert.deepEqual(out.eventShape, [3]);
  // And the domain got substituted.
  assert.equal(out.domain.kind, 'scalar');
  assert.equal(out.domain.prim, 'real');
});
