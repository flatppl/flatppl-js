'use strict';

// `mat-density.resolveNormalizeMasses` rewrites mass-carrying `normalize`
// nodes IN PLACE — `node.op = 'logweighted'`, `node.args = [...]`,
// `delete node.massFrom` — and its comment states what makes that sound: "the
// body is freshly built per lowerMeasure call (no cache; the CLM
// mutation-hazard rule)".
//
// Until the IR walks learned to preserve sharing, that soundness came for
// free: `materialiser-shared.inlineBoundaryDerivations` rebuilt an object for
// every node, so the body could not alias anything. It no longer does — it now
// returns the input node unchanged wherever no child changed, and memoises by
// node identity, because un-sharing the lowered DAG was the dominant cost of
// scoring a graph-shaped model. So the property has to be CHECKED rather than
// assumed.
//
// The property: every node the rewrite touches is private to the body it came
// from — reachable from no persistent binding IR, and by exactly one path
// inside the body. It holds because a mass-carrying `normalize` node is minted
// by `expandMeasure` per call rather than spliced out of a binding, so the
// sharing-preserving walk never aliases one. That is a conclusion about two
// modules acting together, which is exactly the kind of thing that breaks
// quietly later.
//
// `ctx.auditRewriteTargets` turns the check on; it is off in production and
// costs nothing there. These cases cover the arms that actually reach the
// rewrite: a 1-D lambda weight, a lambda weight over a shared intermediate, a
// reified (`functionof`) weight, and an N-D box.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function scoreAudited(src: string, binding: string, N = 512, seed = 7) {
  const proc = processSource(src);
  const errs = proc.diagnostics.filter((d: any) => d.severity === 'error');
  assert.equal(errs.length, 0, errs.map((e: any) => e.message).join(' | '));
  const built = orchestrator.buildDerivations(proc.linkedBindings || proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N,
    rootKey: seed, rootSeed: seed, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    // The subject of this file.
    auditRewriteTargets: true,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx.getMeasure(binding);
}

test('rewrite targets stay private: 1-D lambda weight', async () => {
  // w(x) = x², Z = 1/3, density 3x²; at 0.8 → log 1.92.
  const m = await scoreAudited(`
M = normalize(weighted(t -> t^2, Lebesgue(support = interval(0.0, 1.0))))
ld = logdensityof(M, 0.8)
`, 'ld');
  assert.ok(Math.abs(m.samples[0] - Math.log(1.92)) < 1e-7,
    'got ' + m.samples[0] + ', expected log 1.92');
});

test('rewrite targets stay private: lambda weight over a shared intermediate', async () => {
  // `scale` is a resolvable constant the lambda closes over — the case that
  // first surfaced the in-place rewrite under a frozen body.
  // w(t) = 3t² ⇒ Z = 1 and the density is still 3t².
  const m = await scoreAudited(`
scale = 3.0
M = normalize(weighted(t -> scale * t^2, Lebesgue(support = interval(0.0, 1.0))))
ld = logdensityof(M, 0.8)
`, 'ld');
  assert.ok(Math.abs(m.samples[0] - Math.log(1.92)) < 1e-7,
    'got ' + m.samples[0] + ', expected log 1.92');
});

test('rewrite targets stay private: a DAG-shaped reified weight', async () => {
  // `kernel_poly` is reached from both factors and `shared_leg` from both
  // terms, so the lowered weight graph is a diamond — the shape where the
  // sharing-preserving walk has the most to keep. Positive on [0, 1]: with
  // k = x² + 2x + 3 ∈ [3, 6], w = (k² + k)(k² − k) = k²(k² − 1) ≥ 72.
  const src = `
x = elementof(interval(0.0, 1.0))
kernel_poly = x^2 + 2*x + 3
shared_leg = kernel_poly * kernel_poly
graph_out = (shared_leg + kernel_poly) * (shared_leg - kernel_poly)
w = functionof(graph_out, x = x)
M = normalize(weighted(w, Lebesgue(support = interval(0.0, 1.0))))
ld = logdensityof(M, 0.4)
`;
  const m = await scoreAudited(src, 'ld');

  // Independent oracle: the 1-D normalizer is deterministic quadrature, so
  // log w(0.4) − log ∫₀¹ w agrees to the quadrature error of both sides.
  const k = (t: number) => t * t + 2 * t + 3;
  const wf = (t: number) => Math.pow(k(t), 4) - Math.pow(k(t), 2);
  const n = 200000;
  let Z = 0;
  for (let i = 0; i < n; i++) Z += wf((i + 0.5) / n) / n;
  const expected = Math.log(wf(0.4) / Z);
  assert.ok(Math.abs(m.samples[0] - expected) < 1e-7,
    'got ' + m.samples[0] + ', closed form ' + expected);
});

test('rewrite targets stay private: N-D box weight', async () => {
  // The box normalizer estimates Z from the box's own atoms, so this asserts
  // the audit passes and the score is finite rather than pinning a value —
  // the numeric oracle for this arm lives in weighted-reified-boundary.test.ts.
  const m = await scoreAudited(`
M = normalize(weighted((u, v) -> u * v,
      Lebesgue(support = cartprod(interval(0.0, 2.0), interval(0.0, 3.0)))))
ld = logdensityof(M, [0.5, 0.5])
`, 'ld', 4096);
  assert.ok(Number.isFinite(m.samples[0]), 'N-D box scored ' + m.samples[0]);
});

// ── the guard itself must be able to fail ─────────────────────────────────
//
// A guard that only ever passes proves nothing about the day the invariant
// breaks. These two build the violations by hand.

const { assertRewriteTargetsPrivate } = require('../mat-density.ts');

test('the guard rejects a rewrite target reachable from a binding IR', () => {
  const target = { kind: 'call', op: 'normalize', args: [], massFrom: { ref: 'inner' } };
  const bindingIR = { kind: 'call', op: 'weighted', args: [target] };
  const ctx = { bindings: new Map([['some_binding', { ir: bindingIR }]]) };
  const body = { kind: 'call', op: 'lawof', args: [target] };
  assert.throws(
    () => assertRewriteTargetsPrivate([target], body, ctx),
    /reachable from a persistent binding IR/);
});

test('the guard rejects a rewrite target reachable by two paths', () => {
  const target = { kind: 'call', op: 'normalize', args: [], massFrom: { ref: 'inner' } };
  // One object, two positions.
  const body = { kind: 'call', op: 'mul', args: [target, target] };
  const ctx = { bindings: new Map() };
  assert.throws(
    () => assertRewriteTargetsPrivate([target], body, ctx),
    /reachable by 2 paths in the body/);
});

test('the guard accepts a private rewrite target', () => {
  const target = { kind: 'call', op: 'normalize', args: [], massFrom: { ref: 'inner' } };
  const body = { kind: 'call', op: 'lawof', args: [target] };
  const ctx = { bindings: new Map([['other', { ir: { kind: 'lit', value: 1 } }]]) };
  assertRewriteTargetsPrivate([target], body, ctx);
});
