'use strict';
// Regressions for two viewer-optimizer fixes:
//
// (1) Self-contained objective. The viewer's "Find maximum" runs the optimizer
//     over many worker round-trips across several seconds, often in the
//     background, DURING which a concurrent plot rebuild fires
//     setEnv(merge:false) and wipes the worker session env. If the optimize
//     objective depended on fixed-phase self-refs (e.g. the observed data) in
//     that env, it would silently go −∞ mid-run and the optimizer would return
//     its seeded incumbent (the pivot). optimize-plot now BAKES every fixed
//     self-ref into the objective IR (substituteBoundaryValues) so it needs no
//     session env. This test guards that technique: baking the fixed-phase
//     self-refs of a lowered likelihood body leaves ZERO env-dependent refs.
//
// (2) Curvature width. optimize() returns a per-axis x-space marginal std `sd`
//     (the Laplace curvature), which the viewer uses to frame a sharply-peaked
//     likelihood (mode ± k·sd) instead of the much wider prior — the fix for
//     "the likelihood looks flat in basically all models".

const { test } = require('node:test');
const assert   = require('node:assert/strict');
const fs       = require('node:fs');
const path     = require('node:path');

const { ctxFor } = require('./density/regression-baseline.test.ts');
const { clm, orchestrator, density } = require('..');
const { optimize } = require('../optimizer/index.ts');

const FIX = path.join(__dirname, 'fixtures/baseline');

test('baking fixed-phase self-refs makes the optimize objective env-independent', () => {
  const src = fs.readFileSync(path.join(FIX, 'poisson-glm-link.flatppl'), 'utf8');
  const ctx = ctxFor(src, 1).ctx;
  const dst = { derivations: ctx.derivations, bindings: ctx.bindings, fixedValues: ctx.fixedValues };

  // Lower the likelihood L the way optimize-plot does: free the latents.
  const node = clm.lowerMeasure({ kind: 'ref', ns: 'self', name: 'L' }, dst,
    { boundaries: {}, freeInputs: ['intercept', 'slope'] });
  const ir = node.body;

  // The lowered body has at least one fixed-phase self-ref (the observed data)
  // that today rides the worker session env — the exact thing a merge:false
  // reset wipes.
  const fixedRefs = [...orchestrator.collectSelfRefs(ir)].filter((n) => dst.fixedValues.has(n));
  assert.ok(fixedRefs.length > 0, `expected fixed-phase self-refs, got ${JSON.stringify(fixedRefs)}`);

  // Bake them as optimize-plot does: valueToPlain FIRST (fixedValues holds
  // engine Value objects {shape, data} — substituteBoundaryValues→envValueToIR
  // needs plain arrays, else it mis-bakes an array as record(shape, data) and
  // broadcast/dot-call rejects it: "records and tuples are not allowed as
  // broadcast inputs").
  const fixedEnv: Record<string, any> = {};
  for (const n of fixedRefs) fixedEnv[n] = orchestrator.valueToPlain(dst.fixedValues.get(n));
  const baked = orchestrator.substituteBoundaryValues(ir, fixedEnv);

  // No env-dependent self-refs remain — the objective is self-contained.
  const remaining = [...orchestrator.collectSelfRefs(baked)].filter((n) => dst.fixedValues.has(n));
  assert.deepEqual(remaining, [], `baked body still depends on env: ${JSON.stringify(remaining)}`);

  // …and the array data baked as a proper `vector`, NOT a record(shape, data):
  // a record with a "shape"/"data" field is the mangling fingerprint.
  const json = JSON.stringify(baked);
  assert.ok(!/"name":"shape"/.test(json) && !/"name":"data"/.test(json),
    'array self-ref mis-baked as a record(shape, data) — valueToPlain missing');
});

test('the parameter being varied must NOT be baked — objective varies over it', () => {
  // Root cause of "optimizer returns the pivot" and "likelihood plot is flat":
  // classifyProfileSelfRefs returns EVERY stochastic self-ref in the lowered
  // body — including the latent we optimise / sweep (it has a prior, so it
  // looks random-phase). Both the optimize path and the profile-plot path bake
  // random-phase self-refs to a prior draw (samples[0]); if they bake the
  // varied parameter, the objective is constant at that draw (the pivot) →
  // optimizer can't climb, plot is a flat line. The fix: exclude the varied
  // parameter from the baking. This guards that contract end-to-end.
  const { materialiser } = require('..');
  const src = fs.readFileSync(path.join(FIX, 'poisson-glm-link.flatppl'), 'utf8');
  const ctx = ctxFor(src, 1).ctx;
  const dst = { derivations: ctx.derivations, bindings: ctx.bindings, fixedValues: ctx.fixedValues };
  const sig = orchestrator.signatureOf('L', dst.bindings);
  const observed = orchestrator.resolveIRToValue(sig.obsIR, dst.bindings, dst.fixedValues);
  const varied = 'intercept';
  const boundaries: Record<string, any> = {};
  for (const pn of sig.inputs.map((i: any) => i.paramName)) if (pn !== varied) boundaries[pn] = true;
  const ir = clm.lowerMeasure(sig.body, dst, { boundaries, freeInputs: [varied] }).body;

  const perAtom = materialiser.classifyProfileSelfRefs(ir, dst.bindings, dst.fixedValues).perAtomNames;
  assert.ok(perAtom.includes(varied),
    `classifyProfileSelfRefs should surface the varied param (got ${JSON.stringify(perAtom)}) — that is exactly why it must be excluded from baking`);

  // Build the bake set the FIXED way: random-phase self-refs MINUS the varied
  // param, plus fixed-phase data (held params kept at a fixed draw is fine).
  const fixedEnv: Record<string, any> = { slope: -1.3592 };   // a held non-varied draw
  for (const n of [...orchestrator.collectSelfRefs(ir)].filter((n: string) => dst.fixedValues.has(n)))
    fixedEnv[n] = orchestrator.valueToPlain(dst.fixedValues.get(n));
  const baked = orchestrator.substituteBoundaryValues(ir, fixedEnv);   // `varied` NOT in fixedEnv → stays free

  const pts = [-2.0, -0.244, 1.0];   // sweep the varied param
  const ra = { intercept: Float64Array.from(pts) };
  const lp: number[] = Array.from(density.logDensityN(baked, observed, ra, pts.length,
    { baseEnv: { x_data: dst.fixedValues.get('x_data') }, pointsBatched: false, parseSet: null }) as any);
  const spread = Math.max(...lp) - Math.min(...lp);
  assert.ok(spread > 1, `objective must vary over the swept param (spread ${spread.toFixed(3)} nats) — a flat (≈0) spread is the bug`);
});

test('optimize() returns a finite, peak-narrow curvature width (sd) per axis', async () => {
  // Same sharply-peaked linreg likelihood as optimizer-peaked-mle.test.ts.
  const X = [1.1, 1.5, 1.3, 1.4], Y = [3.2, 4.1, 3.4, 3.9];
  const logL = (p: number[]) => {
    const [a, b, s] = p; if (s <= 0) return -1e12;
    let t = 0; for (let i = 0; i < X.length; i++) { const d = Y[i] - (a + b * X[i]); t += -Math.log(s) - 0.5 * (d * d) / (s * s); }
    return t;
  };
  const domains = [{ kind: 'interval', lo: -1, hi: 2 }, { kind: 'interval', lo: 0, hi: 4 }, { kind: 'posreals' }];
  const scales = [3, 4, 0.45];
  const r = await optimize({ evalCloud: (pts: number[][]) => pts.map(logL), x0: [0.5, 2.0, 0.275], domains, scales, opts: { seed: 1 } });

  assert.ok(Array.isArray(r.sd) && r.sd.length === 3, 'sd is a length-3 array');
  for (let i = 0; i < 3; i++) {
    assert.ok(Number.isFinite(r.sd[i]) && r.sd[i] > 0, `sd[${i}] = ${r.sd[i]} finite & positive`);
    // The peak is far narrower than the prior/search scale — exactly why the
    // mode ± k·sd window de-flattens the plot. (k·sd well inside the scale.)
    assert.ok(4 * r.sd[i] < scales[i], `4·sd[${i}] = ${4 * r.sd[i]} < scale ${scales[i]} (peak narrower than prior)`);
  }
});

test('optimize() recovers a high-dim vector MLE via a per-slot scored cloud', async () => {
  // The eight-schools case: y_i ~ Normal(theta_i, sigma_i) — the likelihood
  // depends on the VECTOR `theta`, not on the scalar hyperparameters mu/tau.
  // The viewer optimises one dimension per array slot and scores the cloud with
  // a per-atom matrix (each atom its own theta-vector); this guards that the
  // optimiser recovers all 8 slots (the MLE is theta_i = y_i). A scalar-only
  // optimiser would leave theta pinned at a prior draw and never move.
  const y = [28, 8, -3, 7, -1, 1, 18, 12], se = [15, 10, 16, 11, 9, 11, 10, 18];
  const J = y.length;
  const logL = (theta: number[]) => {
    let s = 0; for (let i = 0; i < J; i++) { const d = (y[i] - theta[i]) / se[i]; s += -Math.log(se[i]) - 0.5 * d * d; }
    return s;
  };
  const r = await optimize({
    evalCloud: (pts: number[][]) => pts.map(logL),
    x0: new Array(J).fill(0),
    domains: new Array(J).fill({ kind: 'reals' }),
    scales: new Array(J).fill(20),
    opts: { seed: 0xf1a7, starts: 3 },
  });
  for (let i = 0; i < J; i++) {
    assert.ok(Math.abs(r.mode[i] - y[i]) < 1.5, `theta[${i}] ${r.mode[i].toFixed(2)} ≈ y ${y[i]}`);
  }
});
