// @ts-nocheck — test file; compiled separately by node --test (not by tsc)
//
// Regression: a WEIGHTED Lebesgue box must not render as its unweighted base.
//
// The bug: `matLebesgueBox` tagged its output with `dims` but not
// `shape = 'array'`, so `renderEmpiricalMeasure` missed the record/tuple/array
// branch and ran the SCALAR path over the flattened [N·k] buffer. There,
// `listScalarAxes` returned one unnamed axis instead of two, and the histogram
// dropped the N per-atom log-weights against N·k samples. The plot for
// `weighted(phase_weight, phase_space)` came out bit-identical to
// `Lebesgue(support = square)` while the caption correctly reported the
// weighted total mass and ESS.
//
// The subject is the D⁻ → 3π amplitude fixture, whose weight varies over the
// square by orders of magnitude — the shape that makes "weights honoured" and
// "weights dropped" unmistakably different plots. It lives in the engine's
// test-fixture tree (the examples-and-test-fixtures convention keeps one copy
// per repo), which is why this test reaches across the package boundary for it.

import { registerHooks } from 'node:module';
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';

// viewer/src uses bundler-style .js extensions in imports (resolved by esbuild
// at build time). Register a resolver hook so Node --test can load .ts source
// directly without a build step.
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('.js') && context.parentURL?.includes('/packages/viewer/src/')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context);
    }
    return nextResolve(specifier, context);
  }
});

const require_ = createRequire(import.meta.url);
const engine = require_('@flatppl/engine');
const { processSource, orchestrator, materialiser, histogram, empirical } = engine;
const { createWorkerHandler } = require_('@flatppl/engine/worker.ts');

// The viewer reads the engine off a host-provided global (the VS Code webview
// and the web gallery both set it before loading the bundle). Provide it so the
// modules under test resolve `FlatPPLEngine.*` the way they do in a browser.
globalThis.FlatPPLEngine = engine;

const { listScalarAxes } = await import('./util.ts');
const { measureToCsv } = await import('./export-samples.ts');

const FIXTURE = new URL(
  '../../engine/test/fixtures/dminus-to-3pi-amplitude.flatppl', import.meta.url);

// Mirrors the engine's own `makeMatCtx` test helper, inlined so the viewer
// package does not depend on the engine's test tree.
function materialiseAll(source, names, sampleCount) {
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d) => d.severity === 'error');
  assert.deepEqual(errs.map((e) => e.message), [], 'fixture must analyze clean');
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 0xC0FFEEEE });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount,
    rootSeed: 0xC0FFEEEE,
    moduleRegistry: lifted.loweredModule && lifted.loweredModule.moduleRegistry,
    getMeasure(name) {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker(msg) {
      const reply = worker.handle(msg);
      return reply && reply.type === 'error'
        ? Promise.reject(new Error(reply.message))
        : Promise.resolve(reply);
    },
  };
  return Promise.all(names.map((n) => ctx.getMeasure(n)));
}

const SAMPLE_COUNT = 4000;
const src = readFileSync(FIXTURE, 'utf8');
const [phaseSpace, amplitude] = await materialiseAll(
  src, ['phase_space', 'amplitude_measure'], SAMPLE_COUNT);

test('the box measures reach the viewer\'s record/tuple/array branch', () => {
  // `renderEmpiricalMeasure` dispatches on exactly this tag; without it the
  // corner grid, the Joint 2D surface and the Table are all unreachable.
  for (const [name, m] of [['phase_space', phaseSpace], ['amplitude_measure', amplitude]]) {
    assert.equal(m.shape, 'array', name + ": shape must be 'array'");
    assert.deepEqual(m.dims, [2], name + ': the square has two axes');
  }
});

test('listScalarAxes offers the two box coordinates, not one flat axis', () => {
  // The 2-D corner grid and the Joint 2D mode both need >= 2 axes; the scalar
  // fallback returned ONE axis over the interleaved [m, c] buffer.
  for (const [name, m] of [['phase_space', phaseSpace], ['amplitude_measure', amplitude]]) {
    const axes = listScalarAxes(m);
    assert.equal(axes.length, 2, name + ': expected two axes, got ' + axes.length);
    assert.deepEqual(axes.map((a) => a.label), ['[1]', '[2]'],
      name + ': axes carry 1-indexed component labels');
    for (const a of axes) {
      assert.equal(a.samples.length, SAMPLE_COUNT,
        name + ' ' + a.label + ': one sample per ATOM, not per cell');
    }
  }
});

// Area-normalised bin masses for one axis on a SHARED bin grid, so the
// weighted and unweighted marginals are directly comparable bin for bin.
function binnedMarginal(samples, weights, edges) {
  const nb = edges.length - 1;
  const width = edges[1] - edges[0];
  const out = new Float64Array(nb);
  const n = samples.length;
  let total = 0;
  for (let i = 0; i < n; i++) {
    const b = Math.floor((samples[i] - edges[0]) / width);
    if (b < 0 || b >= nb) continue;
    const w = weights ? weights[i] : 1 / n;
    out[b] += w;
    total += w;
  }
  if (total > 0) for (let b = 0; b < nb; b++) out[b] /= total;
  return out;
}

test('the weighted marginal differs from the unweighted one on the same atoms', () => {
  // The exact comparison the bug failed: `amplitude_measure` shares
  // `phase_space`'s atom POSITIONS and differs only in `logWeights`, so a
  // renderer that honours the weights must produce a different marginal and
  // one that drops them produces a bit-identical copy.
  const axes = listScalarAxes(amplitude);
  assert.ok(amplitude.logWeights, 'the weighted measure must carry log-weights');
  assert.equal(amplitude.logWeights.length, SAMPLE_COUNT,
    'log-weights are per atom — the length the axis columns pair with');
  const norm = histogram.normaliseWeights(amplitude.logWeights);

  for (const a of axes) {
    // The histogram itself must ACCEPT the pairing — under the old flattening
    // this call would now throw rather than silently drop the weights.
    const h = histogram.freedmanDiaconisHistogram(
      a.samples, { logWeights: amplitude.logWeights });
    assert.ok(h.ys.length > 1, a.label + ': expected a real binning');

    const weighted = binnedMarginal(a.samples, norm, h.binEdges);
    const dropped = binnedMarginal(a.samples, null, h.binEdges);
    // Total variation distance. Zero — exactly, not approximately — is what a
    // dropped weight gives, since both marginals are then the same atom counts.
    let tvd = 0;
    for (let b = 0; b < weighted.length; b++) tvd += Math.abs(weighted[b] - dropped[b]);
    tvd /= 2;
    assert.ok(tvd > 0.05,
      a.label + ': the weighted and unweighted marginals differ by TVD ' + tvd
      + ', too little to be a reweighted plot');
  }
});

test('the two box measures plot as different marginals, not one shared plot', () => {
  // The user-visible symptom stated as an assertion: the reweighted measure and
  // its base must not be interchangeable. Their atom positions ARE shared (that
  // is what importance weighting means), so the difference has to come from the
  // weights, and the base must carry none.
  assert.equal(phaseSpace.logWeights, null,
    'the plain box is uniform — no weights of its own');
  const [mAmp] = listScalarAxes(amplitude);
  const [mBase] = listScalarAxes(phaseSpace);
  const edges = histogram.freedmanDiaconisHistogram(mBase.samples).binEdges;
  const ampMarginal = binnedMarginal(
    mAmp.samples, histogram.normaliseWeights(amplitude.logWeights), edges);
  const baseMarginal = binnedMarginal(mBase.samples, null, edges);
  let tvd = 0;
  for (let b = 0; b < ampMarginal.length; b++) {
    tvd += Math.abs(ampMarginal[b] - baseMarginal[b]);
  }
  assert.ok(tvd / 2 > 0.05,
    'amplitude_measure and phase_space plot the same marginal (TVD ' + (tvd / 2) + ')');
});

test('the box marginal moments move the way the amplitude weight demands', () => {
  // Signed, closed-form oracles, so this fails on a DROPPED weight rather than
  // merely on any difference. The unweighted atoms are Uniform over the box, so
  //   m-axis:  E[m]  = (m_lo + m_hi)/2, m in [2·mpi, mD − mpi]
  //   c-axis:  E[c²] = 1/3             for c ~ Uniform(−1, 1)
  // both exact. The weight m·q(m)·k(m)·|A(m,c)|² is peaked at the isobar poles
  // (the lowest is the vector at 0.775 GeV), which pulls E[m] DOWN, and it
  // concentrates c away from the edges, which pulls E[c²] DOWN.
  //
  // Monte-Carlo margins at N = 4000 with ESS ≈ 49 %: sd(m) ≈ 0.42 over ~1960
  // effective atoms gives se ≈ 0.010, so the 0.03 band below is ~3σ on an
  // offset that is measured at 0.065. E[c²]'s se is ≈ 0.007, and the observed
  // shortfall from 1/3 is 0.042, so the 0.02 band is ~3σ likewise.
  const mLo = 2 * 0.13957039, mHi = 1.86965 - 0.13957039;
  const midM = (mLo + mHi) / 2;
  const [mAxis, cAxis] = listScalarAxes(amplitude);
  const norm = histogram.normaliseWeights(amplitude.logWeights);

  let wMeanM = 0, uMeanM = 0, wSqC = 0, uSqC = 0;
  for (let i = 0; i < SAMPLE_COUNT; i++) {
    wMeanM += norm[i] * mAxis.samples[i];
    uMeanM += mAxis.samples[i] / SAMPLE_COUNT;
    wSqC += norm[i] * cAxis.samples[i] * cAxis.samples[i];
    uSqC += cAxis.samples[i] * cAxis.samples[i] / SAMPLE_COUNT;
  }
  assert.ok(Math.abs(uMeanM - midM) < 0.02,
    'unweighted E[m] ' + uMeanM + ' must be the box midpoint ' + midM);
  assert.ok(Math.abs(uSqC - 1 / 3) < 0.02,
    'unweighted E[c²] ' + uSqC + ' must be the Uniform(−1,1) value 1/3');
  assert.ok(wMeanM < midM - 0.03,
    'weighted E[m] ' + wMeanM + ' must sit below the midpoint ' + midM);
  assert.ok(wSqC < 1 / 3 - 0.02,
    'weighted E[c²] ' + wSqC + ' must sit below the uniform value 1/3');
});

test('the scatter path importance-resamples instead of striding the atoms', () => {
  // What the corner grid's below-diagonal cells do (render-density.ts): a plain
  // stride would draw the UNWEIGHTED positions under a correct diagonal.
  const idx = empirical.systematicResample(amplitude.logWeights, 2000, () => 0.5);
  assert.equal(idx.length, 2000);
  const [mAxis] = listScalarAxes(amplitude);
  let resampled = 0;
  for (let i = 0; i < idx.length; i++) resampled += mAxis.samples[idx[i]] / idx.length;
  const mid = (2 * 0.13957039 + (1.86965 - 0.13957039)) / 2;
  assert.ok(resampled < mid - 0.05,
    'resampled scatter mean ' + resampled + ' must follow the weights, not the box');
});

test('the CSV export writes one column per coordinate plus the weight', () => {
  // The same flattening broke the export: one 2N-long column, and a weight
  // column indexed past its end for half the rows.
  const csv = measureToCsv(amplitude);
  const lines = csv.trim().split('\n');
  assert.equal(lines[0], '[1],[2],weight');
  assert.equal(lines.length, SAMPLE_COUNT + 1);
  for (const ln of [lines[1], lines[SAMPLE_COUNT]]) {
    const cells = ln.split(',');
    assert.equal(cells.length, 3);
    for (const c of cells) {
      assert.ok(Number.isFinite(Number(c)), 'every cell must be a number, got ' + c);
    }
  }
});
