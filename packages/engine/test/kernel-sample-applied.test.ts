'use strict';

// deriveAppliedKernel — the k_model viewer-plot fix (orchestrator parameter-env
// path). A composite-generative kernel (`k_model` / `k_model_n` reifying
// `lawof(broadcast(post, broadcast(transport, iid(generator, n), [pars])))`)
// cannot be sampled by the viewer's substitute-IR reconstruction (it flattens
// the binding graph and the generative composite tangles — "undefined length").
// deriveAppliedKernel synthesizes the concrete application `<kernel>(<point>)`
// as a binding and re-derives, so it materialises BY NAME via the same
// generative derivation the hand-written `model_dist = k_model(glob_pars)` uses.

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

const SRC = fs.readFileSync(path.join(__dirname, 'fixtures', 'simple-transport1.flatppl'), 'utf8');
// `raw` = the analyzer bindings (pre-lift; what the viewer keeps as
// ctx.currentBindings). deriveAppliedKernel re-runs buildDerivations, so it must
// start from RAW bindings — re-lifting already-lifted bindings won't classify
// the synthetic application. `built` is used only for signatureOf + the oracle.
const rawBindings = () => processSource(SRC).bindings;
const build = () => orchestrator.buildDerivations(processSource(SRC).bindings);

function makeCtx(state: any, N: number) {
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 7 });
  const cache = new Map();
  const ctx: any = {
    derivations: state.derivations, bindings: state.bindings,
    fixedValues: state.fixedValues || new Map(), sampleCount: N, rootKey: 7,
    rootSeed: 7, marginalizationCount: 32,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}
function meanOf(m: any, cap: number): number {
  const f = m && (m.samples || (m.value && m.value.data));
  let s = 0; const n = Math.min(f.length, cap); for (let i = 0; i < n; i++) s += f[i];
  return s / n;
}

test('deriveAppliedKernel: k_model at glob_pars matches the model_dist oracle', async () => {
  const built = build();
  const sig = orchestrator.signatureOf('k_model', built.bindings, built.derivations);
  const env: any = {};
  for (const i of sig.inputs) env[i.paramName] = { a: 0.1, b: 0.3, mu: 1.1 };  // = glob_pars
  const applied = orchestrator.deriveAppliedKernel(rawBindings(), 'k_model', sig, env);
  assert.ok(applied.derivations[applied.name], 'applied k_model got a derivation (by-name materialisable)');

  const mA = await makeCtx(applied, 3000).getMeasure(applied.name);
  assert.ok(mA && mA.samples && mA.samples.length > 1000, 'applied k_model materialises a non-trivial sample');
  const meanA = meanOf(mA, 60000);
  assert.ok(Number.isFinite(meanA), 'applied k_model mean is finite');

  // Oracle: the hand-written `model_dist = k_model(glob_pars)` (the working
  // concrete-application chain). Same generative law at the same point ⇒ means
  // agree within MC noise (heavy-tailed cubic transform — loose band).
  const mB = await makeCtx(built, 3000).getMeasure('model_dist');
  const meanB = meanOf(mB, 60000);
  assert.ok(Math.abs(meanA - meanB) < 1.0,
    `k_model(glob_pars) mean ${meanA.toFixed(3)} ≈ model_dist mean ${meanB.toFixed(3)}`);
});

test('the clm kernel-sample path rejects for k_model (the gap the applied route closes)', async () => {
  // This pins WHY renderKernelSampleForCurrent falls through to
  // deriveAppliedKernel: the canonical-lowering primary (lowerMeasure
  // with explicit boundaries → materialiseMeasureIR), which works for
  // simple kernels, cannot materialise the APPLIED composite k_model —
  // its body is a user-call of k_model_n, so the by-name concrete-
  // application route is the correct mechanism (spec §06 uniform kernel
  // extension), not the IR-direct walk.
  const clm = require('../clm.ts');
  const built = build();
  const sig = orchestrator.signatureOf('k_model', built.bindings, built.derivations);
  const env: any = {};
  for (const i of sig.inputs) env[i.paramName] = { a: 0.1, b: 0.3, mu: 1.1 };
  const lowCtx = { derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map() };
  let node: any = null;
  try { node = clm.lowerMeasure(sig.body, lowCtx, { boundaries: env }); }
  catch (_e) { node = null; }
  if (node) {
    await assert.rejects(
      () => Promise.resolve(materialiser.materialiseMeasureIR(node, makeCtx(built, 500))),
      'clm materialiseMeasureIR should reject for the applied composite k_model');
  }
  // (node === null is equally a clean refusal — either way the viewer
  // proceeds to the applied route, pinned green by the tests above.)
});

test('deriveAppliedKernel: k_model_n (multi-input n + pars) materialises by name', async () => {
  const built = build();
  const sig = orchestrator.signatureOf('k_model_n', built.bindings, built.derivations);
  const env: any = {};
  for (const i of sig.inputs) env[i.paramName] = (i.kwargName === 'n') ? 20 : { a: 0.1, b: 0.3, mu: 1.1 };
  const applied = orchestrator.deriveAppliedKernel(rawBindings(), 'k_model_n', sig, env);
  assert.ok(applied.derivations[applied.name], 'applied k_model_n got a derivation');
  const m = await makeCtx(applied, 3000).getMeasure(applied.name);
  assert.ok(m && m.samples && m.samples.length > 1000 && Number.isFinite(meanOf(m, 60000)),
    'k_model_n applied materialises finite samples');
});
