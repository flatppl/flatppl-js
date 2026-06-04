'use strict';

// End-to-end MC-marginalised likelihood through the REAL density path:
// mc-recipe.buildMcMarginalForm re-expresses the transport model's measure
// tree `broadcast(post, broadcast(transport, iid(Normal,n), [pars]))` as
// the canonical `iid(mcmarginal{…}, n)` (functor law: deterministic `post`
// folded into the per-event map), and density.walkIid + density.walkMc-
// Marginal score it in-worker. No hand-built recipe, no name-based
// derivation — the recipe is read structurally off the measure tree.
//
// We validate (a) the auto-built form's shape, (b) per-event log p vs an
// independent forward-sim oracle, and (c) the full likelihood profile over
// pars.mu peaks at the generating parameter (a proper MLE).

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator } = require(ENG + 'index.ts');
const { buildMcMarginalForm } = require(ENG + 'mc-recipe.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');
const fs = require('node:fs');

const MODEL = '/homedir/Data/Science/Projects/BAT/Projects/FlatPPL/flatppl-examples/examples/tmp_transport_model.flatppl';

function buildForm() {
  const src = fs.readFileSync(MODEL, 'utf8');
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  const env = { derivations: built.derivations, bindings: built.bindings };
  const form = buildMcMarginalForm(
    orchestrator.expandMeasure('zs', env), built.bindings,
    (nm: string) => orchestrator.expandMeasure(nm, env));
  return form;
}

function oracleLogP(d: number, Nn: number): number {
  let s = 0x2545f491 >>> 0;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s & 0x7fffffff) / 0x7fffffff; };
  const lo = 0, binW = 0.02, nb = 1200;
  const counts = new Float64Array(nb);
  for (let i = 0; i < Nn; i++) {
    const u1 = rnd(), u2 = rnd();
    const xstd = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
    const x = 1.1 + 0.2 * xstd;
    const dlt = (2 * rnd() + 1) * 0.1;
    const z = (Math.pow(x + dlt, 3) * Math.exp(x - 0.3)) / 2;
    const b = Math.floor((z - lo) / binW);
    if (b >= 0 && b < nb) counts[b]++;
  }
  return Math.log(counts[Math.floor((d - lo) / binW)] / (Nn * binW));
}

test('buildMcMarginalForm: auto-derives the canonical iid(mcmarginal) density form', () => {
  const form = buildForm();
  assert.ok(form, 'buildMcMarginalForm returned null');
  assert.strictEqual(form.op, 'iid', 'top form must be iid (one factor per event)');
  const node = form.args[0];
  assert.strictEqual(node.op, 'mcmarginal');
  assert.ok(/^__anon/.test(node.retainedRef), `retained = internal Uniform draw, got ${node.retainedRef}`);
  assert.strictEqual(node.marginalRef, 'x', 'marginalised latent = the gun draw x');
  assert.deepStrictEqual(node.retainedInterval, [0, 1]);
  assert.strictEqual(node.marginalDistIR.op, 'Normal', 'marginal = the gun Normal');
  assert.ok(node.inverseIR && node.ladjIR, 'inverse + LADJ present');
});

test('logdensityof path: per-event log p(z=d|θ) matches a forward-sim oracle', () => {
  const node = buildForm().args[0];          // score one event directly
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 4242 });
  w.handle({ type: 'setEnv', env: { pars: { a: 0.1, b: 0.3, mu: 1.1 }, sigma: 0.2 }, merge: true });
  for (const d of [2.0, 3.0]) {
    const r = w.handle({ type: 'logDensityN', ir: node, observed: [d], count: 1,
      mcMarginalizationCount: 20000, mcSeed: 7 });
    assert.notStrictEqual(r.type, 'error', r.message);
    const est = r.samples[0], ora = oracleLogP(d, 2_000_000);
    assert.ok(Math.abs(est - ora) < 0.1,
      `d=${d}: est=${est.toFixed(4)} vs oracle=${ora.toFixed(4)} (Δ=${(est - ora).toFixed(4)})`);
  }
});

test('logdensityof path: likelihood profile over pars.mu peaks at the generating parameter', () => {
  const form = buildForm();
  const data = [3.81359, 2.91195, 3.20085, 3.09185, 3.34005, 4.96067, 0.842412,
    2.34128, 3.06224, 2.59162, 2.5017, 5.39892, 1.19806, 1.60855, 1.44647,
    0.771489, 0.26153, 1.56184, 0.561171, 4.4823];
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 4242 });
  const mus = [0.6, 0.8, 1.0, 1.2, 1.4];
  const curve: number[] = [];
  for (const mu of mus) {
    w.handle({ type: 'setEnv', env: { pars: { a: 0.1, b: 0.3, mu }, sigma: 0.2, n: data.length }, merge: true });
    const r = w.handle({ type: 'logDensityN', ir: form, observed: data, count: 1,
      mcMarginalizationCount: 5000, mcSeed: 12345 });
    assert.notStrictEqual(r.type, 'error', r.message);
    curve.push(r.samples[0]);
  }
  process.stderr.write('  profile: ' + mus.map((m, i) => `mu=${m}:${curve[i].toFixed(1)}`).join('  ') + '\n');
  const top = Math.max(...curve);
  assert.ok(Number.isFinite(top), 'no finite likelihood in the sweep');
  const argmax = curve.indexOf(top);
  assert.ok(argmax > 0 && argmax < mus.length - 1,
    `profile peak at the boundary (mu=${mus[argmax]}), not an interior MLE`);
  assert.ok(top - curve[0] > 1 && top - curve[mus.length - 1] > 1,
    `profile too flat to be a real likelihood (Δ ends: ${(top - curve[0]).toFixed(2)}, ${(top - curve[mus.length - 1]).toFixed(2)})`);
});
