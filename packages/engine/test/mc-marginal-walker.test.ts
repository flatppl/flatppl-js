'use strict';

// density.walkMcMarginal — the SYNC, in-worker Monte-Carlo marginalising-
// pushforward density rule (spec §06 case-3 opt-in; engine-concepts §6).
// It is the density sibling of walkPushfwd: the retained-innovation inverse
// + forward LADJ are EXACT (bijection-registry.invertExpr) and the latent
// is integrated by Monte Carlo. The rule draws latents IN-WORKER via the
// threaded RNG and evaluates the inverse/LADJ via the sync batched
// evaluator — no async worker round-trips.
//
// We validate the per-event log-density through the real worker logDensityN
// path against an INDEPENDENT forward-sim oracle, and confirm `iid(mcmarginal,
// n)` factorises per-event via walkIid.
//
// Per-event law (θ baked as lits):  z = ((x + (2u+1)·0.1)^3 · exp(x−0.3))/2
//   x ~ Normal(1.1, 0.2)   (marginalised latent),  u ~ Uniform(0,1) (retained).

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { invertExpr } = require(ENG + 'bijection-registry.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

const lit = (v: number) => ({ kind: 'lit', value: v, numType: Number.isInteger(v) ? 'integer' : 'real' });
const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const C = (op: string, ...args: any[]) => ({ kind: 'call', op, args });

function makeNode() {
  const delta = C('mul', C('add', C('mul', lit(2), ref('u')), lit(1)), lit(0.1));
  const recipeIR = C('divide',
    C('mul', C('pow', C('add', ref('x'), delta), lit(3)), C('exp', C('sub', ref('x'), lit(0.3)))),
    lit(2));
  const OUT = { kind: 'ref', ns: '%mc', name: '__mc_z__' };
  const inv = invertExpr({ outputExpr: recipeIR, freeRef: { name: 'u' }, outputValue: OUT });
  assert.ok(inv, 'invertExpr should invert the transport recipe in u');
  return {
    kind: 'call', op: 'mcmarginal',
    inverseIR: inv.inverseIR, ladjIR: inv.ladjIR,
    outName: '__mc_z__', retainedRef: 'u', retainedInterval: [0, 1],
    marginalRef: 'x', marginalDistIR: C('Normal', lit(1.1), lit(0.2)),
  };
}

// Independent oracle: forward-simulate z with a seeded PRNG, histogram,
// read density at d. No inverse/LADJ — pure simulation. `mu` is the latent
// x's mean (1.1 for the atom-independent tests; varied for the per-atom one).
function oracleLogPmu(d: number, mu: number, Nn: number): number {
  let s = 0x2545f491 >>> 0;
  const rnd = () => { s = (Math.imul(s, 1103515245) + 12345) >>> 0; return (s & 0x7fffffff) / 0x7fffffff; };
  const lo = 0, binW = 0.02, nb = 1200;
  const counts = new Float64Array(nb);
  for (let i = 0; i < Nn; i++) {
    const u1 = rnd(), u2 = rnd();
    const xstd = Math.sqrt(-2 * Math.log(u1 + 1e-12)) * Math.cos(2 * Math.PI * u2);
    const x = mu + 0.2 * xstd;
    const u = rnd();
    const dlt = (2 * u + 1) * 0.1;
    const z = (Math.pow(x + dlt, 3) * Math.exp(x - 0.3)) / 2;
    const b = Math.floor((z - lo) / binW);
    if (b >= 0 && b < nb) counts[b]++;
  }
  return Math.log(counts[Math.floor((d - lo) / binW)] / (Nn * binW));
}
const oracleLogP = (d: number, Nn: number) => oracleLogPmu(d, 1.1, Nn);

function makeWorker() {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 4242 });
  return w;
}

test('walkMcMarginal: per-event log p(z=d|θ) matches a forward-sim oracle (sync, in-worker)', () => {
  const w = makeWorker();
  const node = makeNode();
  for (const d of [1.8, 2.4, 3.2, 4.0]) {
    const r = w.handle({ type: 'logDensityN', ir: node, observed: [d], count: 1,
      mcMarginalizationCount: 20000, mcSeed: 7 });
    assert.notStrictEqual(r.type, 'error', r.message);
    const est = r.samples[0];
    const ora = oracleLogP(d, 2_000_000);
    assert.ok(Number.isFinite(est) && Number.isFinite(ora), `non-finite at d=${d}`);
    assert.ok(Math.abs(est - ora) < 0.1,
      `d=${d}: walkMcMarginal log p=${est.toFixed(4)} vs oracle=${ora.toFixed(4)} (Δ=${(est - ora).toFixed(4)})`);
  }
});

test('walkMcMarginal: iid(mcmarginal, n) factorises per-event via walkIid', () => {
  const w = makeWorker();
  const node = makeNode();
  const data = [2.2, 2.9, 3.5];
  const iidIR = { kind: 'call', op: 'iid', args: [node, lit(data.length)] };
  const total = w.handle({ type: 'logDensityN', ir: iidIR, observed: data, count: 1,
    mcMarginalizationCount: 8000, mcSeed: 7 }).samples[0];
  // Per-event oracle sum (the iid total must track Σ per-event within MC noise;
  // the walker draws fresh latents per event, so this is statistical, not exact).
  let oraSum = 0;
  for (const d of data) oraSum += oracleLogP(d, 2_000_000);
  assert.ok(Number.isFinite(total), 'iid total non-finite');
  assert.ok(Math.abs(total - oraSum) < 0.2,
    `iid total=${total.toFixed(4)} vs Σ oracle=${oraSum.toFixed(4)} (Δ=${(total - oraSum).toFixed(4)})`);
});

test('walkMcMarginal: per-atom (hierarchical) marginal scores each atom at its own θ', () => {
  const w = makeWorker();
  // A marginal dist that references a per-atom ref `mu_i` (supplied via
  // refArrays) is the hierarchical/posterior case — each prior particle
  // carries its own latent mean. The walker must produce one estimate PER
  // ATOM at that atom's θ (this is what makes bayesupdate over a reified
  // generative kernel yield a posterior — audit §3 / H1), not one shared
  // estimate and not a refusal.
  const node: any = makeNode();
  node.marginalDistIR = C('Normal', ref('mu_i'), lit(0.2));
  const r = w.handle({ type: 'logDensityN', ir: node, observed: [2.4], count: 2,
    refArrays: { mu_i: new Float64Array([1.0, 1.2]) }, mcMarginalizationCount: 20000, mcSeed: 7 });
  assert.notStrictEqual(r.type, 'error', r.message);
  assert.strictEqual(r.samples.length, 2, 'one log-density per atom');
  // Atom 0 (mu=1.0) and atom 1 (mu=1.2) score the SAME observation z=2.4 at
  // DIFFERENT latent means → distinct, each matching its own forward-sim oracle.
  const ora0 = oracleLogPmu(2.4, 1.0, 2_000_000);
  const ora1 = oracleLogPmu(2.4, 1.2, 2_000_000);
  assert.ok(Number.isFinite(r.samples[0]) && Number.isFinite(r.samples[1]), 'per-atom estimates finite');
  assert.ok(Math.abs(r.samples[0] - ora0) < 0.12,
    `atom0 (mu=1.0): walker ${r.samples[0].toFixed(4)} vs oracle ${ora0.toFixed(4)}`);
  assert.ok(Math.abs(r.samples[1] - ora1) < 0.12,
    `atom1 (mu=1.2): walker ${r.samples[1].toFixed(4)} vs oracle ${ora1.toFixed(4)}`);
  assert.notStrictEqual(r.samples[0], r.samples[1], 'per-atom estimates must differ across θ');
});
