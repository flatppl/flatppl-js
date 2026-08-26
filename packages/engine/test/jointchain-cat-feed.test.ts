'use strict';

// =====================================================================
// jointchain-cat-feed.test.ts — spec §06 gives jointchain the cat feed
// =====================================================================
//
// §06 dependent composition, on `jointchain(M, K1, K2, ...)`: "the remaining
// arguments are non-nullary kernels whose inputs bind to the variates of
// everything to their left", and the stochastic-node equivalence is
//
//   model = jointchain(M1, K2, K3)   ≡   a ~ M1
//                                        b ~ K2(a)
//                                        c ~ K3([a, b])
//                                        model = lawof([a, b, c])
//
// the SAME lowering §06 gives `kchain`. So step i binds the `cat` of every
// variate to its left, not the previous variate alone. §06 on the feed
// itself: "A non-record variate — for example the `cat`'d variate of a
// positional `joint` — carries no field names, so it feeds a kernel only
// when the kernel has a single input, to which the whole value is bound".
//
// The engine used to thread only the previous variate here, so a prev-only
// third step sampled and scored NaN with no diagnostic. It now types the
// boundary as the cat, which locates the error on the offending step and
// names `markovchain(kernel, init, n)` — §06's construct for the prev-only
// feed ("Step i is traj_i ~ kappa(traj_{i-1})").
//
// The keyword form is untouched: §06 makes it
// `jointchain(relabel(M, ["name1"]), ...)`, whose components bind to a
// kernel input by FIELD NAME (§04 calling convention), not off a cat.
//
// Every number here is hand-derived from the factorisation, never taken
// from an engine output.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function errorsOf(source: string): any[] {
  const lifted = processSource(source);
  const built = orchestrator.buildDerivations(lifted.bindings);
  return (lifted.diagnostics || []).concat(built.diagnostics || [])
    .filter((d: any) => d.severity === 'error');
}

function setupCtx(source: string, N: number, seed: number) {
  const lifted = processSource(source);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker: (m: any) => {
      const reply = worker.handle(m);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: N, rootSeed: seed, rootKey: [seed, 0],
    marginalizationCount: 32,
  };
  return ctx;
}

/** Normal log-pdf, closed form — the density oracle. */
function normLogpdf(x: number, mu: number, sigma: number) {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma)
    - 0.5 * ((x - mu) / sigma) * ((x - mu) / sigma);
}

// =====================================================================
// A prev-only third step is a located error naming markovchain
// =====================================================================

// All three step spellings lower to the same param-less or single-param
// `functionof`, so all three must report identically. The former behaviour
// was a silent NaN in the third variate for every one of them.
const PREV_ONLY_SPELLINGS: Array<[string, string]> = [
  ['fn placeholder', 'fn(Normal(mu = _, sigma = 1.0))'],
  ['kernelof named', 'kernelof(Normal(mu = prev, sigma = 1.0), prev = prev)'],
  ['lambda',         'p -> Normal(mu = p, sigma = 1.0)'],
];

for (const [label, kernel] of PREV_ONLY_SPELLINGS) {
  test('prev-only third step (' + label + ') is a located error', () => {
    const errs = errorsOf(`
flatppl_compat = "0.1"
K = ${kernel}
ch = jointchain(Normal(0, 1), K, K)
`);
    assert.equal(errs.length, 1, JSON.stringify(errs.map((d: any) => d.message)));
    assert.match(errs[0].message,
      /Normal: kwarg "mu" expects real, got array of real \(length 2\)/);
    // The construct that DOES mean the previous state alone.
    assert.match(errs[0].message, /markovchain\(kernel, init, n\)/);
    assert.match(errs[0].message, /jointchain feeds step 2 the `cat`/);
    assert.ok(errs[0].loc && errs[0].loc.start,
      'the diagnostic must locate the offending step');
  });
}

test('a 4-component prev-only chain names its first bad step', () => {
  // The migrated fixture's old shape: an AR-1 walk of one-input kernels.
  // Step 2 is fed 2 variates and step 3 is fed 3, so both report.
  const errs = errorsOf(`
flatppl_compat = "0.1"
K = kernelof(Normal(mu = prev, sigma = 0.5), prev = prev)
ch = jointchain(Normal(0, 0.1), K, K, K)
`);
  assert.equal(errs.length, 2, JSON.stringify(errs.map((d: any) => d.message)));
  assert.match(errs[0].message, /got array of real \(length 2\)/);
  assert.match(errs[1].message, /got array of real \(length 3\)/);
  for (const e of errs) assert.match(e.message, /markovchain/);
});

// =====================================================================
// The well-formed direction: a step that consumes the cat
// =====================================================================

// Hand-derived joint density at one point. With
//   x_0 ~ Normal(0, 1), x_1 ~ Normal(x_0, 1), x_2 ~ Normal(x_0 + x_1, 1)
// §06's density of a jointchain is the product of the constituent
// conditionals, so at (0.3, 0.7, 1.1) the third term's mean is
// sum([0.3, 0.7]) = 1.0 — the cat feed, not 0.7.
const CAT_CHAIN = `
flatppl_compat = "0.1"
k1 = fn(Normal(mu = _, sigma = 1.0))
k2 = fn(Normal(mu = sum(_), sigma = 1.0))
ch = jointchain(Normal(mu = 0.0, sigma = 1.0), k1, k2)
`;

test('cat-fed 3-step chain: logdensityof is the product of the conditionals', async () => {
  // The literal is §06's own: "logdensityof(jointchain(M, K), [a, b])".
  const ctx = setupCtx(CAT_CHAIN + `
ld = logdensityof(ch, [0.3, 0.7, 1.1])
`, 4, 3);
  const want = normLogpdf(0.3, 0.0, 1.0)
             + normLogpdf(0.7, 0.3, 1.0)
             + normLogpdf(1.1, 0.3 + 0.7, 1.0);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `logdensityof ${m.samples[0]} vs closed form ${want}`);
  // The prev-only reading would score the third term at mu = 0.7.
  const prevOnly = normLogpdf(0.3, 0.0, 1.0)
                 + normLogpdf(0.7, 0.3, 1.0)
                 + normLogpdf(1.1, 0.7, 1.0);
  assert.ok(Math.abs(m.samples[0] - prevOnly) > 0.05,
    'the score must distinguish the cat feed from a prev-only one');
});

test('cat-fed 3-step chain: the sampled covariance matches the closed form',
  async () => {
    // x_1 = x_0 + e_1, x_2 = 2·x_0 + e_1 + e_2, all e ~ N(0, 1) ⇒
    //   Var = {1, 2, 6},  Cov(x_0, x_1) = 1,
    //   Cov(x_0, x_2) = 2,  Cov(x_1, x_2) = 3
    const N = 40000;
    const ctx = setupCtx(CAT_CHAIN, N, 5);
    const m = await ctx.getMeasure('ch');
    // `shape: 'tuple'` is the materialiser's per-component sample CONTAINER,
    // which positional `joint` uses too — not a claim about the variate
    // domain, which §06 makes the `cat` (an array).
    assert.equal(m.shape, 'tuple', 'a positional chain keeps per-component columns');
    const cols = m.elems.map((e: any) => e.samples);
    assert.equal(cols.length, 3);
    for (const c of cols) {
      for (let i = 0; i < c.length; i++) {
        assert.ok(Number.isFinite(c[i]), 'no NaN in the sampled chain');
      }
    }
    const mean = (xs: any) => {
      let s = 0;
      for (let i = 0; i < xs.length; i++) s += xs[i];
      return s / xs.length;
    };
    const cov = (xs: any, ys: any) => {
      const mx = mean(xs), my = mean(ys);
      let s = 0;
      for (let i = 0; i < xs.length; i++) s += (xs[i] - mx) * (ys[i] - my);
      return s / xs.length;
    };
    const want: Array<[number, number, number]> = [
      [0, 0, 1], [1, 1, 2], [2, 2, 6],
      [0, 1, 1], [0, 2, 2], [1, 2, 3],
    ];
    for (const [i, j, w] of want) {
      const got = cov(cols[i], cols[j]);
      assert.ok(Math.abs(got - w) < 0.12 * w,
        `Cov(x_${i}, x_${j}) ≈ ${w}; got ${got.toFixed(3)}`);
    }
    for (let k = 0; k < 3; k++) {
      assert.ok(Math.abs(mean(cols[k])) < 0.05,
        `E[x_${k}] ≈ 0; got ${mean(cols[k]).toFixed(3)}`);
    }
  });

// =====================================================================
// Unchanged: the keyword form and the 2-component chain
// =====================================================================

test('the keyword form still binds by field name, not off a cat', async () => {
  // §06's keyword form is `jointchain(relabel(M, ["a"]), ...)`; a kernel
  // input named `b` binds the component labelled `b` (§04 calling
  // convention). So the third term's mean here is 0.7, not sum([0.3, 0.7]).
  const src = `
flatppl_compat = "0.1"
k1 = kernelof(Normal(mu = a, sigma = 1.0), a = a)
k2 = kernelof(Normal(mu = b, sigma = 1.0), b = b)
ch = jointchain(a = Normal(mu = 0.0, sigma = 1.0), b = k1, c = k2)
ld = logdensityof(ch, record(a = 0.3, b = 0.7, c = 1.1))
`;
  assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
  const ctx = setupCtx(src, 4, 3);
  const want = normLogpdf(0.3, 0.0, 1.0)
             + normLogpdf(0.7, 0.3, 1.0)
             + normLogpdf(1.1, 0.7, 1.0);
  const m = await ctx.getMeasure('ld');
  assert.ok(Math.abs(m.samples[0] - want) < 1e-12,
    `labelled logdensityof ${m.samples[0]} vs closed form ${want}`);
});

test('no false positive: a record cat splats into a lone named input', () => {
  // Positional form, but every component is record-valued, so the cat of the
  // two left variates is the merged record {a, b}. §04 sec:calling-convention:
  // "A sole positional record or table therefore always splats" — the third
  // step's lone input `b` binds the field `b`, not the whole record, which is
  // what the chain's runtime does. Shape of the det-js corpus case
  // `fragment/jointchain_chain3`.
  const errs = errorsOf(`
flatppl_compat = "0.1"
a = draw(Normal(mu = 0.0, sigma = 1.0))
k1 = kernelof(record(b = draw(Normal(mu = a, sigma = 0.5))), a = a)
k2 = kernelof(record(c = draw(Normal(mu = _b_, sigma = 0.25))), b = _b_)
j = jointchain(lawof(record(a = a)), k1, k2)
`);
  assert.deepEqual(errs.map((d: any) => d.message), []);
});

test('no false positive: a 2-component chain has one variate to its left',
  async () => {
    // One left variate binds whole, so a prev-only step is CORRECT here and
    // must not be flagged. Var(x_1) = 2 (a ~ N(0,1), b | a ~ N(a, 1)).
    const src = `
flatppl_compat = "0.1"
K1 = a -> Normal(mu = a, sigma = 1)
ch = jointchain(Normal(0, 1), K1)
`;
    assert.deepEqual(errorsOf(src).map((d: any) => d.message), []);
    const ctx = setupCtx(src, 20000, 11);
    const m = await ctx.getMeasure('ch');
    const xs = m.elems[1].samples;
    let s = 0;
    for (let i = 0; i < xs.length; i++) s += xs[i];
    const mu = s / xs.length;
    let v = 0;
    for (let i = 0; i < xs.length; i++) v += (xs[i] - mu) * (xs[i] - mu);
    v /= xs.length;
    assert.ok(Math.abs(v - 2) < 0.15, `2-step marginal variance ${v} ≈ 2`);
  });

// =====================================================================
// The broadcast executor refuses loudly rather than sampling prev-only
// =====================================================================

test('the broadcast scan refuses a prev-only step instead of threading it',
  async () => {
    // `_executeJointChainComposite` used to bind step k's input to step k-1's
    // column alone, which sampled a calibrated random walk from a model §06
    // does not describe. It now builds the cat, so a scalar-param body cannot
    // consume it — a located runtime refusal, not a wrong number.
    const src = `
flatppl_compat = "0.1"
m_per_group = [0.0, 1.0]
K = kernelof(Normal(mu = prev, sigma = 0.5), prev = prev)
cell = kernelof(jointchain(Normal(mu = m, sigma = 1.0), K, K), m = m)
y = broadcast(cell, m = m_per_group)
`;
    const errs = errorsOf(src).map((d: any) => d.message);
    assert.ok(errs.some((e: string) => /markovchain/.test(e)),
      'the static error names markovchain: ' + JSON.stringify(errs));
    const ctx = setupCtx(src, 100, 7);
    await assert.rejects(() => ctx.getMeasure('y'),
      /jointchain step 2 param 'mu' resolved to .*markovchain/s);
  });
