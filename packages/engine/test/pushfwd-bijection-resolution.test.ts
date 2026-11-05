'use strict';

// Two ways a pushfwd lost its bijection metadata.
//
// 1. A pushfwd inside a kernel body has no derivation of its own, so the
//    by-name expansion that attaches f_inv and logvolume never ran for it.
// 2. §04: a name bound to a callable is that callable. An alias in a
//    bijection slot, or in pushfwd's f slot, must resolve like its target.
//
// §06: log densityof(pushfwd(f, M), y)
//        = log densityof(M, f^{-1}(y)) - logvolume(f^{-1}(y)).
// Oracle values come from scipy 1.18.0, not from this engine: the doubling
// map on Normal(mu, 2) is Normal(2 mu, 4), and exp on Normal(mu, 1.5) is
// lognorm(s = 1.5, scale = exp(mu)).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0x5EED1E55;

function makeCtx(source: string, sampleCount: number) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: string) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount,
    rootSeed: ROOT_SEED,
  };
  return ctx;
}

// Each case is [point expression, expected log density].
async function assertScores(model: string, cases: [string, number][]) {
  const lines = cases.map(([expr], i) => `lp${i} = ${expr}`);
  const ctx = makeCtx(`${model}\n${lines.join('\n')}\n`, 4);
  for (let i = 0; i < cases.length; i++) {
    const [expr, expected] = cases[i];
    const lp = await ctx.getMeasure(`lp${i}`);
    for (const got of lp.samples) {
      assert.ok(Math.abs(got - expected) < 1e-12, `${expr}: got ${got}, expected ${expected}`);
    }
  }
}

// scipy: stats.norm(2 mu, 4).logpdf(y), as [mu, y, logpdf].
const DOUBLED_NORMAL: [number, number, number][] = [
  [1.0, 3.0, -2.336482894324563],
  [1.0, -1.0, -2.586482894324563],
  [-0.5, 3.0, -2.805232894324563],
  [-0.5, 5.5, -3.625545394324563],
  [2.0, -1.0, -3.086482894324563],
  [2.0, 5.5, -2.375545394324563],
];
// scipy: stats.norm(2 mu, 4).logpdf([3, 1, -2]).sum(), as [mu, logpdf].
const DOUBLED_IID: [number, number][] = [
  [1.0, -7.478198682973689],
  [-0.5, -7.571948682973689],
];
// scipy: stats.lognorm(s=1.5, scale=exp(mu)).logpdf(y), as [mu, y, logpdf].
const LOGNORMAL: [number, number, number][] = [
  [0.5, 0.3, -0.7656582409607664],
  [0.5, 2.0, -2.025841007063506],
  [0.5, 7.5, -3.8492913644435633],
  [-1.0, 0.3, -0.1296763714101421],
  [-1.0, 2.0, -2.6546057941034693],
  [-1.0, 7.5, -5.35922671147174],
];
const LOGNORMAL_AT_HALF = LOGNORMAL.filter(([mu]) => mu === 0.5);
// scipy: stats.lognorm(s=1.5, scale=exp(0.5)).ppf(q)
const LOGNORMAL_QUANTILES: [number, number][] = [
  [0.10, 0.2411521183071379],
  [0.25, 0.5994548468931621],
  [0.50, 1.6487212707001282],
  [0.75, 4.534589790285759],
  [0.90, 11.272062827152808],
];

const likelihoodAt = (y: string, mu: number) =>
  `logdensityof(likelihoodof(K, ${y}), record(mu = ${mu}))`;

// ---- 1. pushfwd inside a kernel body -------------------------------------

const DOUBLING: [string, string][] = [
  ['named bijection',
    'mu = elementof(reals)\nb = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\n'
    + 'K = functionof(pushfwd(b, Normal(mu, 2.0)), mu = mu)'],
  ['inline bijection',
    'mu = elementof(reals)\n'
    + 'K = functionof(pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0)), Normal(mu, 2.0)), mu = mu)'],
  ['unannotated affine fn',
    'mu = elementof(reals)\nK = functionof(pushfwd(fn(2.0 * _), Normal(mu, 2.0)), mu = mu)'],
  ['lambda kernel',
    'b = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\n'
    + 'K = mu -> pushfwd(b, Normal(mu, 2.0))'],
  ['named function kernel',
    'b = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\n'
    + 'K(mu) = pushfwd(b, Normal(mu, 2.0))'],
  ['functionof of a named pushfwd measure',
    'mu = elementof(reals)\nb = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\n'
    + 'M = pushfwd(b, Normal(mu, 2.0))\nK = functionof(M, mu = mu)'],
  ['kernelof of a pushfwd draw',
    'mu = elementof(reals)\nb = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\n'
    + 'y ~ pushfwd(b, Normal(mu, 2.0))\nK = kernelof(y, mu = mu)'],
];

for (const [label, model] of DOUBLING) {
  test(`kernel body pushfwd, ${label}, scalar base, scored through likelihoodof`, async () => {
    await assertScores(model, DOUBLED_NORMAL.map(([mu, y, lp]) =>
      [likelihoodAt(String(y), mu), lp] as [string, number]));
  });
}

// §06 over an array variate: f and f_inv map the whole variate, and the
// logvolume is the one log-volume element of the whole map (3 log 2).
const DOUBLING_IID: [string, string][] = [
  ['named bijection',
    'mu = elementof(reals)\nb = bijection(fn(2.0 * _), fn(_ / 2.0), 3.0 * log(2.0))\n'
    + 'K = functionof(pushfwd(b, iid(Normal(mu, 2.0), 3)), mu = mu)'],
  ['inline bijection',
    'mu = elementof(reals)\n'
    + 'K = functionof(pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), 3.0 * log(2.0)), '
    + 'iid(Normal(mu, 2.0), 3)), mu = mu)'],
  ['unannotated affine fn',
    'mu = elementof(reals)\nK = functionof(pushfwd(fn(2.0 * _), iid(Normal(mu, 2.0), 3)), mu = mu)'],
];

for (const [label, model] of DOUBLING_IID) {
  test(`kernel body pushfwd, ${label}, iid vector base, scored through likelihoodof`, async () => {
    await assertScores(model, DOUBLED_IID.map(([mu, lp]) =>
      [likelihoodAt('[3.0, 1.0, -2.0]', mu), lp] as [string, number]));
  });
}

// ---- 2. a callable alias in a bijection slot or in pushfwd's f slot --------

const LN_BASE = 'Normal(0.5, 1.5)';
const ALIASES: [string, string][] = [
  ['an alias in the f slot', `h = exp\nb = bijection(h, log, identity)\nLN = pushfwd(b, ${LN_BASE})`],
  ['an alias in the f_inv slot', `h = log\nb = bijection(exp, h, identity)\nLN = pushfwd(b, ${LN_BASE})`],
  ['an alias in the logvolume slot', `h = identity\nb = bijection(exp, log, h)\nLN = pushfwd(b, ${LN_BASE})`],
  ['an alias chain in the logvolume slot',
    `h = identity\nh2 = h\nb = bijection(exp, log, h2)\nLN = pushfwd(b, ${LN_BASE})`],
  ['an alias of a user fn in the f slot',
    `g = fn(exp(_))\nh = g\nb = bijection(h, log, identity)\nLN = pushfwd(b, ${LN_BASE})`],
  ['an alias of a user fn in the f_inv slot',
    `g = fn(log(_))\nh = g\nb = bijection(exp, h, identity)\nLN = pushfwd(b, ${LN_BASE})`],
  ['an alias of a user fn in the logvolume slot',
    `g = fn(_)\nh = g\nb = bijection(exp, log, h)\nLN = pushfwd(b, ${LN_BASE})`],
  ['an alias declared after the bijection',
    `b = bijection(exp, h, identity)\nh = log\nLN = pushfwd(b, ${LN_BASE})`],
  ['a builtin alias as the pushfwd f', `h = exp\nLN = pushfwd(h, ${LN_BASE})`],
  ['a builtin alias chain as the pushfwd f', `h = exp\nh2 = h\nLN = pushfwd(h2, ${LN_BASE})`],
];

// On main each form failed density, sampling or both, so one test checks both.
for (const [label, model] of ALIASES) {
  test(`pushfwd with ${label}: logdensityof and samples match LogNormal`, async () => {
    await assertScores(model, LOGNORMAL_AT_HALF.map(([, y, lp]) =>
      [`logdensityof(LN, ${y})`, lp] as [string, number]));
    const n = 8192;
    const ctx = makeCtx(`${model}\n`, n);
    const m = await ctx.getMeasure('LN');
    assert.equal(m.samples.length, n);
    for (const [q, x] of LOGNORMAL_QUANTILES) {
      let below = 0;
      for (const s of m.samples) if (s <= x) below++;
      // Binomial sd at n = 8192 is at most 0.0055; 0.025 is about 4.5 sd.
      assert.ok(Math.abs(below / n - q) < 0.025, `F(${x}) = ${below / n}, expected ${q}`);
    }
  });
}

const KERNEL_ALIASES: [string, string][] = [
  ['an alias in a bijection slot',
    'mu = elementof(reals)\nh = identity\nb = bijection(exp, log, h)\n'
    + 'K = functionof(pushfwd(b, Normal(mu, 1.5)), mu = mu)'],
  ['a builtin alias as the pushfwd f',
    'mu = elementof(reals)\nh = exp\nK = functionof(pushfwd(h, Normal(mu, 1.5)), mu = mu)'],
];

for (const [label, model] of KERNEL_ALIASES) {
  test(`kernel body pushfwd with ${label}, scored through likelihoodof`, async () => {
    await assertScores(model, LOGNORMAL.map(([mu, y, lp]) =>
      [likelihoodAt(String(y), mu), lp] as [string, number]));
  });

  test(`kernel body pushfwd with ${label}, scored through a kernel application`, async () => {
    await assertScores(model, LOGNORMAL.map(([mu, y, lp]) =>
      [`logdensityof(K(mu = ${mu}), ${y})`, lp] as [string, number]));
  });
}
