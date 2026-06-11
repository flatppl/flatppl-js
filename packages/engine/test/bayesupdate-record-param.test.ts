'use strict';

// Boundary-feeding, record-param case (audit §3 / H1 + the per-atom MC
// marginal). A reified GENERATIVE kernel whose SINGLE parametric input is a
// whole record (`pars`), scored against a fixed prior — the transport model.
// Two things must hold and were both broken before this fix:
//
//   1. matBayesupdate must FEED the whole prior record onto the kernel's
//      `pars` boundary (per-atom record objects), under BOTH the kwarg name
//      (`pars`) and the %local placeholder the lowering introduces (`_pars_`)
//      — not re-materialise a module binding via getMeasure (which fails:
//      `pars` is an `elementof` free input with no derivation).
//   2. walkMcMarginal must run a PER-ATOM Monte-Carlo marginal (each prior
//      particle carries its own `pars`), not the atom-independent profile
//      estimate — and not throw "per-atom marginal deferred".
//
// Correctness is asserted the sampling≡density way: DATA below was sampled by
// the engine from the model at a=0.1, b=0.3, mu=1.1 (the model's glob_pars).
// A correct likelihood must PEAK at those generating params. We evaluate the
// MC-marginal log-likelihood on 1-D grid slices through the truth and assert
// the maximum sits at the true value (the support is sharp because
// x ~ Normal(mu, 0.2) is narrow, so off-truth params drive the data outside
// the achievable range of z → -inf, the genuine support boundary).

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const der = require(ENG + 'derivations.ts');
const mcRecipe = require(ENG + 'mc-recipe.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

// 40 z-observations sampled by the engine from the transport model at
// (a=0.1, b=0.3, mu=1.1), sigma=0.2 (deterministic — Philox seed 99).
const DATA = [
  2.959, 1.948, 1.904, 3.198, 4.169, 1.986, 2.005, 1.69, 2.896, 2.669,
  7.04, 2.086, 5.154, 1.918, 9.72, 4.562, 2.892, 2.004, 4.078, 0.374,
  4.356, 4.318, 0.359, 3.48, 1.718, 6.558, 2.063, 1.251, 2.322, 1.015,
  3.097, 3.477, 0.256, 5.208, 1.276, 1.513, 2.814, 6.964, 3.647, 5.032,
];

function transportSource(): string {
  return `
sigma = 0.2
pars = elementof(cartprod(a = reals, b = reals, mu = reals))
generator_dist = Normal(pars.mu, sigma)
x ~ generator_dist
a, b = (pars.a, pars.b)
delta_alpha = (2 * draw(Uniform(interval(0, 1))) + 1 ) * a
y = (x + delta_alpha)^3 * exp(x - b)
z = post(y)
generator = kernelof(x, pars = pars)
transport = kernelof(y, x = x, pars = pars)
post = x -> x/2
n = elementof(posintegers)
xs ~ iid(generator(pars), n)
ys ~ transport.(xs, [pars])
zs = post.(ys)
k_model_n = kernelof(zs, n = n, pars = pars)
data = [${DATA.join(', ')}]
k_model = pars -> k_model_n(lengthof(data), pars)
L = likelihoodof(k_model, data)
prior = record(a = Exponential(0.1), b = Exponential(0.3), mu = Exponential(1.1))
posterior = bayesupdate(L, prior)
`;
}

function buildClassified(src: string) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  for (const nm of ['posterior', 'L']) {
    if (built.bindings.has(nm) && !built.derivations[nm]) {
      const c = der.classifyDerivation(built.bindings.get(nm), built.bindings);
      if (c) built.derivations[nm] = c;
    }
  }
  return built;
}

// Evaluate the MC-marginal log-likelihood of DATA at a list of `pars` records,
// one logL per record — the kernel boundary fed exactly as matBayesupdate
// feeds it (under both `pars` and `_pars_`).
function gridLogL(grid: any[]): Float64Array {
  const built = buildClassified(transportSource());
  const bindings = built.bindings;
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 7 });
  w.handle({ type: 'setEnv', env: { sigma: 0.2 } });   // fixed-phase const
  const expand = (name: string) => orchestrator.expandMeasure(name,
    { derivations: built.derivations, bindings });
  const p = built.derivations['posterior'];
  const bodyIR = orchestrator.expandMeasure(p.bodyIR,
    { derivations: built.derivations, bindings });
  const densIR = mcRecipe.buildMcMarginalForm(bodyIR, bindings, expand) || bodyIR;
  const observed = orchestrator.resolveIRToValue(p.obsIR, bindings, new Map());
  const reply = w.handle({
    type: 'logDensityN', ir: densIR, count: grid.length,
    refArrays: { pars: grid, _pars_: grid },
    observed, tally: 'clamped',
    mcMarginalizationCount: 512, mcSeed: 12345,
  });
  if (reply.type === 'error') throw new Error('logDensityN: ' + reply.message);
  return reply.samples;
}

function argmax(a: Float64Array): number {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

test('record-param likelihood peaks at the generating params (sampling≡density)', () => {
  // `mu` slice — x ~ Normal(mu, 0.2) makes mu razor-sharp: only the truth is
  // feasible for ALL 40 data points; the rest fall outside z's support → -inf.
  const muVals = [0.3, 0.7, 0.9, 1.1, 1.3, 1.7, 2.5];
  const muLL = gridLogL(muVals.map((mu) => ({ a: 0.1, b: 0.3, mu })));
  assert.strictEqual(muVals[argmax(muLL)], 1.1,
    `mu likelihood peaks at ${muVals[argmax(muLL)]}, not the truth 1.1`);
  assert.ok(Number.isFinite(muLL[3]), 'logL at the true mu must be finite');

  // `a` slice — the latent fibre tightens as a grows; the truth a=0.1 is the
  // largest feasible (a≥0.2 puts the data outside support here).
  const aVals = [0.05, 0.1, 0.2, 0.5, 1, 2];
  const aLL = gridLogL(aVals.map((a) => ({ a, b: 0.3, mu: 1.1 })));
  assert.strictEqual(aVals[argmax(aLL)], 0.1,
    `a likelihood peaks at ${aVals[argmax(aLL)]}, not the truth 0.1`);

  // `b` slice — b enters through exp(x − b); a shallow ridge near the truth.
  // The truth b=0.3 must be FINITE and within a couple of nats of the slice
  // max, and large b must be excluded.
  const bVals = [0.1, 0.3, 0.5, 1, 2, 5];
  const bLL = gridLogL(bVals.map((b) => ({ a: 0.1, b, mu: 1.1 })));
  assert.ok(Number.isFinite(bLL[1]), 'logL at the true b must be finite');
  assert.ok(bLL[1] >= bLL[argmax(bLL)] - 2.0,
    `logL at true b=0.3 (${bLL[1].toFixed(2)}) is far below the slice max ${bLL[argmax(bLL)].toFixed(2)}`);
  assert.ok(!Number.isFinite(bLL[3]) || bLL[3] < bLL[1],
    'large b should be excluded / lower than the truth');
});

test('record-param posterior materialises end-to-end with per-atom MC marginal', async () => {
  // Integration smoke: the whole bayesupdate path runs and yields a record
  // posterior over a/b/mu with finite mass and a positive effective sample
  // size. (The IS efficiency is low — the likelihood support is sharp — so we
  // assert structure, not the noisy posterior mean; correctness is the grid
  // test above.)
  const built = buildClassified(transportSource());
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 4242 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 400, rootKey: 4242,
    marginalizationCount: 32,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  const post = await ctx.getMeasure('posterior');
  assert.ok(post && post.fields, 'posterior must be a record measure');
  assert.deepStrictEqual(Object.keys(post.fields).sort(), ['a', 'b', 'mu']);
  assert.ok(Number.isFinite(post.logTotalmass), 'posterior logTotalmass must be finite');
  assert.ok(post.n_eff > 1, `posterior n_eff ${post.n_eff} must exceed 1`);
});
