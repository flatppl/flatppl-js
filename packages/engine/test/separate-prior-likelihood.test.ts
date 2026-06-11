'use strict';

// The "prior and likelihood defined fully separately" idiom
// (flatppl-examples/bayesian_inference_1): the parameters are `elementof`
// BOUNDARIES (not draws), the prior is a `joint(...)` of independent priors
// over them, and the forward model reaches the observation distribution's
// params through DERIVED value bindings. Three engine gaps this surfaced, all
// fixed and pinned here:
//
//   1. forward_kernel sampling: `obs ~ M` lowers to `obs = draw(M)`; inlined
//      into the kernel body's record field that `draw` reached the sampler as
//      an unknown distribution. Now materialiseMeasureIR peels draw(M)→M.
//   2. likelihood density: the same surviving `draw` hit the density walker
//      ("unsupported measure op 'draw'") → an empty curve. walkAcc now peels
//      draw/lawof.
//   3. posterior: (a) the kernel's dist-params reach the boundaries through
//      derived value bindings (`a = 5*theta2`) the density side would
//      getMeasure → "no derivation for 'a'"; matBayesupdate now inlines them
//      (inlineBoundaryDerivations). (b) the whole posterior was cascade-pruned
//      because its body's parameterized internals weren't "resolvable" →
//      "Not plottable"; derivationRefsValid now skips parameterized/stochastic
//      kernel internals for bayesupdate.

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

// data generated from mu≈3.6 — recovered by 5*theta2 at theta2≈0.72.
const DATA = [1.2, 3.4, 5.1, 2.8, 4.0, 3.7, 5.5, 2.1, 4.3, 3.9];

function model(): string {
  return `
theta1 = elementof(reals)
theta2 = elementof(reals)
theta1_dist = Normal(0, 1)
theta2_dist = Exponential(1)
prior = joint(theta1 = theta1_dist, theta2 = theta2_dist)
c = 5
f_a = par -> c * par
f_b = fn(abs(_) * _)
a = f_a(theta2)
b = f_b(theta1, theta2)
obs ~ iid(Normal(mu = a, sigma = b), 10)
forward_kernel = kernelof(record(obs = obs))
observed_data = [${DATA.join(', ')}]
L = likelihoodof(forward_kernel, record(obs = observed_data))
posterior = bayesupdate(L, prior)
`;
}

function build() {
  return orchestrator.buildDerivations(processSource(model()).bindings);
}

function makeCtx(built: any, N: number) {
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 7 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N, rootKey: 7,
    rootSeed: 7, marginalizationCount: 32,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return { ctx, w };
}

test('separate prior/likelihood: forward_kernel samples (draw-of-measure peel)', async () => {
  const built = build();
  const { ctx } = makeCtx(built, 4000);
  const sig = orchestrator.signatureOf('forward_kernel', built.bindings, built.derivations);
  const paramNames = sig.inputs.map((i: any) => i.paramName);
  const env: Record<string, number> = {};
  for (const i of sig.inputs) env[i.paramName] = (i.kwargName === 'theta2' ? 1.0 : 0.5);
  const clm = require('../clm.ts');
  const lowCtx = { derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map() };
  const node = clm.lowerMeasure(sig.body, lowCtx, { boundaries: env });
  const ir = orchestrator.substituteBoundaryValues(node.body, env);
  const m = await materialiser.materialiseMeasureIR(ir, ctx);
  const f = m.fields ? m.fields.obs : m;
  const s = f.samples || (f.value && f.value.data);
  let mean = 0; const n = Math.min(s.length, 4000);
  for (let i = 0; i < n; i++) mean += s[i];
  mean /= n;
  // theta2=1 ⇒ a=5*1=5 ⇒ obs ~ Normal(5, b=abs(0.5)*1=0.5); mean ≈ 5.
  assert.ok(mean > 4.5 && mean < 5.5, `forward_kernel obs mean ${mean.toFixed(3)} should be ~5`);
});

test('separate prior/likelihood: likelihood density is finite and peaks near the MLE', async () => {
  const built = build();
  const { w } = makeCtx(built, 1);
  const sig = orchestrator.signatureOf('forward_kernel', built.bindings, built.derivations);
  const clm = require('../clm.ts');
  const boundaries: Record<string, any> = {};
  for (const i of sig.inputs) boundaries[i.paramName] = true;
  const node = clm.lowerMeasure(sig.body, { derivations: built.derivations,
    bindings: built.bindings, fixedValues: built.fixedValues || new Map() },
    { boundaries });
  const ir = node.body;   // boundary refs resolve by NAME via worker env
  const observed = { obs: DATA };
  const logL = (t1: number, t2: number) => {
    w.handle({ type: 'setEnv', env: { theta1: t1, theta2: t2 } });
    const r = w.handle({ type: 'logDensityN', ir, observed, count: 1, tally: 'clamped' });
    assert.notStrictEqual(r.type, 'error', r.message);
    return r.samples[0];
  };
  // mu=5*theta2; data mean ~3.6 ⇒ theta2≈0.72. sigma=|theta1|*theta2; data
  // sd ~1.2 ⇒ |theta1|≈1.6. All finite; the near-MLE point beats off-MLE ones.
  const atMLE = logL(1.6, 0.72);
  assert.ok(Number.isFinite(atMLE), 'likelihood at the MLE must be finite (was an empty curve)');
  assert.ok(atMLE > logL(0.5, 1.0), 'likelihood must peak nearer the MLE than at default_pars');
  assert.ok(atMLE > logL(0.5, 0.3), 'likelihood must beat a too-tight-sigma point');
});

test('separate prior/likelihood: posterior classifies (not pruned) and materialises', async () => {
  const built = build();
  // Cascade-prune must NOT drop the posterior just because its body internals
  // are parameterized (the "Not plottable" bug).
  assert.ok(built.derivations['posterior'], 'posterior must keep its derivation (else "Not plottable")');
  assert.strictEqual(built.derivations['posterior'].kind, 'bayesupdate');

  const { ctx } = makeCtx(built, 4000);
  const post = await ctx.getMeasure('posterior');
  assert.ok(post && post.fields, 'posterior is a record measure');
  assert.deepStrictEqual(Object.keys(post.fields).sort(), ['theta1', 'theta2']);
  assert.ok(post.n_eff > 1, `posterior n_eff ${post.n_eff} > 1`);
  // mu = 5*theta2 must recover the data mean (~3.6) ⇒ E_post[theta2] ≈ 0.72,
  // well below the prior mean 1.0.
  const lw = post.logWeights;
  let mx = -Infinity; for (let i = 0; i < lw.length; i++) if (lw[i] > mx) mx = lw[i];
  const s = post.fields.theta2.samples;
  let sw = 0, sxw = 0;
  for (let i = 0; i < s.length; i++) { const wt = Math.exp(lw[i] - mx); sw += wt; sxw += s[i] * wt; }
  const eTheta2 = sxw / sw;
  assert.ok(eTheta2 > 0.45 && eTheta2 < 0.95,
    `E_post[theta2] ${eTheta2.toFixed(3)} should sit near the MLE 0.72 (5*theta2 ≈ data mean 3.6)`);
});
