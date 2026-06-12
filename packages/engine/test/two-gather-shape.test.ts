'use strict';

// Regression: a difference-of-two-gathers (`theta[person] .- b[item]`,
// two different-length iid sources, with no literal-length operand to
// anchor the broadcast axis) used to mis-infer its shape, with three
// downstream symptoms:
//
//   1. the gather dim was dropped in `inferGet` (a ref/named-array index
//      contributed no length), so the broadcast axis went `%dynamic` and
//      the density variate footprint resolved to a SCALAR — `logdensityof`
//      of a joint with such a field threw a type-check ("expects … y:
//      boolean, got … array of boolean (length N)") and the likelihood
//      density under-consumed its value ("field did not fully consume").
//   2. `measureN` of a record whose FIRST field is a vector atom reported
//      N*K atoms (the flattened sample buffer, ignoring `.dims`), so the
//      bayesupdate posterior's logWeights had N*K entries — N*(K-1) of them
//      NaN — and `logTotalmass` / `n_eff` came out NaN.
//
// Fixes: `inferGet` takes an array-valued selector's length as the result
// dim; `measureN` divides the sample buffer by prod(dims) to recover the
// atom count.
//
// Oracle (INDEPENDENT — Distributions.jl): the joint log-density of
//   theta ~ iid(Normal(0,1.5),4); b ~ iid(Normal(0,1.5),5)
//   y ~ Bernoulli.(invlogit.(theta[person] .- b[item]))
// at theta=[-1,-0.3,0.5,1.2], b=[-0.8,-0.2,0.3,0.9,1.5], y=<below> is
//   sum(logpdf(Normal(0,1.5), theta)) + sum(logpdf(Normal(0,1.5), b))
//     + sum(logpdf(Bernoulli(invlogit(theta[p]-b[i])), y))  = -23.755087911694748

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0x5EED1234;
const PERSON = '[1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]';
const ITEM   = '[1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]';
const YOBS   = '[true,true,false,false,false,true,true,true,false,false,true,false,true,true,false,true,true,true,true,true]';

function makeCtx(source: any) {
  const lifted = processSource(source);
  const errs = lifted.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs, [], 'diagnostics: ' + JSON.stringify(errs));
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (n: any) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, p);
      return p;
    },
    sendWorker:  (m: any) => {
      const r = worker.handle(m);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
    sampleCount: 500,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

test('two-gather joint density matches the independent oracle', async () => {
  const ctx = makeCtx(`
person = ${PERSON}
item   = ${ITEM}
theta ~ iid(Normal(0.0, 1.5), 4)
b ~ iid(Normal(0.0, 1.5), 5)
eta = theta[person] .- b[item]
prob = invlogit.(eta)
y ~ Bernoulli.(prob)
joint = lawof(record(theta = theta, b = b, y = y))
lp = logdensityof(joint, record(theta = [-1.0,-0.3,0.5,1.2], b = [-0.8,-0.2,0.3,0.9,1.5], y = ${YOBS}))
`);
  const lp = await ctx.getMeasure('lp');
  assert.ok(Math.abs(lp.samples[0] - (-23.755087911694748)) < 1e-9,
    `got ${lp.samples[0]}, expected -23.755087911694748`);
});

test('two-gather bayesupdate posterior has finite logTotalmass / n_eff (record of only vector-atom fields)', async () => {
  const ctx = makeCtx(`
person = ${PERSON}
item   = ${ITEM}
theta ~ iid(Normal(0.0, 1.5), 4)
b ~ iid(Normal(0.0, 1.5), 5)
prior = lawof(record(theta = theta, b = b))
prob = invlogit.(theta[person] .- b[item])
y ~ Bernoulli.(prob)
forward_kernel = kernelof(record(y = y), theta = theta, b = b)
L = likelihoodof(forward_kernel, record(y = ${YOBS}))
posterior = bayesupdate(L, prior)
`);
  const po = await ctx.getMeasure('posterior');
  assert.equal(po.logWeights.length, 500, 'one weight per prior atom (not N*K)');
  let nan = 0;
  for (const x of po.logWeights) if (Number.isNaN(x)) nan++;
  assert.equal(nan, 0, 'no NaN weights');
  assert.ok(Number.isFinite(po.logTotalmass), 'finite logTotalmass, got ' + po.logTotalmass);
  assert.ok(Number.isFinite(po.n_eff) && po.n_eff > 0, 'finite positive n_eff, got ' + po.n_eff);
});
