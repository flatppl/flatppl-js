'use strict';

// Regression: a likelihood kernel that binds an intermediate
// difference-of-two-gathers (`eta = theta[person] .- b[item]`) and then
// broadcasts a link + distribution over it
//   prob = invlogit.(eta); y ~ Bernoulli.(prob)
// used to fail density with "field 'y' did not fully consume its value".
//
// Root cause: the dissolver collapses `broadcast(invlogit_fn, eta)` to a
// BARE `invlogit(eta)` (invlogit is a batched-safe op). When `eta` is an
// atom-batched `[N, K]` value, the unary scalar-broadcast path
// (ARITH_OPS_N via broadcastN) assumed rank-0/rank-1 inputs and collapsed
// the inner K axis to a per-atom scalar, so the Bernoulli broadcast scored
// K=1 atoms against a length-K observation vector. The fix routes a real
// shape-rich arg of a unary primitive through the shape-preserving
// elementwise path (sampler-eval-batched: _cxBroadcast / _cxElementwise).
//
// Two checks: (1) the SEPARATE-binding form now materialises its posterior
// (the exact case that threw); (2) it scores IDENTICALLY to the INLINED
// form `prob = invlogit.(theta[person] .- b[item])`, which was the
// known-good path (its forward log-likelihood was verified against an
// independent oracle, −10.366566 for these data).

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0x5EED1234;

function posteriorWeights(probLine: any) {
  const src = `
P = 4
I = 5
person = [1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4,4]
item   = [1,2,3,4,5,1,2,3,4,5,1,2,3,4,5,1,2,3,4,5]
theta ~ iid(Normal(0.0, 1.5), P)
b ~ iid(Normal(0.0, 1.5), I)
prior = lawof(record(theta = theta, b = b))
${probLine}
y ~ Bernoulli.(prob)
forward_kernel = kernelof(record(y = y), theta = theta, b = b)
L = likelihoodof(forward_kernel, record(y = [true,false,true,false,false,true,true,true,false,false,true,false,true,true,false,true,true,true,true,true]))
posterior = bayesupdate(L, prior)
`;
  const lifted = processSource(src);
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
    sampleCount: 1000,
    rootSeed:    ROOT_SEED,
  };
  return ctx.getMeasure('posterior');
}

test('two-gather intermediate binding: separate-binding likelihood materialises', async () => {
  const po = await posteriorWeights('eta = theta[person] .- b[item]\nprob = invlogit.(eta)');
  assert.ok(po && po.logWeights, 'posterior materialised with weights');
});

test('two-gather intermediate binding: separate form scores identically to inlined', async () => {
  const sep = await posteriorWeights('eta = theta[person] .- b[item]\nprob = invlogit.(eta)');
  const inl = await posteriorWeights('prob = invlogit.(theta[person] .- b[item])');
  assert.equal(sep.logWeights.length, inl.logWeights.length);
  let compared = 0;
  for (let i = 0; i < sep.logWeights.length; i++) {
    const a = sep.logWeights[i], b = inl.logWeights[i];
    // Same seed → same prior atoms; finite entries must match exactly
    // (NaN slots, if any, occupy identical positions in both).
    if (Number.isNaN(a) && Number.isNaN(b)) continue;
    assert.ok(Math.abs(a - b) < 1e-12,
      `atom ${i}: separate ${a} vs inlined ${b}`);
    compared++;
  }
  assert.ok(compared > 0, 'compared at least one finite weight');
});
