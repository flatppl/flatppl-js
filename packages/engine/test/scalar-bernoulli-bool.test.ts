'use strict';

// Regression: a scalar (non-broadcast) discrete likelihood scored against
// a boolean observation. `z ~ Bernoulli(t)` with a `true`/`false` obs used
// to throw "density: cannot consume scalar from value of type boolean" —
// consumeScalar handled number / Value / typed-array / array but not a
// bare JS boolean (the array branches already coerced boolean vectors via
// `+v[0]`, so broadcast `Bernoulli.(p)` worked; only the scalar leaf
// rejected a boolean). Now a boolean scalar coerces to 0/1.
//
// Oracle (closed form): logpdf(Bernoulli(0.3), true)  = log(0.3) = -1.2039728043259361
//                       logpdf(Bernoulli(0.3), false) = log(0.7) = -0.35667494393873245

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0xB001;

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
    sampleCount: 8,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

test('scalar Bernoulli logdensity consumes a boolean observation', async () => {
  const t = await makeCtx('Y = lawof(Bernoulli(0.3))\nlp = logdensityof(Y, true)\n').getMeasure('lp');
  assert.ok(Math.abs(t.samples[0] - Math.log(0.3)) < 1e-12, `true: got ${t.samples[0]}`);
  const f = await makeCtx('Y = lawof(Bernoulli(0.3))\nlp = logdensityof(Y, false)\n').getMeasure('lp');
  assert.ok(Math.abs(f.samples[0] - Math.log(0.7)) < 1e-12, `false: got ${f.samples[0]}`);
});

test('scalar Bernoulli likelihood + bayesupdate materialises with a boolean obs', async () => {
  const po = await makeCtx(`
t ~ Beta(2.0, 2.0)
prior = lawof(record(t = t))
z ~ Bernoulli(t)
forward_kernel = kernelof(record(z = z), t = t)
L = likelihoodof(forward_kernel, record(z = true))
posterior = bayesupdate(L, prior)
`).getMeasure('posterior');
  assert.ok(po && po.logWeights, 'posterior materialised');
  assert.ok(Number.isFinite(po.logTotalmass), 'finite logTotalmass');
});
