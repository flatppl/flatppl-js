'use strict';

// Regression: density of a joint/record measure where a field's
// distribution parameter is a DERIVED binding of an earlier field.
//
// Spec 06-measure-algebra.md → "joint chain": the joint density is the
// product of the constituent CONDITIONAL densities, and each conditional
// must be evaluated at the point's conditioning values. For
//   p ~ Uniform(0,1); a = p * 10; x ~ Beta(a, 2)
// the joint log-density at (p=0.3, x=0.5) must use a = 0.3*10 = 3, i.e.
//   logpdf(Uniform(0,1), 0.3) + logpdf(Beta(3, 2), 0.5) = 0 + log(1.5).
//
// Bug (pre-fix): walkJointFieldsOrPositional threads only sibling FIELDS
// into the overlay, so the derived binding `a` was resolved from baseEnv's
// stale module value (computed with a non-point `p`) instead of from the
// point. A direct field-ref param (Normal(m, 1)) was already correct.
//
// Oracle (INDEPENDENT — Distributions.jl):
//   logpdf(Beta(3,2), 0.5)            = 0.4054651081081646
//   logpdf(Normal(0,1),0.3)+logpdf(Normal(0.3,1),0.5) = -1.9028770664093457

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0xB1737CFC;

function scalarOf(src: any, name: any) {
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
    sampleCount: 8,
    rootSeed:    ROOT_SEED,
  };
  return ctx.getMeasure(name).then((v: any) => v.samples[0]);
}

test('joint density: derived-binding distribution parameter is scored at the point', async () => {
  const lp = await scalarOf(`
p ~ Uniform(interval(0.0, 1.0))
a = p * 10.0
x ~ Beta(a, 2.0)
prior = lawof(record(p = p, x = x))
lp = logdensityof(prior, record(p = 0.3, x = 0.5))
`, 'lp');
  // 0 (Uniform) + logpdf(Beta(3,2), 0.5)
  assert.ok(Math.abs(lp - 0.4054651081081646) < 1e-12,
    `got ${lp}, expected 0.4054651081081646`);
});

test('joint density: direct field-ref distribution parameter (control)', async () => {
  const lp = await scalarOf(`
m ~ Normal(0.0, 1.0)
x ~ Normal(m, 1.0)
prior = lawof(record(m = m, x = x))
lp = logdensityof(prior, record(m = 0.3, x = 0.5))
`, 'lp');
  assert.ok(Math.abs(lp - (-1.9028770664093457)) < 1e-12,
    `got ${lp}, expected -1.9028770664093457`);
});
