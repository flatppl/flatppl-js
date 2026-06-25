'use strict';
// Task 4 (U3): the module registry must reach the LOCAL quadrature env used to
// resolve a support-restricted normalizer Z = ∫_S w(x) dx for a
// `normalize(truncate(weighted(w, Lebesgue), S))` base — the shape the
// HS3/pyhf converter emits for a generic_dist chebychev density (rf207).
//
// The normalizer is evaluated by mat-density's `weightedBaseLogMass` via
// `sampler.evaluateExpr(weightFn.body, env)` over a midpoint quadrature, NOT
// through the worker session. So pushModuleRegistry (which threads the worker
// env) does NOT cover it: the weight `w(x) = poly.chebyshev(k, x)` is a
// cross-module call that evaluateExpr resolves through env.__moduleRegistry.
// Before the fix the local quadrature env lacked __moduleRegistry and the eval
// threw "cross-module ref 'poly.chebyshev' has no entry in env.__moduleRegistry".
// The fix threads ctx.moduleRegistry into that local env.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const SEED = 0xBA5E;

function ctxFor(src: string, N: number) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: N,
    rootKey: SEED,
    rootSeed: SEED,
    marginalizationCount: 64,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx);
      cache.set(n, m);
      return m;
    },
    sendWorker: (m: any) => {
      const r = w.handle(m);
      return r && r.type === 'error'
        ? Promise.reject(new Error(r.message))
        : Promise.resolve(r);
    },
  };
  return ctx;
}

// Model mirrors the converter's generic_dist chebychev shape: a
// normalize(truncate(weighted(x -> poly.chebyshev(k, x), Lebesgue(reals)), S))
// density scored through an iid likelihood.  poly.chebyshev(1, x) = T_1(x) = x.
// alpha is the free elementof param the likelihood_density derivation requires.
// Oracle (closed-form, independent): on S = [1, 5] the unnormalized density is
// w(x) = T_1(x) = x; Z = ∫₁⁵ x dx = 12; pdf(x) = x/12; iid log-score over
// obs = [2, 3, 4] is log(2/12) + log(3/12) + log(4/12) = -4.276666119016055.
const SRC = `
poly = standard_module("polynomials", "0.1")
alpha ~ Normal(0, 1)
cb = normalize(truncate(weighted(x -> poly.chebyshev(1, x), Lebesgue(reals)), interval(1.0, 5.0)))
obs ~ iid(cb, 3)
fk = kernelof(record(obs = obs), alpha = alpha)
observed_data = [2.0, 3.0, 4.0]
L = likelihoodof(fk, record(obs = observed_data))
ld = logdensityof(L, 0.5)
`;

test('module registry reaches the truncate-normalizer quadrature env (poly.chebyshev resolves)', async () => {
  const ctx = ctxFor(SRC, 1);
  assert.equal(ctx.derivations['ld'].kind, 'likelihood_density',
    'ld should be a likelihood_density derivation');
  const m = await ctx.getMeasure('ld');
  const s: Float64Array | null = m.samples ?? (m.value && m.value.data) ?? null;
  assert.ok(s && s.length > 0 && Number.isFinite(s[0]),
    `chebyshev normalize(truncate(...)) likelihood did not resolve to a finite value; got: ${s}`);
  const v = (s as Float64Array)[0];
  // Closed-form: T_1(x)=x, Z=∫₁⁵ x dx=12, ld = Σ log(x_i/12) over [2,3,4].
  const oracle = Math.log(2 / 12) + Math.log(3 / 12) + Math.log(4 / 12);
  assert.ok(Math.abs(v - oracle) < 1e-6,
    `ld ${v} should match the closed-form oracle ${oracle} (Δ ${Math.abs(v - oracle)})`);
});
