'use strict';

// =====================================================================
// continued-poisson-support.test.ts
// =====================================================================
//
// §09's distribution table gives `ContinuedPoisson(rate)` the domain `reals`
// and the support `nonnegreals`, and states the density
//
//     λ^x e^{-λ} / Γ(x+1)   for x ≥ 0
//
// §08 "Variate domain and support" makes the domain "the set over which
// density evaluation is defined (returning 0 outside the support)" and repeats
// it: "Density formulas below specify the value on the support only; outside
// the support the density is zero." So x < 0 scores −∞.
//
// THE DEFECT. The REGISTRY entry carried the bare formula with no support
// mask, so every non-integer x < 0 returned a FINITE number: at λ = 3,
// −4.121671087258755 at x = −0.5 and −5.51 at x = −1.5, where the spec value
// is −∞. A pyhf `shapesys` aux term at a negative γ therefore had a finite
// log-density and nothing excluded that half-space. The negative INTEGERS came
// out right by accident — `gammaln` has a pole at each of them.
//
// THE ORACLE IS THE SPEC RULE plus the closed form above, evaluated here.
// Nothing in flatppl-rust computes a density, so it is not consulted.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');
const sampler = require('../sampler.ts');

const LOGPDF = sampler._internal.REGISTRY.ContinuedPoisson.logpdfFn;

function score(src: string, target: string): Promise<number> {
  const ps = processSource(src);
  const errs = ps.diagnostics.filter((d: any) => d.severity === 'error');
  assert.deepEqual(errs.map((d: any) => d.message), []);
  const built = orchestrator.buildDerivations(ps.bindings);
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 4,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    moduleRegistry: ps.loweredModule && ps.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx.getMeasure(target).then((m: any) => m.samples[0]);
}

// =====================================================================
// The mask
// =====================================================================

test('the density is −∞ below zero, in and out of the poles', () => {
  // The two points the pre-fix formula got WRONG: it returned a finite value at
  // both, 2.7 and 4.1 nats below the density at x = 2.5 but finite all the same.
  assert.equal(LOGPDF(-0.5, 3.0), -Infinity, 'x = −0.5 is outside nonnegreals');
  assert.equal(LOGPDF(-1.5, 3.0), -Infinity, 'x = −1.5 is outside nonnegreals');
  // The negative integers, where `gammaln`'s pole gave the right answer for the
  // wrong reason. Pinned so a future `gammaln` swap cannot silently undo them.
  assert.equal(LOGPDF(-1, 3.0), -Infinity, 'x = −1');
  assert.equal(LOGPDF(-2, 3.0), -Infinity, 'x = −2');
  // Just below the boundary. §09's rule is x ≥ 0, so the smallest negative
  // double is out and 0 itself is in.
  assert.equal(LOGPDF(-Number.MIN_VALUE, 3.0), -Infinity, 'x = −5e-324');
  assert.equal(LOGPDF(-Infinity, 3.0), -Infinity, 'x = −∞');
  // The mask is written so a NaN variate is refused, not passed to the formula.
  assert.equal(LOGPDF(NaN, 3.0), -Infinity, 'x = NaN');
});

test('the density on the support is unchanged', () => {
  // λ^x e^{-λ}/Γ(x+1) at λ = 3, in logs — mpmath, 50 digits.
  //   x = 0    → −λ                      = −3
  //   x = 2.5  → 2.5·ln 3 − 3 − lnΓ(3.5) = −1.454442880676800
  //   x = 4    → 4·ln 3 − 3 − ln 4!      = −1.783604675675507
  assert.equal(LOGPDF(0, 3.0), -3, 'x = 0 is IN nonnegreals');
  assert.ok(Math.abs(LOGPDF(2.5, 3.0) - (-1.4544428806767999963)) < 1e-14,
    `x = 2.5 gave ${LOGPDF(2.5, 3.0)}`);
  // At a non-negative INTEGER the continuous extension must agree with
  // Poisson's log-pmf, which §09 states outright: lnΓ(x+1) = ln(x!) there.
  const poisson = sampler._internal.REGISTRY.Poisson.logpdfFn;
  assert.ok(Math.abs(LOGPDF(4, 3.0) - poisson(4, 3.0)) < 1e-14,
    `x = 4: continued ${LOGPDF(4, 3.0)} vs Poisson ${poisson(4, 3.0)}`);
  assert.ok(Math.abs(LOGPDF(4, 3.0) - (-1.7836046756755068541)) < 1e-14,
    `x = 4 gave ${LOGPDF(4, 3.0)}`);
});

// =====================================================================
// Through the model surface
// =====================================================================

const AUX = (obs: string) => `
hepphys = standard_module("particle-physics", "0.1")
g = elementof(cartpow(posreals, 2))
aux_model = hepphys.ContinuedPoisson.(g .* [3.0, 3.0])
L = likelihoodof(aux_model, ${obs})
ld = logdensityof(L, record(g = [1.0, 1.0]))
`;

test('a §09 aux term scores −∞ at a negative observation', async () => {
  // One in-support and one out-of-support entry. The sum is −∞, so the negative
  // half-space is excluded; before the mask the −0.5 entry contributed
  // −4.121671087258755 and the pair summed to a finite −5.576113967935555.
  const mixed = await score(AUX('[2.5, -0.5]'), 'ld');
  assert.equal(mixed, -Infinity, `got ${mixed}`);
  // Both in support: 2 × logpdf(2.5; 3), so the mask has not moved the support.
  const inside = await score(AUX('[2.5, 2.5]'), 'ld');
  const want = 2 * -1.4544428806767999963;
  assert.ok(Math.abs(inside - want) < 1e-12, `got ${inside}, closed form ${want}`);
});
