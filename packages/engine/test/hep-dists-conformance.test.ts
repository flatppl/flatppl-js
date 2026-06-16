'use strict';

// Numerical conformance for the spec §09 particle-physics distribution
// densities, scored through the engine via module-qualified heads
// (`hepphys.<Dist>.(…)`) — the form the HS3/pyhf importer emits. Each
// distribution is given one free `elementof` param (fed from θ) with the
// rest literal, scored at fixed observation points.
//
// Oracle: scipy.special / scipy.stats + numerical quadrature of the spec's
// unnormalized density (independent of the engine's analytic normalization).
// Values reproduced to ≲1e-12; the JS logpdf matched the oracle to ≤9e-16.

const test = require('node:test');
const assert = require('node:assert');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const HEAD = 'hepphys = standard_module("particle-physics", "0.1")\n';
const TOL = 1e-9;

function score(src: string, target: string) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 8,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    getMeasure: (n: string) => { if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m; },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx.getMeasure(target).then((m: any) => m.samples[0]);
}

// Score `hepphys.<dist>.(<free>, <literals…>)` at observation `obs`, feeding
// the free leading param `pname = pval`.
async function scoreDist(dist: string, pname: string, pval: number, litArgs: string, obs: number) {
  const src = `${HEAD}${pname} = elementof(reals)
d = hepphys.${dist}.(${pname}${litArgs})
L = likelihoodof(d, ${obs})
ld = logdensityof(L, record(${pname} = ${pval}))
`;
  return score(src, 'ld');
}

async function checkAll(dist: string, pname: string, pval: number, litArgs: string,
                        cases: Array<[number, number]>) {
  for (const [obs, expected] of cases) {
    const got = await scoreDist(dist, pname, pval, litArgs, obs);
    assert.ok(Math.abs(got - expected) < TOL,
      `${dist} @x=${obs}: got ${got}, want ${expected} (Δ=${Math.abs(got - expected)})`);
  }
}

test('CrystalBall(m0,sigma,alpha,n) density', () =>
  checkAll('CrystalBall', 'm0', 0.0, ', 1.0, 1.5, 2.5', [
    [-3.0, -3.7228466427263354], [0.5, -1.1182119272953486], [2.0, -2.9932119272953486]]));

test('DoubleSidedCrystalBall density', () =>
  checkAll('DoubleSidedCrystalBall', 'm0', 0.0, ', 1.0, 1.5, 1.2, 3.0, 2.0, 4.0', [
    [-2.0, -2.8065391819655585], [0.3, -1.2736439721707202], [4.0, -4.404372261977843]]));

test('Argus(resonance,slope,power) density', () =>
  checkAll('Argus', 'resonance', 10.0, ', -5.0, 0.5', [
    [3.0, -4.85688630064276], [7.0, -2.299105377151818], [9.5, -0.7585029014579211]]));

test('RelativisticBreitWigner(mean,width) density', () =>
  checkAll('RelativisticBreitWigner', 'mean', 5.0, ', 1.5', [
    [3.0, -2.5388453168045753], [5.0, -0.8248472088834178], [7.0, -3.244326053348963]]));

test('BifurcatedNormal(mean,sigmaL,sigmaR) density', () =>
  checkAll('BifurcatedNormal', 'mean', 0.2, ', 1.0, 2.0', [
    [-1.0, -2.0444036413128375], [1.3, -1.4756536413128374], [3.0, -2.3044036413128373]]));

test('Voigtian(mean,width,sigma) density', () =>
  checkAll('Voigtian', 'mean', 0.0, ', 2.0, 1.5', [
    [0.0, -1.7854078235037398], [1.0, -1.9173918837847361], [3.0, -2.850585580865787]]));
