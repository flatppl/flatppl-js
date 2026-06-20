'use strict';

// =====================================================================
// optimizer-integration.test.ts
// =====================================================================
//
// End-to-end check of the "Find maximum" path on REAL models, through the
// same engine pieces the viewer's optimize-plot.ts drives: signatureOf →
// clm.lowerMeasure (free inputs unbaked) → density.logDensityN (per-θ
// params via refArrays, shared observation — the normal likelihood mode,
// NOT pointsBatched) → optimizer.optimize. Verifies the optimizer recovers
// the closed-form MLE. No DOM/worker thread — the worker's density function
// is called directly.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const eng = require('../index.ts');
const der = require('../derivations.ts');
const sampler = require('../sampler.ts');

function setup(src: string) {
  const dst = der.buildDerivations(eng.processSource(src).bindings);
  const ctx = { derivations: dst.derivations, bindings: dst.bindings, fixedValues: dst.fixedValues };
  const sig = eng.orchestrator.signatureOf('L', dst.bindings);
  const paramNames = sig.inputs.map((i: any) => i.paramName);
  const clm = eng.clm.lowerMeasure(sig.body, ctx, { boundaries: {}, freeInputs: paramNames });
  const observed = eng.orchestrator.resolveIRToValue(sig.obsIR, dst.bindings, dst.fixedValues);
  const irBaked = clm.body;
  const evalCloud = async (cloud: number[][]) => {
    const K = cloud.length;
    const refArrays: any = {};
    for (let j = 0; j < paramNames.length; j++) {
      const col = new Float64Array(K);
      for (let k = 0; k < K; k++) col[k] = cloud[k][j];
      refArrays[paramNames[j]] = col;
    }
    const logps = eng.density.logDensityN(irBaked, observed, refArrays, K, { baseEnv: {} });
    return Array.from(logps, (v: any) => (Number.isFinite(v) ? v : -Infinity));
  };
  return { sig, paramNames, evalCloud };
}

test('Find-maximum: Normal-mean likelihood MLE equals the sample mean', async () => {
  // data mean = 3.0 → MLE(mu) = 3.0
  const src = [
    'mu = elementof(reals)',
    'obs_model = iid(Normal(mu, 1.0), 5)',
    'data = [1.0, 2.0, 3.0, 4.0, 5.0]',
    'fwd = functionof(obs_model, mu = mu)',
    'L = likelihoodof(fwd, data)',
  ].join('\n');
  const { sig, evalCloud } = setup(src);
  assert.equal(sig.inputs.length, 1, 'one free input (mu)');
  const fit = await eng.optimizer.optimize({
    evalCloud, x0: [0], domains: [{ kind: 'real' }], scales: [5], opts: { seed: 1 },
  });
  assert.ok(Math.abs(fit.mode[0] - 3.0) < 0.05, `MLE mu = ${fit.mode[0]} (expect 3.0)`);
});

test('Find-maximum: Normal (mu, sigma) MLE = (mean, population std) over two axes', async () => {
  // mean = 3.0, population variance = 2.0 → sigma_MLE = sqrt(2) ≈ 1.4142
  const src = [
    'mu = elementof(reals)',
    'sigma = elementof(posreals)',
    'obs_model = iid(Normal(mu, sigma), 5)',
    'data = [1.0, 2.0, 3.0, 4.0, 5.0]',
    'fwd = functionof(obs_model, mu = mu, sigma = sigma)',
    'L = likelihoodof(fwd, data)',
  ].join('\n');
  const { sig, paramNames, evalCloud } = setup(src);
  assert.equal(sig.inputs.length, 2, 'two free inputs (mu, sigma)');
  // domains in the param order signatureOf reports
  const domByName: any = { mu: { kind: 'real' }, sigma: { kind: 'posreals' } };
  const domains = sig.inputs.map((i: any) => domByName[i.kwargName] || { kind: 'real' });
  const fit = await eng.optimizer.optimize({
    evalCloud, x0: sig.inputs.map((i: any) => (i.kwargName === 'sigma' ? 1 : 0)),
    domains, scales: sig.inputs.map((i: any) => (i.kwargName === 'sigma' ? 2 : 5)),
    opts: { seed: 2, starts: 3 },
  });
  const muIdx = sig.inputs.findIndex((i: any) => i.kwargName === 'mu');
  const sigIdx = sig.inputs.findIndex((i: any) => i.kwargName === 'sigma');
  assert.ok(Math.abs(fit.mode[muIdx] - 3.0) < 0.05, `mu = ${fit.mode[muIdx]} (expect 3.0)`);
  assert.ok(Math.abs(fit.mode[sigIdx] - Math.sqrt(2)) < 0.05,
    `sigma = ${fit.mode[sigIdx]} (expect ${Math.sqrt(2).toFixed(4)})`);
});

test('Find-maximum: function plot maximises a deterministic expression (evaluateN path)', async () => {
  // f(a,b) = -((a-2)² + (b+1)²) → argmax at (2, -1)
  const src = [
    'a = elementof(reals)',
    'b = elementof(reals)',
    'f = functionof(0 - ((a - 2.0)^2 + (b + 1.0)^2), a = a, b = b)',
  ].join('\n');
  const dst = der.buildDerivations(eng.processSource(src).bindings);
  const ctx = { derivations: dst.derivations, bindings: dst.bindings, fixedValues: dst.fixedValues };
  const sig = eng.orchestrator.signatureOf('f', dst.bindings);
  assert.equal(sig.kind, 'function');
  const paramNames = sig.inputs.map((i: any) => i.paramName);
  const clm = eng.clm.lowerMeasure(sig.body, ctx, { boundaries: {}, freeInputs: paramNames });
  const irBaked = clm.body;
  const evalCloud = async (cloud: number[][]) => {
    const K = cloud.length;
    const refArrays: any = {};
    for (let j = 0; j < paramNames.length; j++) {
      const col = new Float64Array(K);
      for (let k = 0; k < K; k++) col[k] = cloud[k][j];
      refArrays[paramNames[j]] = col;
    }
    const out: any = sampler.evaluateExprN(irBaked, refArrays, K, {});
    const arr = out && out.shape && out.data ? out.data
      : (typeof out === 'number' ? new Array(K).fill(out) : out);
    return Array.from({ length: K }, (_, k: number) =>
      (Number.isFinite(arr[k]) ? arr[k] : -Infinity));
  };
  const fit = await eng.optimizer.optimize({
    evalCloud, x0: [0, 0], domains: [{ kind: 'real' }, { kind: 'real' }], scales: [5, 5],
    opts: { seed: 3 },
  });
  const ai = sig.inputs.findIndex((i: any) => i.kwargName === 'a');
  const bi = sig.inputs.findIndex((i: any) => i.kwargName === 'b');
  assert.ok(Math.abs(fit.mode[ai] - 2.0) < 0.02, `a = ${fit.mode[ai]} (expect 2.0)`);
  assert.ok(Math.abs(fit.mode[bi] + 1.0) < 0.02, `b = ${fit.mode[bi]} (expect -1.0)`);
});
