'use strict';

// =====================================================================
// likelihoodof(relabel(M, [axis]), record(axis = v)) — the single-
// observation, record-keyed likelihood density.
// =====================================================================
//
// Regression guard. relabel(M, names) is the output-side axis renaming of
// spec §04 — density-transparent (the per-point log-density of M is
// unchanged by its axis labels). The labels DO, however, name which record
// field carries M's observation when the obs is a record:
//
//   gauss_x = relabel(Normal(mu=mu, sigma=sigma), ["x"])
//   obs     = likelihoodof(gauss_x, record(x = v))
//
// must score logdensityof(obs, record(mu=m, sigma=s)) to the plain Normal
// log-pdf of v at (m, s) — relabel is transparent to the density.
//
// History: the iid-accepts-relabel change kept measure-relabels as `relabel`
// nodes (correct for the iid/density walk) instead of rewriting them to
// `record(x = Normal(…))`. That broke this record-keyed path: density.walkAcc
// peeled the relabel and handed the WHOLE record object to Normal's scalar
// leaf, which threw "cannot consume scalar from value of type object" — masked
// upstream as "Cannot read properties of undefined (reading 'length')". The
// peel now projects the named axis off the record before recursing on M.
//
// Both forms must coexist: iid(relabel(M), n) (test/iid-relabel.test.ts) and
// this record-keyed likelihoodof.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0x5EED2;  // distinct from other test files

function makeCtx(source: any) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker:  (msg: any) => {
      const reply = worker.handle(msg);
      if (reply && reply.type === 'error') return Promise.reject(new Error(reply.message));
      return Promise.resolve(reply);
    },
    sampleCount: 1,
    rootSeed:    ROOT_SEED,
  };
  return ctx;
}

// Closed-form Normal log-pdf — an INDEPENDENT oracle (not via the engine).
function normalLogpdf(x: number, mu: number, sigma: number): number {
  return -0.5 * Math.log(2 * Math.PI) - Math.log(sigma)
    - 0.5 * ((x - mu) / sigma) ** 2;
}

function modelSrc(v: number, m: number, s: number): string {
  return `
mu = elementof(reals)
sigma = elementof(posreals)
gauss_x = relabel(Normal(mu = mu, sigma = sigma), ["x"])
obs = likelihoodof(gauss_x, record(x = ${v}))
ld = logdensityof(obs, record(mu = ${m}, sigma = ${s}))
`;
}

const CASES: [number, number, number][] = [
  [1.27, 0.0, 1.0],   // the HS3 gaussian repro point
  [1.27, 0.5, 1.0],
  [2.0, 0.3, 1.5],
  [-0.4, 0.0, 2.25],
];

for (const [v, m, s] of CASES) {
  test(`likelihoodof(relabel(Normal,["x"]), record(x=${v})) @ mu=${m},sigma=${s} == Normal logpdf`,
    async () => {
      const measure = await makeCtx(modelSrc(v, m, s)).getMeasure('ld');
      const lp = measure.samples[0];
      const want = normalLogpdf(v, m, s);
      assert.ok(Number.isFinite(lp), `engine logp not finite: ${lp}`);
      assert.ok(Math.abs(lp - want) < 1e-9,
        `record-keyed relabel logp ${lp} != closed-form Normal ${want}`);
    });
}
