'use strict';

// Categorical/Categorical0 logpmf must refuse an off-lattice (non-integer)
// query with -Infinity, matching spec §08 (Categorical domain is `integers`)
// and scipy `rv_discrete.logpmf`, instead of silently truncating the query
// to the lattice via `k | 0` and answering with a positive-probability atom.
//
// ORACLE — scipy rv_discrete (python MCP, NOT engine output), for
// p = [0.1, 0.2, 0.3, 0.4]:
//   cat = rv_discrete(values=([1,2,3,4], p))       # Categorical, 1-based
//   cat.logpmf(2.5)    = -inf
//   cat.logpmf(2.9)    = -inf
//   cat.logpmf(1.0001) = -inf
//   cat.logpmf(2)      = -1.6094379124341003   (= log 0.2)
//   cat.logpmf(3)      = -1.2039728043259361   (= log 0.3)
// Categorical0 (0-based, same p) shifts the lattice by one: cat0.logpmf(k)
// = cat.logpmf(k+1), so the same off-lattice points and on-lattice points
// k=1 (-> log 0.2) / k=2 (-> log 0.3) apply unchanged.
//
// Poisson/Bernoulli/Binomial already refuse off-lattice queries (verified
// directly against @stdlib's own logpmf/pmf, not scipy) — pinned here as a
// regression only, no fix needed for those three.

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function buildCtx(src: string) {
  const proc = processSource(src);
  const built = orchestrator.buildDerivations(proc.bindings);
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 1,
    rootKey: 3, rootSeed: 3, marginalizationCount: 32,
    moduleRegistry: proc.loweredModule && proc.loweredModule.moduleRegistry,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const m = materialiser.materialiseMeasure(n, ctx); cache.set(n, m); return m;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}

const P = '[0.1, 0.2, 0.3, 0.4]';

async function scoreCategorical(k: number) {
  const ctx = buildCtx(`
M = Categorical(p = ${P})
ld = logdensityof(M, ${k})
`);
  const m = await ctx.getMeasure('ld');
  return m.samples[0];
}

async function scoreCategorical0(k: number) {
  const ctx = buildCtx(`
M = Categorical0(p = ${P})
ld = logdensityof(M, ${k})
`);
  const m = await ctx.getMeasure('ld');
  return m.samples[0];
}

// -- Categorical (1-based) -------------------------------------------------

for (const k of [2.5, 2.9, 1.0001]) {
  test(`Categorical logpmf(${k}) is -Infinity (off-lattice)`, async () => {
    const got = await scoreCategorical(k);
    assert.strictEqual(got, -Infinity, `got ${got}, expected -Infinity`);
  });
}

test('Categorical logpmf(2) = log 0.2 (on-lattice, scipy oracle)', async () => {
  const ORACLE = -1.6094379124341003;
  const got = await scoreCategorical(2);
  assert.ok(Math.abs(got - ORACLE) < 1e-10, `got ${got}, expected ${ORACLE}`);
});

test('Categorical logpmf(3) = log 0.3 (on-lattice, scipy oracle)', async () => {
  const ORACLE = -1.2039728043259361;
  const got = await scoreCategorical(3);
  assert.ok(Math.abs(got - ORACLE) < 1e-10, `got ${got}, expected ${ORACLE}`);
});

// -- Categorical0 (0-based) -------------------------------------------------

for (const k of [1.5, 1.9, 0.0001]) {
  test(`Categorical0 logpmf(${k}) is -Infinity (off-lattice)`, async () => {
    const got = await scoreCategorical0(k);
    assert.strictEqual(got, -Infinity, `got ${got}, expected -Infinity`);
  });
}

test('Categorical0 logpmf(1) = log 0.2 (on-lattice, scipy oracle)', async () => {
  const ORACLE = -1.6094379124341003;
  const got = await scoreCategorical0(1);
  assert.ok(Math.abs(got - ORACLE) < 1e-10, `got ${got}, expected ${ORACLE}`);
});

test('Categorical0 logpmf(2) = log 0.3 (on-lattice, scipy oracle)', async () => {
  const ORACLE = -1.2039728043259361;
  const got = await scoreCategorical0(2);
  assert.ok(Math.abs(got - ORACLE) < 1e-10, `got ${got}, expected ${ORACLE}`);
});

// -- Poisson/Bernoulli/Binomial regression pins (already correct) ----------
// Verified directly against @stdlib's own logpmf/pmf (not scipy — these are
// regression pins on existing correct behaviour, not new oracle claims).

test('Poisson logpmf(2.5) is -Infinity (off-lattice, regression pin)', async () => {
  const ctx = buildCtx(`
M = Poisson(rate = 3.0)
ld = logdensityof(M, 2.5)
`);
  const got = (await ctx.getMeasure('ld')).samples[0];
  assert.strictEqual(got, -Infinity, `got ${got}, expected -Infinity`);
});

test('Bernoulli logpmf(0.5) is -Infinity (off-lattice, regression pin)', async () => {
  const ctx = buildCtx(`
M = Bernoulli(p = 0.3)
ld = logdensityof(M, 0.5)
`);
  const got = (await ctx.getMeasure('ld')).samples[0];
  assert.strictEqual(got, -Infinity, `got ${got}, expected -Infinity`);
});

test('Binomial logpmf(2.5) is -Infinity (off-lattice, regression pin)', async () => {
  const ctx = buildCtx(`
M = Binomial(n = 5, p = 0.3)
ld = logdensityof(M, 2.5)
`);
  const got = (await ctx.getMeasure('ld')).samples[0];
  assert.strictEqual(got, -Infinity, `got ${got}, expected -Infinity`);
});
