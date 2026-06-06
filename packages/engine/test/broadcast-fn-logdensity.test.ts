'use strict';

// broadcast(fn(logdensityof(M, _)), grid) and its dot-sugar must
// evaluate M's log-density at every grid point in ONE binding,
// bit-exact to the per-point scalar logdensityof loop. Routes through
// the existing broadcast_logdensity materialise path.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource } = require('../index.ts');
const orchestrator = require('../orchestrator.ts');
const materialiser = require('../materialiser.ts');
const { createWorkerHandler } = require('../worker.ts');

function mk(src: string): any {
  const lifted = processSource(src);
  const built = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: [1, 2, 3] });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure: (name: any) => {
      if (cache.has(name)) return cache.get(name);
      const p = materialiser.materialiseMeasure(name, ctx);
      cache.set(name, p);
      return p;
    },
    sendWorker: (m: any) => {
      const r = worker.handle(m);
      if (r && r.type === 'error') return Promise.reject(new Error(r.message));
      return Promise.resolve(r);
    },
    sampleCount: 1000,
    rootSeed: [1, 2, 3],
  };
  return ctx;
}

const MODEL = `
A = Normal(mu = -2.0, sigma = 0.8)
B = Normal(mu =  3.0, sigma = 1.3)
mix = normalize(superpose(weighted(0.4, A), weighted(0.6, B)))
`;
const PTS = [-2.0, 0.0, 3.0, 7.5];

async function scalarRef(): Promise<number[]> {
  let src = MODEL;
  PTS.forEach((p, i) => { src += `p${i} = logdensityof(mix, ${p})\n`; });
  const ctx = mk(src);
  const out: number[] = [];
  for (let i = 0; i < PTS.length; i++) out.push((await ctx.getMeasure('p' + i)).samples[0]);
  return out;
}

test('broadcast(fn(logdensityof(M,_)), grid): bit-exact to scalar loop', async () => {
  const ctx = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(fn(logdensityof(mix, _)), grid)`);
  const m = await ctx.getMeasure('gv');
  assert.ok(m && m.samples, 'gv must materialise to a measure with samples');
  assert.equal(m.samples.length, PTS.length, `expected ${PTS.length} densities`);
  const ref = await scalarRef();
  for (let i = 0; i < PTS.length; i++) {
    assert.equal(m.samples[i], ref[i],
      `point ${PTS[i]}: broadcast=${m.samples[i]} scalar=${ref[i]}`);
  }
});

test('broadcast(fn(logdensityof(M,_)), grid): agrees with the existing 3-arg form', async () => {
  const fn3 = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(logdensityof, mix, grid)`);
  const fnW = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(fn(logdensityof(mix, _)), grid)`);
  const a = await fn3.getMeasure('gv');
  const b = await fnW.getMeasure('gv');
  assert.equal(b.samples.length, a.samples.length);
  for (let i = 0; i < a.samples.length; i++) assert.equal(b.samples[i], a.samples[i]);
});

test('dot-sugar fn(logdensityof(M,_)).(grid): bit-exact to scalar loop', async () => {
  const ctx = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = (fn(logdensityof(mix, _))).(grid)`);
  const m = await ctx.getMeasure('gv');
  assert.ok(m && m.samples, 'gv must materialise to a measure with samples');
  assert.equal(m.samples.length, PTS.length);
  const ref = await scalarRef();
  for (let i = 0; i < PTS.length; i++) {
    assert.equal(m.samples[i], ref[i],
      `point ${PTS[i]}: dot=${m.samples[i]} scalar=${ref[i]}`);
  }
});

// --- Regression: shapes that must NOT be captured by the new classifier ---

test('regression: value-fn broadcast still maps elementwise', async () => {
  // broadcast(fn(2*_ + 1), A) is a pure-arithmetic value function — body
  // op is `add`/`mul`, not logdensityof, so the new classifier returns
  // null and the existing value-fn path handles it.
  const ctx = mk(`A = [1.0, 2.0, 3.0]\ngv = broadcast(fn(2 * _ + 1), A)`);
  const m = await ctx.getMeasure('gv');
  assert.deepEqual(Array.from(m.samples), [3.0, 5.0, 7.0]);
});

test('regression: kernel broadcast (Normal over params) still samples', async () => {
  // broadcast(Normal, mus, sigmas) is a kernel broadcast — head is a
  // bare distribution ref, not a functionof, so the new classifier
  // returns null. Must still produce a materialisable measure.
  const ctx = mk(`
mus = [-5.0, 0.0, 5.0]
sigmas = [0.1, 0.1, 0.1]
gv = broadcast(Normal, mus, sigmas)
`);
  const m = await ctx.getMeasure('gv');
  assert.ok(m && (m.samples || m.fields || m.dims), 'kernel broadcast must materialise');
});

test('regression: bare 3-arg broadcast(logdensityof, M, pts) unchanged', async () => {
  const ctx = mk(MODEL + `grid = [${PTS.join(', ')}]\ngv = broadcast(logdensityof, mix, grid)`);
  const m = await ctx.getMeasure('gv');
  const ref = await scalarRef();
  for (let i = 0; i < PTS.length; i++) assert.equal(m.samples[i], ref[i]);
});
