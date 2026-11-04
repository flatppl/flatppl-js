'use strict';

// `bijection(...)` is an ordinary function-valued expression (spec §06), so
// `pushfwd(bijection(f, f_inv, lv), M)` must mean the same as binding the
// bijection to a name first. Both spellings of the parts are covered: inline
// `fn(...)` and the spec's bare builtin names `bijection(exp, log, identity)`.
//
// Oracle: pushfwd(exp, Normal(mu, sigma)) = LogNormal(mu, sigma). The values
// below come from scipy 1.18.0, `stats.lognorm(s=1.5, scale=exp(0.5))`
// (logpdf and ppf), not from this engine.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0x5EED1E55;

function makeCtx(source: string, sampleCount: number) {
  const lifted = processSource(source);
  const built  = orchestrator.buildDerivations(lifted.bindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: ROOT_SEED });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings:    built.bindings,
    fixedValues: built.fixedValues || new Map(),
    getMeasure:  (name: string) => {
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
    sampleCount,
    rootSeed: ROOT_SEED,
  };
  return ctx;
}

// scipy: stats.lognorm(s=1.5, scale=exp(0.5)).logpdf(y)
const LOGNORMAL_LOGPDF: [number, number][] = [
  [0.3, -0.7656582409607664],
  [2.0, -2.025841007063506],
  [7.5, -3.8492913644435633],
];
// scipy: stats.lognorm(s=1.5, scale=exp(0.5)).ppf(q)
const LOGNORMAL_QUANTILES: [number, number][] = [
  [0.10, 0.2411521183071379],
  [0.25, 0.5994548468931621],
  [0.50, 1.6487212707001282],
  [0.75, 4.534589790285759],
  [0.90, 11.272062827152808],
];

const BASE = 'Normal(mu = 0.5, sigma = 1.5)';
const FORMS: [string, string][] = [
  ['inline fn parts',
    `LN = pushfwd(bijection(fn(exp(_)), fn(log(_)), fn(_)), ${BASE})`],
  ['named fn parts',
    `b = bijection(fn(exp(_)), fn(log(_)), fn(_))\nLN = pushfwd(b, ${BASE})`],
  ['inline builtin parts',
    `LN = pushfwd(bijection(exp, log, identity), ${BASE})`],
  ['named builtin parts',
    `b = bijection(exp, log, identity)\nLN = pushfwd(b, ${BASE})`],
];

for (const [label, model] of FORMS) {
  test(`pushfwd(bijection) ${label}: logdensityof matches LogNormal`, async () => {
    const lines = LOGNORMAL_LOGPDF.map(([y], i) => `lp${i} = logdensityof(LN, ${y})`);
    const ctx = makeCtx(`${model}\n${lines.join('\n')}\n`, 8);
    for (let i = 0; i < LOGNORMAL_LOGPDF.length; i++) {
      const [y, expected] = LOGNORMAL_LOGPDF[i];
      const lp = await ctx.getMeasure(`lp${i}`);
      for (const got of lp.samples) {
        assert.ok(Math.abs(got - expected) < 1e-12, `y=${y}: got ${got}, expected ${expected}`);
      }
    }
  });

  test(`pushfwd(bijection) ${label}: samples follow LogNormal quantiles`, async () => {
    const n = 8192;
    const ctx = makeCtx(`${model}\n`, n);
    const m = await ctx.getMeasure('LN');
    assert.equal(m.samples.length, n);
    for (const [q, x] of LOGNORMAL_QUANTILES) {
      let below = 0;
      for (const s of m.samples) if (s <= x) below++;
      // Binomial sd at n = 8192 is at most 0.0055; 0.025 is about 4.5 sd.
      assert.ok(Math.abs(below / n - q) < 0.025, `F(${x}) = ${below / n}, expected ${q}`);
    }
  });
}

test('pushfwd(bijection) inline with a literal logvolume matches the affine law', async () => {
  // y = 2x + 1 with x ~ Normal(0, 1) is Normal(1, 2).
  // scipy: stats.norm(1, 2).logpdf(y)
  const ctx = makeCtx(`
M = pushfwd(bijection(fn(2.0 * _ + 1.0), fn((_ - 1.0) / 2.0), 0.6931471805599453), Normal(mu = 0.0, sigma = 1.0))
a = logdensityof(M, -1.0)
b = logdensityof(M, 4.5)
`, 8);
  for (const [name, expected] of [['a', -2.112085713764618], ['b', -3.143335713764618]] as const) {
    const lp = await ctx.getMeasure(name);
    for (const got of lp.samples) assert.ok(Math.abs(got - expected) < 1e-12, `${name}: ${got}`);
  }
});

for (const [label, model] of [
  ['inline', 'y ~ pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0)), Normal(1.0, 2.0))'],
  ['named', 'b = bijection(fn(2.0 * _), fn(_ / 2.0), log(2.0))\ny ~ pushfwd(b, Normal(1.0, 2.0))'],
  ['inline, named scalar', 'c = log(2.0)\ny ~ pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), c), Normal(1.0, 2.0))'],
]) {
  test(`pushfwd(bijection) ${label} with a scalar-expression logvolume`, async () => {
    // §06: logvolume may be a scalar value. y = 2x with x ~ Normal(1, 2) is
    // Normal(2, 4). scipy: stats.norm(2, 4).logpdf(y)
    const ctx = makeCtx(`${model}
a = logdensityof(lawof(y), 3.0)
b2 = logdensityof(lawof(y), -5.0)
`, 8);
    for (const [name, expected] of [['a', -2.336482894324563], ['b2', -3.836482894324563]] as const) {
      const lp = await ctx.getMeasure(name);
      for (const got of lp.samples) assert.ok(Math.abs(got - expected) < 1e-12, `${name}: ${got}`);
    }
  });
}

test('pushfwd(bijection) inline inside iid matches the LogNormal product', async () => {
  // scipy: sum of stats.lognorm(s=1.5, scale=exp(0.5)).logpdf at 2.0, 1.0, 0.5
  const ctx = makeCtx(`
M = iid(pushfwd(bijection(fn(exp(_)), fn(log(_)), fn(_)), ${BASE}), 3)
lp = logdensityof(M, [2.0, 1.0, 0.5])
`, 8);
  const lp = await ctx.getMeasure('lp');
  for (const got of lp.samples) assert.ok(Math.abs(got - (-4.353412263457712)) < 1e-12, `${got}`);
});
