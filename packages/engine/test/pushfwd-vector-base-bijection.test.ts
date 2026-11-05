'use strict';

// §06 `bijection(f, f_inv, logvolume)` "is semantically `f`", and the density
// of `pushfwd(f, M)` is `log p_M(f_inv(y)) - logvolume(f_inv(y))`. Over an
// array-valued base, f and f_inv therefore map the WHOLE variate, and
// logvolume is the one log-volume element of the whole map (3 log 2 for
// y = 2x on R^3), not a per-element term.
//
// Oracle values come from scipy 1.18.0, not from this engine.

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

async function assertScores(source: string, expected: number) {
  const ctx = makeCtx(source, 4);
  const lp = await ctx.getMeasure('lp');
  for (const got of lp.samples) {
    assert.ok(Math.abs(got - expected) < 1e-12, `got ${got}, expected ${expected}`);
  }
}

// scipy: stats.norm(1, 2).logpdf(y / 2).sum() - 3 log 2 at y = [3, 1, -2]
// (= stats.norm(2, 4).logpdf(y).sum()).
const SCALED_IID = -7.478198682973691;
const OBS = 'lp = logdensityof(lawof(y), [3.0, 1.0, -2.0])';
const IID_BASE = 'iid(Normal(1.0, 2.0), 3)';

for (const [label, model] of [
  ['named, literal logvolume',
    `b = bijection(fn(2.0 * _), fn(_ / 2.0), 2.0794415416798357)\ny ~ pushfwd(b, ${IID_BASE})`],
  ['inline, literal logvolume',
    `y ~ pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), 2.0794415416798357), ${IID_BASE})`],
  ['named, constant fn logvolume',
    `b = bijection(fn(2.0 * _), fn(_ / 2.0), fn(3.0 * log(2.0)))\ny ~ pushfwd(b, ${IID_BASE})`],
  ['inline, scalar-expression logvolume',
    `y ~ pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), 3.0 * log(2.0)), ${IID_BASE})`],
]) {
  test(`pushfwd(bijection) over an iid vector base, ${label}`, async () => {
    await assertScores(`${model}\n${OBS}\n`, SCALED_IID);
  });
}

test('pushfwd of an unannotated affine fn over an iid vector base sums the element LADJ', async () => {
  // §06 case 1: an affine map needs no annotation. y = 2x on R^3 has
  // log|det J| = 3 log 2, the same law as the annotated form above.
  await assertScores(`y ~ pushfwd(fn(2.0 * _), ${IID_BASE})\n${OBS}\n`, SCALED_IID);
});

test('pushfwd(bijection) with a point-dependent logvolume over an iid vector base', async () => {
  // y = exp.(x), log|det J| = sum(x). scipy:
  // stats.lognorm(s=1.5, scale=exp(0.5)).logpdf([2.0, 1.0, 0.5]).sum()
  for (const model of [
    'b = bijection(fn(exp.(_)), fn(log.(_)), fn(sum(_)))\ny ~ pushfwd(b, iid(Normal(0.5, 1.5), 3))',
    'y ~ pushfwd(bijection(fn(exp.(_)), fn(log.(_)), fn(sum(_))), iid(Normal(0.5, 1.5), 3))',
  ]) {
    await assertScores(`${model}\nlp = logdensityof(lawof(y), [2.0, 1.0, 0.5])\n`, -4.353412263457712);
  }
});

const MV_BASE = 'MvNormal([1.0, 0.0, -1.0], rowstack([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 1.5]]))';

test('pushfwd(bijection) over an MvNormal base, affine map', async () => {
  // scipy: multivariate_normal(mu, S).logpdf((y - c) / 2) - 3 log 2
  // (= multivariate_normal(2 mu + c, 4 S).logpdf(y)), c = [1, 0, -1].
  await assertScores(`
b = bijection(fn(2.0 * _ + [1.0, 0.0, -1.0]), fn((_ - [1.0, 0.0, -1.0]) / 2.0), 2.0794415416798357)
y ~ pushfwd(b, ${MV_BASE})
${OBS}
`, -5.464772545293059);
});

test('pushfwd(bijection) over an MvNormal base, point-dependent logvolume', async () => {
  // Multivariate log-normal. scipy:
  // multivariate_normal(mu, S).logpdf(log(z)) - sum(log(z)), z = [2, 1, 0.5]
  await assertScores(`
y ~ pushfwd(bijection(fn(exp.(_)), fn(log.(_)), fn(sum(_))), ${MV_BASE})
lp = logdensityof(lawof(y), [2.0, 1.0, 0.5])
`, -3.2589084569881046);
});

test('iid of a vector-base pushfwd(bijection) scores each row', async () => {
  // scipy: stats.norm(2, 4).logpdf([[3, 1, -2], [0.5, 0, 4]]).sum()
  await assertScores(`
b = bijection(fn(2.0 * _), fn(_ / 2.0), 2.0794415416798357)
M = iid(pushfwd(b, ${IID_BASE}), 2)
lp = logdensityof(M, [[3.0, 1.0, -2.0], [0.5, 0.0, 4.0]])
`, -14.714209865947378);
});

test('a vector-base pushfwd(bijection) inside a positional joint consumes its whole variate', async () => {
  // scipy: stats.norm(0, 1).logpdf(0.5) + stats.norm(2, 4).logpdf([3, 1, -2]).sum()
  // §06 forbids mixing shape classes in a positional joint, so both parts are vectors.
  await assertScores(`
b = bijection(fn(2.0 * _), fn(_ / 2.0), 2.0794415416798357)
M = joint(iid(Normal(0.0, 1.0), 1), pushfwd(b, ${IID_BASE}))
lp = logdensityof(M, [0.5, 3.0, 1.0, -2.0])
`, -8.522137216178363);
});

test('broadcast(logdensityof) scores each vector point of a vector-base pushfwd(bijection)', async () => {
  // scipy: stats.norm(2, 4).logpdf(row).sum() per row
  const ctx = makeCtx(`
M = pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), 2.0794415416798357), ${IID_BASE})
lp = broadcast(logdensityof, M, [[3.0, 1.0, -2.0], [0.5, 0.0, 4.0]])
`, 1);
  const lp = await ctx.getMeasure('lp');
  const expected = [SCALED_IID, -7.236011182973689];
  assert.equal(lp.samples.length, 2);
  lp.samples.forEach((got: number, i: number) =>
    assert.ok(Math.abs(got - expected[i]) < 1e-12, `point ${i}: ${got}`));
});

test('pushfwd(bijection) over a counting iid vector base', async () => {
  // y = x + 1 with x ~ iid(Poisson(2), 3). scipy:
  // stats.poisson(2).logpmf([0, 2, 1]).sum(); a non-lattice point has mass 0.
  const ctx = makeCtx(`
b = bijection(fn(_ + 1.0), fn(_ - 1.0), 0.0)
y ~ pushfwd(b, iid(Poisson(2.0), 3))
lp = logdensityof(lawof(y), [1.0, 3.0, 2.0])
off = logdensityof(lawof(y), [1.0, 3.5, 2.0])
`, 4);
  const lp = await ctx.getMeasure('lp');
  for (const got of lp.samples) assert.ok(Math.abs(got - (-4.613705638880109)) < 1e-12, `${got}`);
  const off = await ctx.getMeasure('off');
  for (const got of off.samples) assert.equal(got, -Infinity);
});

test('pushfwd(bijection) over a matrix iid base maps the whole matrix', async () => {
  // y = 2x on R^{2x2}, log|det J| = 4 log 2. scipy:
  // stats.norm(1, 2).logpdf([3, 1, -2, 0.5] / 2).sum() - 4 log 2
  await assertScores(`
b = bijection(fn(2.0 * _), fn(_ / 2.0), fn(4.0 * log(2.0)))
y ~ pushfwd(b, iid(Normal(1.0, 2.0), [2, 2]))
lp = logdensityof(lawof(y), rowstack([[3.0, 1.0], [-2.0, 0.5]]))
`, -9.853744077298254);
});

test('pushfwd(bijection) over a vector base refuses an f_inv that changes the shape', async () => {
  const ctx = makeCtx(`
y ~ pushfwd(bijection(fn(2.0 * _), fn(sum(_)), 0.0), ${IID_BASE})
${OBS}
`, 4);
  await assert.rejects(ctx.getMeasure('lp'), /f_inv must return a variate of shape \[3\]/);
});

test('pushfwd(bijection) over a vector base refuses a non-scalar logvolume', async () => {
  // §06: logvolume is the log-volume element of the whole map, one scalar.
  const ctx = makeCtx(`
y ~ pushfwd(bijection(fn(2.0 * _), fn(_ / 2.0), fn(log(2.0) + 0.0 * _)), ${IID_BASE})
${OBS}
`, 4);
  await assert.rejects(ctx.getMeasure('lp'), /logvolume/);
});
