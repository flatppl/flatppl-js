'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const { buildLogPi } = require('../mcmc-density.ts');
const densityPrims = require('../density-prims.ts');
const valueLib = require('../value.ts');
const density = require('../density.ts');
const { orchestrator } = require('..');
const T = require('../types.ts');

async function score(measure: string, point: string, prefix = ''): Promise<number> {
  const { proc, ctx } = ctxFor(`${prefix}\nM = ${measure}\nlp = logdensityof(M, ${point})`, 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  return (await ctx.getMeasure('lp')).value.data[0];
}

// §06: the complete atom has mass one; changing any component leaves support.
test('Dirac density scores singleton and vector atoms without scalar consumption', async () => {
  assert.equal(await score('Dirac([1.2])', '[1.2]'), 0);
  assert.equal(await score('Dirac([1.2, 2.3])', '[1.2, 2.3]'), 0);
  assert.equal(await score('Dirac([1.2, 2.3])', '[1.2, 2.4]'), -Infinity);
  assert.equal(await score('Dirac("atom")', '"atom"'), 0);
  assert.equal(await score('Dirac("atom")', '"other"'), -Infinity);
  assert.equal(await score('Dirac(["a", "b"])', '["a", "b"]'), 0);
  const { proc, ctx } = ctxFor('lp = builtin_logdensityof(Dirac, [1, 2], [1, 2])', 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.equal((await ctx.getMeasure('lp')).value.data[0], 0);
});

test('Dirac matrices compare logical entries and retain their distinction from nested vectors', async () => {
  const matrix = 'Dirac(rowstack([[1, 2], [3, 4]]))';
  assert.equal(await score(matrix, 'transpose(rowstack([[1, 3], [2, 4]]))'), 0);
  assert.equal(await score(matrix, 'rowstack([[1, 2], [3, 5]])'), -Infinity);
  assert.equal(await score('Dirac(atom)', 'atom', 'atom = rowstack([[1, 2], [3, 4]])'), 0);
  assert.equal(await score('Dirac([[1, 2], [3, 4]])', '[[1, 2], [3, 4]]'), 0);
  const nested = valueLib.asValue([[1, 2], [3, 4]]);
  const flat = { shape: [2, 2], data: Float64Array.of(1, 2, 3, 4) };
  assert.equal(densityPrims.builtinLogdensityof('Dirac', { value: flat }, flat), 0);
  assert.equal(densityPrims.builtinLogdensityof('Dirac', { value: flat }, nested), -Infinity);
});

test('Dirac record atoms retain nested values and ordered field structure', async () => {
  const atom = 'record(label = "a", nested = record(flag = true, xs = [1.2, 2.3]))';
  assert.equal(await score(`Dirac(value = ${atom})`, atom), 0);
  assert.equal(await score(`Dirac(value = ${atom})`, 'record(label = "a", nested = record(flag = false, xs = [1.2, 2.3]))'), -Infinity);
  const point = { a: 1, b: [2, 3] };
  assert.equal(densityPrims.builtinLogdensityof('Dirac', { value: point }, { a: 1, b: [2, 3] }), 0);
  assert.equal(densityPrims.builtinLogdensityof('Dirac', { value: point }, { b: [2, 3], a: 1 }), -Infinity);
  assert.equal(await score('Dirac(value = record(re = 1, im = 2, extra = 3))', 'record(re = 1, im = 2, extra = 4)'), -Infinity);
});

test('Dirac container footprints compose through iid and positional joints', async () => {
  assert.equal(await score('iid(Dirac([1, 2]), 2)', '[[1, 2], [1, 2]]'), 0);
  assert.equal(await score('iid(Dirac([1, 2]), 2)', 'points', 'points = [[1, 2], [1, 2]]'), 0);
  assert.equal(await score('iid(Dirac([1, 2]), 2)', '[[1, 2], [1, 3]]'), -Infinity);
  assert.equal(await score('joint(Dirac([1, 2]), Dirac([3]))', '[1, 2, 3]'), 0);
  assert.equal(await score('joint(Dirac([1, 2]), Dirac([3]))', '[1, 2, 4]'), -Infinity);
  const records = 'joint(Dirac(value = record(a = 1, b = [2, 3])), Dirac(value = record(c = 4)))';
  assert.equal(await score(records, 'record(a = 1, b = [2, 3], c = 4)'), 0);
  assert.equal(await score('iid(Dirac(value = record(a = 1, b = [2, 3])), 2)', 'table(a = [1, 1], b = [[2, 3], [2, 3]])'), 0);
  assert.equal(await score('iid(Dirac("a"), 2)', '["a", "a"]'), 0);
  assert.equal(await score('joint(Dirac("a"), Dirac("b"))', '["a", "b"]'), 0);
  assert.equal(await score('Dirac([complex(1, 2), complex(3, 4)])', '[complex(1, 2), complex(3, 4)]'), 0);
  assert.equal(await score('iid(Dirac(complex(1, 2)), 2)', '[complex(1, 2), complex(1, 2)]'), 0);
  assert.equal(await score('joint(Dirac([complex(1, 2)]), Dirac([complex(3, 4)]))', 'xs', 'xs = [complex(1, 2), complex(3, 4)]'), 0);
});

test('live record and vector Dirac support is resolved separately for each likelihood input', async () => {
  const { proc, ctx } = ctxFor(`
theta ~ Normal(0, 1)
y ~ Dirac(value = record(label = "a", xs = [theta, theta + 1]))
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = record(label = "a", xs = [1, 2])))
posterior = bayesupdate(L, lawof(record(theta = theta)))`, 3);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  const points = [{ theta: 1 }, { theta: 2 }, { theta: 1 }];
  assert.deepEqual(Array.from(scorer.priorLikBatch(points).lik), [0, -Infinity, 0]);
  assert.deepEqual(points.map(p => scorer.likOf(p)), [0, -Infinity, 0]);
});

test('affine vector Dirac density preserves the whole generated atom despite inverse roundoff', async () => {
  const measure = 'pushfwd(x -> 0.3 .* x .+ 0.1, Dirac([1.2, 2.3]))';
  assert.equal(await score(measure, `[${0.1 + 0.3 * 1.2}, ${0.1 + 0.3 * 2.3}]`), 0);
  assert.equal(await score(measure, '[0.46, 0.8]'), -Infinity);
});

test('live vector inputs retain their complete Dirac support in scalar and batch likelihoods', async () => {
  const { proc, ctx } = ctxFor(`
theta ~ iid(Normal(0, 1), 2)
y ~ Dirac(value = record(xs = theta))
K = kernelof(record(y = y), theta = theta)
L = likelihoodof(K, record(y = record(xs = [1, 2])))
posterior = bayesupdate(L, lawof(record(theta = theta)))`, 3);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const scorer = await buildLogPi(ctx, ctx.derivations.posterior);
  const points = [{ theta: [1, 2] }, { theta: [1, 3] }, { theta: [1, 2] }];
  assert.deepEqual(Array.from(scorer.priorLikBatch(points).lik), [0, -Infinity, 0]);
  assert.deepEqual(points.map(p => scorer.likOf(p)), [0, -Infinity, 0]);
});

test('batched Dirac density pairs each whole point with its fixed or live support', async () => {
  const { proc, ctx } = ctxFor('M = Dirac([1, 2])\nlp = broadcast(logdensityof, M, [[1, 2], [1, 3], [1, 2]])', 1);
  assert.deepEqual(proc.diagnostics.filter((d: any) => d.severity === 'error'), []);
  assert.deepEqual(Array.from((await ctx.getMeasure('lp')).samples), [0, -Infinity, 0]);
  const live = ctxFor('theta ~ iid(Normal(0, 1), 2)\nM = Dirac(theta)', 1).ctx;
  const ir = orchestrator.expandMeasure('M', live);
  const points = valueLib.asValue([[1, 2], [1, 3], [1, 4]]);
  const parents = { theta: valueLib.asValue([[1, 2], [1, 2], [1, 4]]) };
  assert.deepEqual(Array.from(density.logDensityN(ir, points, parents, 3, { pointsBatched: true })), [0, -Infinity, 0]);
});

test('table Dirac atoms score whole tables and incompatible footprints are diagnosed', async () => {
  assert.equal(await score('Dirac(value = table(a = [1, 2]))', 'table(a = [1, 2])'), 0);
  assert.equal(await score('Dirac(value = table(a = [1, 2]))', 'table(a = [1, 3])'), -Infinity);
  for (const { atom, point, message } of [
    { atom: '[1, 2]', point: '[1]', message: /incompatible shapes/ },
    { atom: 'record(a = [1])', point: 'record(a = [1, 2])', message: /record field has the wrong shape/ },
    { atom: 'record(a = 1)', point: 'record(b = 1)', message: /missing record field/ },
  ]) {
    const { proc } = ctxFor(`lp = logdensityof(Dirac(value = ${atom}), ${point})`, 1);
    assert.ok(proc.diagnostics.some((d: any) => d.severity === 'error' && message.test(d.message)));
  }
});

test('undecidable Dirac footprints defer to runtime without claiming incompatibility', () => {
  const ctx = ctxFor('M = Dirac([1, 2])', 1).ctx;
  const sourceIR = orchestrator.expandMeasure('M', ctx);
  const vector = T.array(1, [2], T.REAL), dynamic = T.array(1, ['%dynamic'], T.REAL);
  for (const [atom, observed] of [[T.any(), vector], [vector, T.deferred()], [dynamic, vector], [vector, dynamic]]) {
    const ir = { ...sourceIR, meta: { type: T.measure(atom) } };
    const footprint = densityPrims.staticConsume(ir, observed);
    assert.equal(footprint.deferred, true);
    assert.equal(footprint.error, undefined);
    assert.equal(density.logDensityN(ir, [1, 2], {}, 1, {})[0], 0);
    assert.equal(density.logDensityN(ir, [1, 3], {}, 1, {})[0], -Infinity);
  }
});

test('iid Dirac consumption preserves the unscored complex suffix and rejects missing rows', () => {
  const ctx = ctxFor('M = iid(Dirac([complex(1, 2)]), 1)', 1).ctx;
  const ir = orchestrator.expandMeasure('M', ctx);
  const points = [[{ re: 1, im: 2 }], [{ re: 3, im: 4 }]];
  const consumed = density.logDensityConsumeN(ir, valueLib.asValue(points), {}, 1, {});
  assert.deepEqual(Array.from(consumed.logps), [0]);
  assert.deepEqual(consumed.rest, valueLib.asValue(points.slice(1)));
  assert.deepEqual(density.logDensityConsumeN(ir, points, {}, 1, {}).rest, points.slice(1));
  const two = ctxFor('M = iid(Dirac([complex(1, 2)]), 2)', 1).ctx;
  assert.throws(() => density.logDensityN(orchestrator.expandMeasure('M', two), valueLib.asValue(points.slice(0, 1)), {}, 1, {}), /only 1 available/);
});
