'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const library = `flatppl_compat = "0.1"
cols(x) = get.(fill(x, sizeof(x)[2]), all, indicesof(x)[2])
b = bijection(x -> colstack(x), cols, x -> 0.0)
f(p, n) = pushfwd(b, ((ni, pi) -> Multinomial(ni, pi)).(n, cols(p)))
`;

async function score(point: string): Promise<number> {
  const source = `flatppl_compat = "0.1"
a = load_module("m.flatppl")
p = rowstack([[0.2, 0.1], [0.3, 0.1], [0.5, 0.8]])
law = a.f(p, [5, 5])
lp = logdensityof(joint(mc = law, aux = Poisson(2)), record(mc = rowstack(${point}), aux = 3))
`;
  const model = processSource(source, { bundle: { sources: { 'm.flatppl': library } } });
  assert.deepEqual(model.diagnostics.filter((d: any) => d.severity === 'error'), []);
  const built = orchestrator.buildDerivations(model.linkedBindings, { moduleRegistry: model.linkedModuleRegistry });
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 1 });
  const ctx: any = { ...built, sampleCount: 1, rootSeed: 1,
    getMeasure: (name: string) => materialiser.materialiseMeasure(name, ctx),
    sendWorker: (msg: any) => {
      const result = worker.handle(msg);
      return result?.type === 'error' ? Promise.reject(new Error(result.message)) : Promise.resolve(result);
    } };
  const result = await ctx.getMeasure('lp');
  return (result.samples ?? result.value.data)[0];
}

test('a matrix counting bijection scores a product of multinomials', async () => {
  const actual = await score('[[1, 0], [2, 1], [2, 4]]');
  const expected = Math.log(30 * 0.2 * 0.3 ** 2 * 0.5 ** 2) + Math.log(5 * 0.1 * 0.8 ** 4)
    + 3 * Math.log(2) - 2 - Math.log(6);
  assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} != ${expected}`);
  assert.equal(await score('[[2, 0], [2, 1], [2, 4]]'), -Infinity);
});
