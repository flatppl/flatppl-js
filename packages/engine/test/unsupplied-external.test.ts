'use strict';

// An `external(...)` input takes its value from the host. This engine has no
// way to receive one, so a query that reads an unsupplied external must name
// that input, not fail with an unrelated "no derivation for '<binding>'".

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

function materialise(model: string, name: string, sources: Record<string, string> = {}) {
  const r = processSource(model, { bundle: { sources } });
  const built = orchestrator.buildDerivations(r.linkedBindings);
  const worker = createWorkerHandler();
  worker.handle({ type: 'init', seed: 3 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations,
    bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(),
    sampleCount: 4,
    rootSeed: 3,
    getMeasure: (n: string) => {
      if (!cache.has(n)) cache.set(n, materialiser.materialiseMeasure(n, ctx));
      return cache.get(n);
    },
    sendWorker: (msg: any) => {
      const reply = worker.handle(msg);
      return reply && reply.type === 'error' ? Promise.reject(new Error(reply.message)) : Promise.resolve(reply);
    },
  };
  return ctx.getMeasure(name);
}

const LIB = { 'lib.flatppl': 'c = external(anything)\nd = external(reals)\n' };

for (const [model, name, input, sources] of [
  ['x = external(reals)\nlp = logdensityof(Normal(x, 1.0), 0.1)\n', 'lp', 'x', {}],
  ['m = load_module("lib.flatppl")\nlp = logdensityof(m.c, 0.1)\n', 'lp', 'm.c', LIB],
  ['m = load_module("lib.flatppl")\nz = m.d + 1.0\n', 'z', 'm.d', LIB],
  ['m = load_module("lib.flatppl")\nz = m.c([1.0])\n', 'z', 'm.c', LIB],
] as const) {
  test(`an unsupplied external refuses by name: ${model.trim().split('\n').pop()}`, async () => {
    await assert.rejects(
      materialise(model, name, sources),
      { message: `'${input}': an external input supplied by the host is not supported by this engine` },
    );
  });
}

test('a substituted external is an ordinary value', async () => {
  const value = await materialise(
    'm = load_module("lib.flatppl", d = 2.0)\nz = m.d + 1.0\n',
    'z',
    LIB,
  );
  assert.equal(value.samples[0], 3.0);
});
