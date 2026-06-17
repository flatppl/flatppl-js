'use strict';

// =====================================================================
// relabel is type-kind-transparent: iid (and the other measure-algebra
// ops) must accept a relabel'd measure.
// =====================================================================
//
// Spec basis (flatppl-design/docs/04-design.md "Interface adaptation"):
// the output-side renaming `relabel(M, names)` "lifts directly to sets,
// functions, measures, and kernels" and "for measures it is equivalent
// to pushfwd(fn(relabel(_, names)), M)". So relabel(M) IS a measure and
// iid(relabel(M), n) must type-check. Density evaluation already treats
// relabel as transparent (density.ts walkPushfwd peels it), so inference
// is the only inconsistent layer.
//
// Two checks:
//   1. Type-check: iid(relabel(Normal, [name]), n) yields NO error
//      diagnostics, both inline and via a named binding g = relabel(...).
//   2. Numeric equivalence: scoring logdensityof(likelihoodof(iid(
//      relabel(Normal(...)), n), obs), θ) EQUALS the same with a plain
//      Normal (relabel is density-transparent) to ~1e-12.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { processSource, orchestrator, materialiser } = require('..');
const { createWorkerHandler } = require('../worker.ts');

const ROOT_SEED = 0x1ABE1;  // distinct from other test files

function errorMessages(src: any): any[] {
  const r = processSource(src);
  return (r.diagnostics || [])
    .filter((d: any) => d.severity === 'error')
    .map((d: any) => d.message);
}

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

const PLAIN_SRC = `
m = elementof(reals)
s = elementof(posreals)
o = [0.0, 1.0]
g = Normal(mu = m, sigma = s)
L = likelihoodof(iid(g, lengthof(o)), o)
ld = logdensityof(L, record(m = 0.4, s = 1.3))
`;

const NAMED_SRC = `
m = elementof(reals)
s = elementof(posreals)
o = [0.0, 1.0]
g = relabel(Normal(mu = m, sigma = s), ["x"])
L = likelihoodof(iid(g, lengthof(o)), o)
ld = logdensityof(L, record(m = 0.4, s = 1.3))
`;

const INLINE_SRC = `
m = elementof(reals)
s = elementof(posreals)
o = [0.0, 1.0]
L = likelihoodof(iid(relabel(Normal(mu = m, sigma = s), ["x"]), lengthof(o)), o)
ld = logdensityof(L, record(m = 0.4, s = 1.3))
`;

test('iid accepts a relabel\'d measure — no type errors (named binding)', () => {
  assert.deepEqual(errorMessages(NAMED_SRC), []);
});

test('iid accepts a relabel\'d measure — no type errors (inline relabel arg)', () => {
  assert.deepEqual(errorMessages(INLINE_SRC), []);
});

test('plain iid still type-checks (baseline)', () => {
  assert.deepEqual(errorMessages(PLAIN_SRC), []);
});

test('relabel is density-transparent: iid(relabel(Normal)) logp == iid(Normal) logp', async () => {
  const plain = await makeCtx(PLAIN_SRC).getMeasure('ld');
  const named = await makeCtx(NAMED_SRC).getMeasure('ld');
  const inline = await makeCtx(INLINE_SRC).getMeasure('ld');
  const lpPlain  = plain.samples[0];
  const lpNamed  = named.samples[0];
  const lpInline = inline.samples[0];
  assert.ok(Number.isFinite(lpPlain), `plain logp not finite: ${lpPlain}`);
  assert.ok(Math.abs(lpNamed - lpPlain) < 1e-12,
    `named relabel logp ${lpNamed} != plain ${lpPlain}`);
  assert.ok(Math.abs(lpInline - lpPlain) < 1e-12,
    `inline relabel logp ${lpInline} != plain ${lpPlain}`);
});
