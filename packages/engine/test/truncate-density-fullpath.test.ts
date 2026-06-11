'use strict';

// C1 (CRITICAL, audit §3): `truncate(M, S)` density threw
// `'density: truncate requires parseSet opt'` on EVERY production path —
// the worker's logDensityN never supplied `opts.parseSet`, and no other
// caller did either. So any logdensityof/densityof/likelihoodof/bayesupdate
// over a tree containing a bare truncate crashed. A second gap: the truncate
// node's BASE measure (commonly a hoisted anon, e.g.
// `truncate(Normal(0,1), …)` → `truncate(ref __anon0, …)`) wasn't inlined by
// the structural expansion, dead-ending the density walker on a measure ref
// the dumb worker can't resolve.
//
// Fix: the worker builds `opts.parseSet` from ir-shared.parseSetIR (+ an
// env-eval fallback for ref-bound intervals); _expandStructural now recurses
// into truncate's base measure. Spec §06: truncate does NOT normalise — the
// density inside S equals the base density, outside S it's -inf.

const test = require('node:test');
const assert = require('node:assert');
const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const der = require(ENG + 'derivations.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function evalDensities(src: string, names: string[]) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  for (const nm of names) {
    if (built.bindings.has(nm) && !built.derivations[nm]) {
      const c = der.classifyDerivation(built.bindings.get(nm), built.bindings);
      if (c) built.derivations[nm] = c;
    }
  }
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed: 1 });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: 500, rootKey: 1,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return ctx;
}
const scalar1 = (m: any) => (m.value ? m.value.data[0] : (m.samples ? m.samples[0] : m));
const logN01 = (x: number) => -0.5 * Math.log(2 * Math.PI) - x * x / 2;

test('truncate density: inside support = base density, outside = -inf (interval, inline base)', async () => {
  const ctx = evalDensities(`
tn = truncate(Normal(0.0, 1.0), interval(-1.0, 1.0))
ld_in = logdensityof(tn, 0.5)
ld_out = logdensityof(tn, 2.0)
`, ['tn', 'ld_in', 'ld_out']);
  const inV = +scalar1(await ctx.getMeasure('ld_in'));
  const outV = +scalar1(await ctx.getMeasure('ld_out'));
  // Non-normalising: inside equals the base Normal(0,1) logpdf exactly.
  assert.ok(Math.abs(inV - logN01(0.5)) < 1e-9,
    `inside-support density ${inV} should equal base logpdf ${logN01(0.5)}`);
  assert.strictEqual(outV, -Infinity, 'outside-support density must be -inf');
});

test('truncate density: named set (posreals) and named base binding', async () => {
  const ctx = evalDensities(`
pos = truncate(Normal(0.0, 1.0), posreals)
ld_pos_in = logdensityof(pos, 0.5)
ld_pos_out = logdensityof(pos, -0.5)
base = Normal(0.0, 1.0)
tn_named = truncate(base, interval(-1.0, 1.0))
ld_named = logdensityof(tn_named, 0.5)
`, ['pos', 'ld_pos_in', 'ld_pos_out', 'tn_named', 'ld_named']);
  assert.ok(Math.abs(+scalar1(await ctx.getMeasure('ld_pos_in')) - logN01(0.5)) < 1e-9,
    'posreals inside-support density should equal base logpdf');
  assert.strictEqual(+scalar1(await ctx.getMeasure('ld_pos_out')), -Infinity,
    'posreals outside-support (x<0) density must be -inf');
  assert.ok(Math.abs(+scalar1(await ctx.getMeasure('ld_named')) - logN01(0.5)) < 1e-9,
    'named-base truncate density should inline the base and equal its logpdf');
});
