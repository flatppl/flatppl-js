'use strict';

// Timing gates for the scalar evaluation paths. They run only through
// `npm run test:perf`, never in CI: GitHub's virtualised runners time too
// unreliably to gate on a ratio. Run them on a quiet machine. Each gate
// interleaves its sides trial by trial and compares medians, so a burst of
// load hits every side alike. The bit-exactness of each path is checked in
// the CI suite (eval-compile, interpreter-scalar-kernels, profile-scalar-ops).

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const sampler = require('../../sampler.ts');
const batched = require('../../sampler-eval-batched.ts');
const profile = require('../../sampler-profile-compile.ts');
const { mcInverseWorkload } = require('../_mc-inverse-workload.ts');

const TRIALS = 9;
const hr = () => Number(process.hrtime.bigint()) / 1e6;
const median = (a: number[]) => a.slice().sort((x, y) => x - y)[a.length >> 1];

// Median wall time of each side over TRIALS rounds. Each round times every
// side once, starting at a different side, after one untimed warm-up round.
function interleaved(sides: Record<string, () => void>): Record<string, number> {
  const names = Object.keys(sides);
  for (const n of names) sides[n]();
  const times: Record<string, number[]> = {};
  for (const n of names) times[n] = [];
  for (let t = 0; t < TRIALS; t++) {
    for (let k = 0; k < names.length; k++) {
      const n = names[(t + k) % names.length];
      const t0 = hr();
      sides[n]();
      times[n].push(hr() - t0);
    }
  }
  const out: Record<string, number> = {};
  for (const n of names) out[n] = median(times[n]);
  return out;
}

// A toggle an older engine lacks is a no-op there, so its gate compares a path
// with itself and fails, as it should for code without that optimisation.
const toggle = (mod: any, name: string, on: boolean) => {
  if (typeof mod[name] === 'function') mod[name](on);
};

const report = (gate: string, slow: number, fast: number, bound: number) =>
  process.stderr.write(`  perf ${gate}: ${slow.toFixed(1)} ms vs ${fast.toFixed(1)} ms, `
    + `${(slow / fast).toFixed(2)}x (gate ${bound}x)\n`);

// The MC inverse of simple-transport, 20 x 6000 points, three ways: the
// interpreter's shared ARITH_OPS loop, its per-op scalar kernels, and the
// fused-loop compiler. Both generated paths splice each op's body, so only the
// fusion separates them: about 1.2x on Apple M-series, 1.4x on batserv02.
test('MC inverse: the kernels and the compiled loop beat the ARITH_OPS loop', () => {
  const { ir, refArrays, count, baseEnv } = mcInverseWorkload();
  const run = (compile: boolean, kernels: boolean) => () => {
    toggle(batched, '_setCompileEvalN', compile);
    toggle(batched, '_setScalarKernels', kernels);
    for (let i = 0; i < 3; i++) sampler.evaluateExprN(ir, refArrays, count, baseEnv, undefined);
  };
  const t = interleaved({ loop: run(false, false), kernels: run(false, true), compiled: run(true, true) });
  toggle(batched, '_setCompileEvalN', true);
  toggle(batched, '_setScalarKernels', true);
  report('kernels vs ARITH_OPS loop', t.loop, t.kernels, 1.6);
  report('compiled vs ARITH_OPS loop', t.loop, t.compiled, 2);
  report('compiled vs kernels', t.kernels, t.compiled, 1);
  // 2.4x to 5.1x measured; a process whose kernels V8 does not inline reads
  // about 1.7x. Unspliced kernels read at most 1.43x, no kernels 1.0x.
  assert.ok(t.kernels * 1.6 <= t.loop, `the kernels (${t.kernels.toFixed(1)} ms) must be ≥1.6x faster than the ARITH_OPS loop (${t.loop.toFixed(1)} ms)`);
  assert.ok(t.compiled * 2 <= t.loop, `compiled (${t.compiled.toFixed(1)} ms) must be ≥2x faster than the ARITH_OPS loop (${t.loop.toFixed(1)} ms)`);
  assert.ok(t.compiled <= t.kernels, `compiled (${t.compiled.toFixed(1)} ms) must not be slower than the kernels (${t.kernels.toFixed(1)} ms)`);
});

// The same inverse as a profile body, swept over 20000 points: the scalar
// fast path against the old dispatch path.
test('profile of the MC inverse: the scalar fast path beats the old path', () => {
  const { processSource, orchestrator } = require('../../index.ts');
  const mcr = require('../../mc-recipe.ts');
  const src = fs.readFileSync(path.resolve(__dirname, '../fixtures/simple-transport1.flatppl'), 'utf8');
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const expand = (nm: any) => orchestrator.expandMeasure(nm, { derivations: built.derivations, bindings: built.bindings });
  const mc = mcr.buildMcMarginalForm(expand('zs'), built.bindings, expand).args[0];
  const env: any = { pars: { a: 0.1, b: 0.3, mu: 1.0 }, sigma: 0.2, [mc.outName]: 3.1 };
  const P = 20000;
  const sweep = (prog: any) => () => {
    for (let i = 0; i < P; i++) { env.x = env[mc.retainedRef] = 0.3 + 1.7 * i / P; prog.nextPoint(); prog.evalPoint(env); }
  };
  toggle(profile, '_setScalarFastPath', false);
  const slow = sampler.compileProfileBody(mc.inverseIR);
  toggle(profile, '_setScalarFastPath', true);
  const fast = sampler.compileProfileBody(mc.inverseIR);
  const t = interleaved({ old: sweep(slow), fast: sweep(fast) });
  report('profile fast path vs old path', t.old, t.fast, 2);
  assert.ok(t.fast * 2 <= t.old, `the fast path (${t.fast.toFixed(1)} ms) must be ≥2x faster than the old path (${t.old.toFixed(1)} ms)`);
});
