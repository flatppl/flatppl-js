'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const C = require('../sampler-eval-compile.ts');

const ref = (name: any) => ({ kind: 'ref', ns: 'self', name });
const lit = (v: any) => ({ kind: 'lit', value: v });
const call = (op: any, ...args: any[]) => ({ kind: 'call', op, args });

test('_hasPerAtomRef: true when a per-atom name appears', () => {
  const ir = call('add', ref('x'), call('mul', ref('pars'), lit(2)));
  assert.equal(C._hasPerAtomRef(ir, new Set(['x'])), true);
  assert.equal(C._hasPerAtomRef(ir, new Set(['nope'])), false);
});

test('_hasPerAtomRef: nested under a non-compile op still detected', () => {
  const ir = call('get_field', ref('x'), lit('a'));
  assert.equal(C._hasPerAtomRef(ir, new Set(['x'])), true);
});

test('_COMPILE_ARITY: has the scalar op set, excludes structural ops', () => {
  assert.equal(C._COMPILE_ARITY.add, 2);
  assert.equal(C._COMPILE_ARITY.exp, 1);
  assert.equal(C._COMPILE_ARITY.ifelse, 3);
  assert.equal(C._COMPILE_ARITY.tuple, undefined);
  assert.equal(C._COMPILE_ARITY.get_field, undefined);
  assert.equal(C._COMPILE_ARITY.aggregate, undefined);
});

// --- compiler core: a Normal-like inverse expression ---
// Mirror the dependency injection sampler-eval-batched does at init.
const ARITH = {
  add: (a: any, b: any) => a + b, sub: (a: any, b: any) => a - b, mul: (a: any, b: any) => a * b,
  div: (a: any, b: any) => Math.floor(a / b), divide: (a: any, b: any) => a / b, neg: (a: any) => -a,
  pow: (a: any, b: any) => Math.pow(a, b), exp: (a: any) => Math.exp(a), log: (a: any) => Math.log(a),
};
C.initCompiler({ ARITH_OPS: ARITH, evaluateExpr: require('../sampler.ts').evaluateExpr,
                 resolveConst: (n: any) => { throw new Error('no const ' + n); } });

test('compilePlan + runPlan: bit-exact to hand arithmetic over a batch', () => {
  // u = ((cbrt(z*2 / exp(x - b)) - x)/a - 1)/2, with a,b folded scalars.
  // Build it from refs x,z (per-atom) and folded a=0.1,b=0.3.
  const irB = lit(0.3), irA = lit(0.1);
  const ir = call('divide',
    call('sub',
      call('divide',
        call('sub',
          call('pow', call('divide', call('mul', ref('z'), lit(2)),
                            call('exp', call('sub', ref('x'), irB))),
               lit(1 / 3)),
          ref('x')),
        irA),
      lit(1)),
    lit(2));
  const perAtom = new Set(['x', 'z']);
  const plan = C.compilePlan(ir, perAtom);
  assert.ok(plan, 'expression must be compilable');
  const N = 1000;
  const x = new Float64Array(N), z = new Float64Array(N);
  for (let i = 0; i < N; i++) { x[i] = 0.5 + i * 1e-3; z[i] = 1 + i * 2e-3; }
  const refArrays = { x, z };
  const out = C.runPlan(plan, refArrays, {}, null, N);
  assert.equal(out.length, N);
  for (let i = 0; i < N; i++) {
    const inner = (z[i] * 2) / Math.exp(x[i] - 0.3);
    const expected = ((Math.pow(inner, 1 / 3) - x[i]) / 0.1 - 1) / 2;
    assert.equal(out[i], expected, `atom ${i}: ${out[i]} vs ${expected}`);
  }
});

test('compilePlan: returns null for a non-compilable op (get_field with per-atom ref)', () => {
  const ir = call('get_field', ref('x'), lit('a'));   // structural over per-atom x
  assert.equal(C.compilePlan(ir, new Set(['x'])), null);
});

test('compilePlan: returns null for unknown op over per-atom data', () => {
  const ir = call('mystery_op', ref('x'));
  assert.equal(C.compilePlan(ir, new Set(['x'])), null);
});

// --- differential: evaluateExprN with compiler ON must bit-match OFF ---
const batched = require('../sampler-eval-batched.ts');
const sampler = require('../sampler.ts');

function evalBoth(ir: any, refArrays: any, N: any) {
  batched._setCompileEvalN(false);
  const off = sampler.evaluateExprN(ir, refArrays, N, {}, undefined);
  batched._setCompileEvalN(true);
  const on = sampler.evaluateExprN(ir, refArrays, N, {}, undefined);
  return { off, on };
}

test('evaluateExprN: compiled === interpreted, bit-for-bit (real expr)', () => {
  const N = 4096;
  const x = new Float64Array(N), z = new Float64Array(N);
  for (let i = 0; i < N; i++) { x[i] = Math.sin(i) * 2; z[i] = 1 + (i % 50) * 0.1; }
  const ir = call('add',
    call('mul', call('exp', call('sub', ref('x'), lit(0.3))), ref('z')),
    call('log', call('add', call('abs', ref('z')), lit(1))));
  const { off, on } = evalBoth(ir, { x, z }, N);
  for (let i = 0; i < N; i++) assert.equal(on[i], off[i], `atom ${i}`);
  batched._setCompileEvalN(true);
});

test('evaluateExprN: ineligible IR (get_field) falls back, still correct', () => {
  // record with a per-atom field forces the compiler to bail; result
  // must equal the interpreter (which handles get_field per-atom).
  const N = 16;
  const x = new Float64Array(N); for (let i = 0; i < N; i++) x[i] = i;
  const ir = call('add', call('get_field', ref('rec'), lit('v')), lit(1));
  // rec is per-atom records — interpreter path; just assert ON===OFF.
  const recs = { rec: Array.from({ length: N }, (_, i) => ({ v: i })) };
  // Pass records via refArrays as a plain array (back-compat path);
  // compiler bails (get_field), interpreter computes v+1.
  batched._setCompileEvalN(false);
  let off; try { off = sampler.evaluateExprN(ir, recs, N, {}, undefined); } catch (e: any) { off = 'ERR:' + e.message; }
  batched._setCompileEvalN(true);
  let on; try { on = sampler.evaluateExprN(ir, recs, N, {}, undefined); } catch (e: any) { on = 'ERR:' + e.message; }
  if (typeof off === 'string') assert.equal(on, off);
  else for (let i = 0; i < N; i++) assert.equal(on[i], off[i]);
  batched._setCompileEvalN(true);
});

const fc = require('fast-check');
// fc.xorshift128plus does not exist in fast-check 4.x; the RNG lives in
// the peer package pure-rand. Require it via its sub-path export so we
// can pass a seeded RandomGenerator to new fc.Random(rng).
const { xorshift128plus: _prandXor } = require('pure-rand/generator/xorshift128plus');

// Build a random IR over the ops the compiler emits. Leaves are a
// per-atom ref ('x'/'y') or a numeric literal. Depth-bounded.
function randIR(rng: any, depth: any): any {
  const UN = ['neg', 'abs', 'exp', 'log', 'sqrt', 'sin', 'cos', 'tanh', 'log1p', 'expm1'];
  const BIN = ['add', 'sub', 'mul', 'div', 'pow', 'min', 'max', 'atan2'];
  if (depth <= 0 || rng.nextInt(0, 2) === 0) {
    // Only per-atom leaves so every generated IR is eligible (has a
    // per-atom ref) — atom-independent IRs are a separate correctness
    // concern the earlier unit tests already cover.
    return rng.nextInt(0, 1) === 0 ? ref('x') : ref('y');
  }
  if (rng.nextInt(0, 1) === 0) {
    return call(UN[rng.nextInt(0, UN.length - 1)], randIR(rng, depth - 1));
  }
  return call(BIN[rng.nextInt(0, BIN.length - 1)], randIR(rng, depth - 1), randIR(rng, depth - 1));
}

test('fuzz: compiled === interpreted bit-for-bit over random eligible IRs', () => {
  fc.assert(fc.property(fc.integer({ min: 1, max: 2 ** 31 - 1 }), (seed: any) => {
    const rng = new fc.Random(_prandXor(seed, 0));
    const ir = call('add', randIR(rng, 4), lit(0));   // ensure a call root
    const N = 257;
    const x = new Float64Array(N), y = new Float64Array(N);
    for (let i = 0; i < N; i++) { x[i] = (i - 128) * 0.05; y[i] = 0.2 + (i % 13) * 0.07; }
    batched._setCompileEvalN(false);
    const off = sampler.evaluateExprN(ir, { x, y }, N, {}, undefined);
    batched._setCompileEvalN(true);
    const on = sampler.evaluateExprN(ir, { x, y }, N, {}, undefined);
    // Both must be Float64Array(N) and bit-identical (NaN===NaN handled
    // via Object.is so a NaN-producing op like sqrt(neg) still matches).
    if (!(off && off.length === N && on && on.length === N)) return false;
    for (let i = 0; i < N; i++) if (!Object.is(on[i], off[i])) return false;
    return true;
  }), { numRuns: 300 });
  batched._setCompileEvalN(true);
});

test('perf: evaluateExprN compiled is faster than interpreted on the MC inverse', () => {
  const path = require('node:path');
  const ENG = path.resolve(__dirname, '..') + '/';
  const { processSource, orchestrator } = require(ENG + 'index.ts');
  const { deriveMcLikelihoodRecipe } = require(ENG + 'mat-density.ts');
  const bijReg = require(ENG + 'bijection-registry.ts');
  const { createWorkerHandler } = require(ENG + 'worker.ts');
  const fs = require('node:fs');
  const src = fs.readFileSync('test/fixtures/simple-transport.flatppl', 'utf8');
  const { bindings } = processSource(src);
  const built = orchestrator.buildDerivations(bindings);
  const r = deriveMcLikelihoodRecipe('z', ['pars'], built.bindings);
  const OUT = { kind: 'ref', ns: '%mc', name: '__mc_z__' };
  const inv = bijReg.invertExpr({ outputExpr: r.recipeIR, freeRef: r.retainedRef, outputValue: OUT });
  const baseEnv = { pars: { a: 0.1, b: 0.3, mu: 1.0 }, sigma: 0.2 };
  const w = createWorkerHandler(); w.handle({ type: 'init', seed: [1, 2, 3] });
  w.handle({ type: 'setEnv', env: baseEnv, merge: true });
  const data = [3.8, 2.9, 3.2, 3.1, 3.3, 5.0, 0.84, 2.3, 3.1, 2.6, 2.5, 5.4, 1.2, 1.6, 1.4, 0.77, 0.26, 1.6, 0.56, 4.5];
  const M = 6000, D = data.length, count = D * M;
  const xs = w.handle({ type: 'sampleN', ir: r.marginalDistIR, count: M, refArrays: {}, seed: [1, 2, 3, 4] }).samples;
  const zcol = new Float64Array(count), xcol = new Float64Array(count);
  for (let d = 0; d < D; d++) for (let m = 0; m < M; m++) { zcol[d * M + m] = data[d]; xcol[d * M + m] = xs[m]; }
  const refArrays = { [OUT.name]: zcol, [r.retainedRef.name]: xcol, x: xcol };
  const hr = () => Number(process.hrtime.bigint()) / 1e6;
  const RUNS = 9;
  batched._setCompileEvalN(false);
  let off0 = sampler.evaluateExprN(inv.inverseIR, refArrays, count, baseEnv, undefined);  // warm
  let t = hr(); for (let i = 0; i < RUNS; i++) sampler.evaluateExprN(inv.inverseIR, refArrays, count, baseEnv, undefined);
  const tOff = hr() - t;
  batched._setCompileEvalN(true);
  let on0 = sampler.evaluateExprN(inv.inverseIR, refArrays, count, baseEnv, undefined);   // warm + compile
  t = hr(); for (let i = 0; i < RUNS; i++) sampler.evaluateExprN(inv.inverseIR, refArrays, count, baseEnv, undefined);
  const tOn = hr() - t;
  process.stderr.write(`  evaluateExprN inverse x${RUNS}: interpreted ${tOff.toFixed(0)}ms, compiled ${tOn.toFixed(0)}ms (${(tOff / tOn).toFixed(1)}x)\n`);
  // Bit-exact always; the ≥2x ratio assertion is skipped under the
  // kill switch (then both paths are the interpreter → ratio ~1).
  for (let i = 0; i < count; i++) assert.equal(on0[i], off0[i]);
  if (process.env.FLATPPL_NO_EVALN_COMPILE !== '1') {
    assert.ok(tOn * 2 <= tOff, `compiled (${tOn.toFixed(0)}ms) must be ≥2x faster than interpreted (${tOff.toFixed(0)}ms)`);
  }
  batched._setCompileEvalN(true);
});

test('toggle: _setCompileEvalN(false) forces the interpreter path', () => {
  const N = 64; const x = new Float64Array(N); for (let i = 0; i < N; i++) x[i] = i * 0.01;
  const ir = call('exp', ref('x'));
  batched._setCompileEvalN(false);
  const off = sampler.evaluateExprN(ir, { x }, N, {}, undefined);
  batched._setCompileEvalN(true);
  const on = sampler.evaluateExprN(ir, { x }, N, {}, undefined);
  for (let i = 0; i < N; i++) assert.equal(on[i], off[i]);
  batched._setCompileEvalN(true);
});
