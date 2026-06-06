'use strict';
const { test } = require('node:test');
const assert = require('node:assert/strict');
const C = require('../sampler-eval-compile.ts');

const ref = (name) => ({ kind: 'ref', ns: 'self', name });
const lit = (v) => ({ kind: 'lit', value: v });
const call = (op, ...args) => ({ kind: 'call', op, args });

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
  add: (a, b) => a + b, sub: (a, b) => a - b, mul: (a, b) => a * b,
  div: (a, b) => a / b, divide: (a, b) => a / b, neg: (a) => -a,
  pow: (a, b) => Math.pow(a, b), exp: (a) => Math.exp(a), log: (a) => Math.log(a),
};
C.initCompiler({ ARITH_OPS: ARITH, evaluateExpr: require('../sampler.ts').evaluateExpr,
                 resolveConst: (n) => { throw new Error('no const ' + n); } });

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

function evalBoth(ir, refArrays, N) {
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
  let off; try { off = sampler.evaluateExprN(ir, recs, N, {}, undefined); } catch (e) { off = 'ERR:' + e.message; }
  batched._setCompileEvalN(true);
  let on; try { on = sampler.evaluateExprN(ir, recs, N, {}, undefined); } catch (e) { on = 'ERR:' + e.message; }
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
function randIR(rng, depth) {
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
  fc.assert(fc.property(fc.integer({ min: 1, max: 2 ** 31 - 1 }), (seed) => {
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
