'use strict';

// Symbolic inverse + LADJ over FlatPIR (bijection-registry.invertExpr,
// spec §06 case-1). We validate two invariants for each forward map
// f(u; frozen):
//   (1) inverse ∘ forward = identity:  inverseIR(f(u0)) ≈ u0.
//   (2) ladjIR(u0) ≈ log|f'(u0)| by central finite difference.
// Both are checked by evaluating the produced IRs through the engine's
// single-point evaluator (sampler.evaluateExpr), so the test exercises
// the real value-op semantics, not a re-implementation.

const test = require('node:test');
const assert = require('node:assert');
const { invertExpr } = require('../bijection-registry.ts');
const { evaluateExpr } = require('../sampler.ts');

const lit = (v: number) => ({ kind: 'lit', value: v, numType: Number.isInteger(v) ? 'integer' : 'real' });
const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
const call = (op: string, ...args: any[]) => ({ kind: 'call', op, args });
const U = ref('u');           // free variable
const Y = ref('y');           // output placeholder

function evalAt(ir: any, env: any) { return evaluateExpr(ir, env); }

// Numerical forward LADJ: log|f'(u0)| via central difference.
function numericLadj(forwardExpr: any, frozen: any, u0: number) {
  const h = 1e-6;
  const fp = evalAt(forwardExpr, Object.assign({}, frozen, { u: u0 + h }));
  const fm = evalAt(forwardExpr, Object.assign({}, frozen, { u: u0 - h }));
  return Math.log(Math.abs((fp - fm) / (2 * h)));
}

function checkBijection(name: string, forwardExpr: any, frozen: any, u0: number) {
  const res = invertExpr({ outputExpr: forwardExpr, freeRef: { name: 'u' }, outputValue: Y });
  assert.ok(res, `${name}: invertExpr returned null (expected invertible)`);
  // (1) inverse ∘ forward = identity.
  const z0 = evalAt(forwardExpr, Object.assign({}, frozen, { u: u0 }));
  const uRec = evalAt(res!.inverseIR, Object.assign({}, frozen, { y: z0 }));
  assert.ok(Math.abs(uRec - u0) < 1e-6,
    `${name}: inverse∘forward ≠ id — u0=${u0}, recovered=${uRec}`);
  // (2) forward LADJ matches the numerical derivative.
  const ladj = evalAt(res!.ladjIR, Object.assign({}, frozen, { u: u0 }));
  const ladjNum = numericLadj(forwardExpr, frozen, u0);
  assert.ok(Math.abs(ladj - ladjNum) < 1e-4,
    `${name}: ladj ${ladj} ≠ numerical ${ladjNum}`);
}

test('invertExpr: exp', () => {
  checkBijection('exp', call('exp', U), {}, 0.7);
});

test('invertExpr: log', () => {
  checkBijection('log', call('log', U), {}, 2.3);
});

test('invertExpr: neg', () => {
  checkBijection('neg', call('neg', U), {}, 1.4);
});

test('invertExpr: affine 3*(u+2) (mul + add, frozen scalars)', () => {
  // y = 3 * (u + 2). Exercises mul (LADJ log|3|) ∘ add (LADJ 0).
  checkBijection('affine', call('mul', lit(3), call('add', U, lit(2))), {}, 0.9);
});

test('invertExpr: divide both orientations', () => {
  checkBijection('divide x/c', call('divide', U, lit(4)), {}, 1.1);       // free = numerator
  checkBijection('divide a/x', call('divide', lit(5), U), {}, 1.7);       // free = denominator
});

test('invertExpr: pow literal exponent (u+1)^3', () => {
  checkBijection('pow3', call('pow', call('add', U, lit(1)), lit(3)), {}, 0.6);
});

test('invertExpr: transport map (x+δ)^3·exp(x−b), δ=(2u+1)·a', () => {
  // The driver: y = (x + (2u+1)·a)^3 · exp(x − b), u the internal
  // Uniform, x/a/b frozen (x is the MC-marginalised gun draw; a,b the
  // record params). A composition of mul/add/pow/exp/divide registry
  // bijections — must invert symbolically end-to-end.
  const delta = call('mul', call('add', call('mul', lit(2), U), lit(1)), ref('a'));
  const forward = call('mul',
    call('pow', call('add', ref('x'), delta), lit(3)),
    call('exp', call('sub', ref('x'), ref('b'))));
  const frozen = { x: 1.0, a: 0.1, b: 0.3 };
  checkBijection('transport', forward, frozen, 0.5);
  checkBijection('transport@0.2', forward, frozen, 0.2);
  checkBijection('transport@0.85', forward, frozen, 0.85);
});

test('invertExpr: refuses non-bijective / non-registry shapes', () => {
  // u appears in two arguments → not single-input straight-line.
  assert.strictEqual(
    invertExpr({ outputExpr: call('mul', U, U), freeRef: { name: 'u' }, outputValue: Y }), null);
  // unknown op (no registry rule).
  assert.strictEqual(
    invertExpr({ outputExpr: call('sin', U), freeRef: { name: 'u' }, outputValue: Y }), null);
  // pow with non-literal exponent (base-free but exponent is a ref).
  assert.strictEqual(
    invertExpr({ outputExpr: call('pow', U, ref('k')), freeRef: { name: 'u' }, outputValue: Y }), null);
  // free variable absent.
  assert.strictEqual(
    invertExpr({ outputExpr: call('exp', lit(1)), freeRef: { name: 'u' }, outputValue: Y }), null);
});
