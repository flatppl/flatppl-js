'use strict';

// parseTruncationBox (mat-density.ts) — set IR → per-axis truncation box for
// N-D quadrature (engine G2). Parses interval(...) (1-D) and cartprod(...)
// of intervals (N-D) into { lo, hi, kind } axes; null for any other set
// shape (caller defers to the existing by-name/normalizer path); throws
// above the dimension cap of 3. Consumer wiring is a separate task.

const test = require('node:test');
const assert = require('node:assert');
const matDensity = require('../mat-density.ts');
const { parseTruncationBox } = matDensity;

const iv = (lo: any, hi: any) => ({ kind: 'call', op: 'interval', args: [lit(lo), lit(hi)] });
const lit = (v: any) => ({ kind: 'lit', value: v });
const cart = (...xs: any[]) => ({ kind: 'call', op: 'cartprod', args: xs });

test('1-D interval → one finite axis', () => {
  assert.deepEqual(parseTruncationBox(iv(-2, 3)),
    [{ lo: -2, hi: 3, kind: 'finite' }]);
});

test('2-D cartprod of intervals → two finite axes', () => {
  assert.deepEqual(parseTruncationBox(cart(iv(0, 1), iv(-4, 4))),
    [{ lo: 0, hi: 1, kind: 'finite' }, { lo: -4, hi: 4, kind: 'finite' }]);
});

test('semi-infinite interval(0, inf) → semi-lo', () => {
  assert.deepEqual(parseTruncationBox(iv(0, Infinity)),
    [{ lo: 0, hi: Infinity, kind: 'semi-lo' }]);
});

test('unrecognized set shape → null', () => {
  assert.equal(parseTruncationBox({ kind: 'call', op: 'stdsimplex', args: [lit(3)] }), null);
  assert.equal(parseTruncationBox(cart(iv(0, 1), { kind: 'ref', name: 'S' })), null);
});

test('over-cap dimension throws', () => {
  assert.throws(() => parseTruncationBox(cart(iv(0, 1), iv(0, 1), iv(0, 1), iv(0, 1))),
    /dimension 4 exceeds .*cap.*3/i);
});

// makeIntegrandND — weight IR + change-of-variables (Task 3). The weight
// body IR here is hand-built to the shape `samplerLib.evaluateExpr` (sampler.ts)
// actually consumes: `{kind:'lit',value}`, `{kind:'ref',name}` (resolveRef
// only reads `ir.name`, ns-agnostic), and `{kind:'call',op,args}` dispatched
// through ARITH_OPS. Confirmed against a REAL lowering — processSource('w =
// x -> 1.0/(1.0+x*x)') then orchestrator.buildDerivations(...).bindings.get
// ('w').ir — which lowers the surface `/` to op **'divide'** (true division),
// NOT 'div' (spec §07: `div` is integer floor division, `Math.floor(a/b)` in
// sampler.ts's ARITH_OPS — using it here would silently compute
// floor(1/(1+x²)) ≡ 0 a.e. and the π/2 test below would not integrate to
// the intended function at all).
const { makeIntegrandND } = matDensity;
const { adaptiveCubature } = require('../quadrature.ts');

// weight body IR for  1  (constant) as a functionof body; evaluateExpr resolves
// a bare literal. Use the sampler evaluator's literal path.
function litBody(v: number) { return { kind: 'lit', value: v }; }

test('makeIntegrandND: finite 2-D box of constant 1 integrates to area', () => {
  const axes = [{ lo: 0, hi: 2, kind: 'finite' }, { lo: -1, hi: 3, kind: 'finite' }];
  const integ = makeIntegrandND(litBody(1), ['x', 'y'], axes, {}, {});
  const r = adaptiveCubature(integ, 2);
  assert.ok(Math.abs(r.Z - 8.0) < 1e-8, `Z=${r.Z}`); // area 2×4
});

test('makeIntegrandND: semi-lo change-of-variables integrates 1/(1+x²) over (0,∞) = π/2', () => {
  // weight body: 1/(1+x*x). Build via the sampler expr IR the engine uses.
  const body = { kind: 'call', op: 'divide', args: [
    { kind: 'lit', value: 1 },
    { kind: 'call', op: 'add', args: [{ kind: 'lit', value: 1 },
      { kind: 'call', op: 'mul', args: [{ kind: 'ref', name: 'x' }, { kind: 'ref', name: 'x' }] }] },
  ] };
  const axes = [{ lo: 0, hi: Infinity, kind: 'semi-lo' }];
  const integ = makeIntegrandND(body, ['x'], axes, {}, {});
  const r = adaptiveCubature(integ, 1);
  assert.ok(Math.abs(r.Z - Math.PI / 2) < 1e-6, `Z=${r.Z} vs ${Math.PI / 2}`);
});
