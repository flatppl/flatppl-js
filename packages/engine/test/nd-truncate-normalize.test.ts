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
