'use strict';

// Spec §07 signal-processing: 1-D valid-mode convolution &
// cross-correlation.
//
//   conv(v, k)_i      = ⟨ v[i : i+K-1], reverse(k) ⟩
//   crosscorr(v, k)_i = ⟨ v[i : i+K-1], k ⟩
//
// Result length = lengthof(v) − lengthof(k) + 1.

const { test } = require('node:test');
const assert = require('node:assert/strict');

const sampler = require('../sampler.ts');
const { toJS } = require('./_value-helpers.ts');

function lit(v: any)        { return { kind: 'lit', value: v }; }
function vec(...vs: any[])  { return { kind: 'call', op: 'vector', args: vs.map(lit) }; }
function call(op: any, ...args: any[]) { return { kind: 'call', op, args }; }
const evRaw = (ir: any) => sampler.evaluateExpr(ir, {});
const ev = (ir: any) => toJS(evRaw(ir));

// =====================================================================
// conv — spec example
// =====================================================================

test('conv([1,2,3,4], [1,0,-1]) ⇒ [2, 2] (spec example)', () => {
  // reverse(kernel) = [-1, 0, 1]
  // window 0: [1,2,3] · [-1,0,1] = -1+0+3 = 2
  // window 1: [2,3,4] · [-1,0,1] = -2+0+4 = 2
  assert.deepEqual(ev(call('conv', vec(1, 2, 3, 4), vec(1, 0, -1))),
    [2, 2]);
});

test('conv with kernel = [1]: identity (no change)', () => {
  assert.deepEqual(ev(call('conv', vec(5, 6, 7), vec(1))),
    [5, 6, 7]);
});

test('conv with kernel length = vector length ⇒ single output', () => {
  // reverse(kernel) = [3, 2, 1]; window 0 = [1,2,3] · [3,2,1] = 10
  const r = ev(call('conv', vec(1, 2, 3), vec(1, 2, 3)));
  assert.deepEqual(r, [10]);
});

test('conv: kernel longer than vector ⇒ error', () => {
  assert.throws(() => ev(call('conv', vec(1, 2), vec(1, 2, 3))),
    /kernel length.*> vector length/);
});

test('conv: non-vector argument ⇒ error', () => {
  assert.throws(() => ev(call('conv',
    call('rowstack',
      { kind: 'call', op: 'vector',
        args: [vec(1, 2), vec(3, 4)] }),
    vec(1))),
    /both arguments must be rank-1 vectors/);
});

// =====================================================================
// crosscorr — spec example
// =====================================================================

test('crosscorr([1,2,3,4], [1,0,-1]) ⇒ [-2, -2] (spec example)', () => {
  // window 0: [1,2,3] · [1,0,-1] = 1+0-3 = -2
  // window 1: [2,3,4] · [1,0,-1] = 2+0-4 = -2
  assert.deepEqual(ev(call('crosscorr', vec(1, 2, 3, 4), vec(1, 0, -1))),
    [-2, -2]);
});

test('crosscorr with kernel = reverse(k) of a conv kernel matches conv', () => {
  // crosscorr(v, k) == conv(v, reverse(k))
  // Take k = [1, 0, -1]; reverse = [-1, 0, 1].
  // conv(v, [-1, 0, 1]) gradient detector;
  // crosscorr(v, [1, 0, -1]) same effective filter.
  const v = vec(2, 4, 6, 8, 10);
  const k = vec(1, 0, -1);
  const cc = ev(call('crosscorr', v, k));
  const cv = ev(call('conv', v, vec(-1, 0, 1)));
  assert.deepEqual(cc, cv);
});

test('crosscorr: equal-length input/kernel ⇒ single inner product', () => {
  // [1,2,3] · [4,5,6] = 4+10+18 = 32
  assert.deepEqual(ev(call('crosscorr', vec(1, 2, 3), vec(4, 5, 6))),
    [32]);
});

// =====================================================================
// Sanity: result length identity
// =====================================================================

test('conv/crosscorr result length = n − K + 1', () => {
  // n=10, K=3 → 8 outputs
  const v = vec(1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
  const k = vec(1, 1, 1);
  const c1 = ev(call('conv', v, k));
  const c2 = ev(call('crosscorr', v, k));
  assert.equal(c1.length, 8);
  assert.equal(c2.length, 8);
  // Symmetric kernel ⇒ conv == crosscorr.
  assert.deepEqual(c1, c2);
});
