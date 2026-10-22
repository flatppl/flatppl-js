'use strict';

// `broadcast(logdensityof, M, pts)` scores all points in one batched pass whose N
// axis indexes the evaluation POINTS (density.ts `pointsBatched`): atom i scores
// `pts[i]`. `walkTruncate` gated the support on the first point only, so one
// out-of-support first point sent every point to -Infinity, and an in-support
// first point let every out-of-support point keep its density.
//
// ORACLE: closed forms, computed by hand in Python with math.erf, not by the
// engine. With Z the mass of Normal(0, 1) on the set,
//   truncate(Normal, S)             at x in S:  log N(x; 0, 1)
//   normalize(truncate(Normal, S))  at x in S:  log N(x; 0, 1) - log Z
//   either, at x not in S:                      -Infinity
// Z(interval(0, 5)) = Phi(5) - Phi(0) = 0.4999997133484282, Z(posreals) = 0.5.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ctxFor } = require('./_ctx-factory.ts');
const density = require('../density.ts');
const { parseSetIR } = require('../ir-shared.ts');
const { processSource } = require('../index.ts');

const RAW: Record<string, number> = { '0.5': -1.0439385332046727, '1': -1.4189385332046727, '2': -2.9189385332046727 };
const NORM: Record<string, Record<string, number>> = {
  'interval(0, 5)': { '0.5': -0.35079077934141945, '1': -0.7257907793414194, '2': -2.2257907793414193 },
  posreals: { '0.5': -0.3507913526447274, '1': -0.7257913526447274, '2': -2.2257913526447273 },
};
const INSIDE: Record<string, (x: number) => boolean> = {
  'interval(0, 5)': (x) => x >= 0 && x <= 5,
  posreals: (x) => x > 0,
};

async function scoreAll(measure: string, pts: number[]) {
  const src = `M = ${measure}\nlp = broadcast(logdensityof, M, [${pts.join(', ')}])\n`;
  const { ctx } = ctxFor(src, 1);
  const m = await ctx.getMeasure('lp');
  return Array.from((m.value?.data ?? m.samples) as ArrayLike<number>);
}

// Two orderings put an out-of-support point first and an in-support point first.
const ORDERS = [[-1, 0.5, 1, 6, 2], [0.5, -1, 1, 6, 2], [6, 0.5, 1, -1, 2]];

for (const set of Object.keys(INSIDE)) {
  for (const normalized of [false, true]) {
    test(`${normalized ? 'normalize(truncate' : 'truncate'}(Normal(0, 1), ${set})${normalized ? ')' : ''} gates each point on its own membership`, async () => {
      const measure = normalized
        ? `normalize(truncate(Normal(0, 1), ${set}))` : `truncate(Normal(0, 1), ${set})`;
      for (const pts of ORDERS) {
        const got = await scoreAll(measure, pts);
        pts.forEach((x, i) => {
          if (!INSIDE[set](x)) {
            assert.equal(got[i], -Infinity, `${set} x=${x} in order ${pts}: got ${got[i]}`);
          } else if (String(x) in RAW) {
            const want = normalized ? NORM[set][String(x)] : RAW[String(x)];
            assert.ok(Math.abs(got[i] - want) < 1e-12, `${set} x=${x} in order ${pts}: got ${got[i]}, want ${want}`);
          }
        });
      }
    });
  }
}

test('a vector set gates each point whole, not the first points as one vector', () => {
  // `truncate(Lebesgue(R^3), stdsimplex(3))` has density 0 (log) on the simplex.
  const ir = processSource('M = truncate(Lebesgue(support = cartpow(reals, 3)), stdsimplex(3))\n')
    .loweredModule.bindings.get('M').rhs;
  const pts = [[0.2, 0.3, 0.5], [0.5, 0.5, 0.5], [0.1, 0.1, 0.8], [0.9, 0.2, -0.1]];
  const opts = { parseSet: (s: any) => parseSetIR(s, null) };
  const batched: number[] = Array.from(density.logDensityN(ir, pts, {}, pts.length, { ...opts, pointsBatched: true }));
  const perPoint: number[] = pts.map((p) => density.logDensityN(ir, p, {}, 1, opts)[0]);
  // The last point sums to 1 but has a negative coordinate, so it is off the simplex.
  assert.deepEqual(perPoint, [0, -Infinity, 0, -Infinity]);
  assert.deepEqual(batched, perPoint);
});

test('points held as a Value are gated per point too', () => {
  const values = require('../value.ts');
  const opts = { parseSet: (s: any) => parseSetIR(s, null), pointsBatched: true };
  const scalar = processSource('M = truncate(Normal(0, 1), interval(0, 5))\n')
    .loweredModule.bindings.get('M').rhs;
  const xs = [-1, 0.5, 6, 1];
  const got: number[] = Array.from(density.logDensityN(scalar, values.asValue(xs), {}, xs.length, opts));
  assert.deepEqual(got.map((v: number) => v === -Infinity), [true, false, true, false]);
  assert.ok(Math.abs(got[1] - RAW['0.5']) < 1e-12 && Math.abs(got[3] - RAW['1']) < 1e-12);

  const vector = processSource('M = truncate(Lebesgue(support = cartpow(reals, 3)), stdsimplex(3))\n')
    .loweredModule.bindings.get('M').rhs;
  const pts = [[0.5, 0.5, 0.5], [0.2, 0.3, 0.5], [0.1, 0.1, 0.8]];
  const vgot: number[] = Array.from(density.logDensityN(vector, values.asValue(pts), {}, pts.length, opts));
  assert.deepEqual(vgot, [-Infinity, 0, 0]);
});

test('an upper bound that varies per atom is gated per atom against its own point', () => {
  const ir = processSource('hi = elementof(reals)\nM = truncate(Normal(0, 1), interval(0, hi))\n')
    .loweredModule.bindings.get('M').rhs;
  const opts = { parseSet: (s: any) => parseSetIR(s, null), pointsBatched: true };
  const got: number[] = Array.from(density.logDensityN(ir, [1, 3, 1, 3], { hi: new Float64Array([2, 2, 5, 5]) }, 4, opts));
  assert.deepEqual(got.map((v: number) => Number.isFinite(v)), [true, false, true, true]);
  assert.ok(Math.abs(got[0] - RAW['1']) < 1e-12);
});

test('a lower bound that varies per atom, and bounds that both vary, gate per atom too', () => {
  const opts = { parseSet: (s: any) => parseSetIR(s, null), pointsBatched: true };
  const lower = processSource('lo = elementof(reals)\nM = truncate(Normal(0, 1), interval(lo, 5))\n')
    .loweredModule.bindings.get('M').rhs;
  const gotLo: number[] = Array.from(density.logDensityN(lower, [1, 1, 3, 3], { lo: new Float64Array([0, 2, 2, 4]) }, 4, opts));
  assert.deepEqual(gotLo.map((v: number) => Number.isFinite(v)), [true, false, true, false]);
  const both = processSource('lo = elementof(reals)\nhi = elementof(reals)\nM = truncate(Normal(0, 1), interval(lo, hi))\n')
    .loweredModule.bindings.get('M').rhs;
  const refs = { lo: new Float64Array([0, 2, 0, 4]), hi: new Float64Array([2, 5, 0.5, 5]) };
  const gotBoth: number[] = Array.from(density.logDensityN(both, [1, 1, 1, 3], refs, 4, opts));
  assert.deepEqual(gotBoth.map((v: number) => Number.isFinite(v)), [true, false, false, false]);
});
