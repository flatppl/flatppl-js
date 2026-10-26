'use strict';

// The single-point profile compiler (sampler-profile-compile.ts) calls the
// fused-loop compiler's plain SCALAR_OPS when every argument of a scalar op is
// a number at that point. Otherwise it keeps its old path: `ops.dispatch` for
// a declared op, the ARITH_OPS entry for add/sub/mul/neg. Both must give the
// same result. These tests compare SCALAR_OPS with that reference on numeric
// edge inputs, compiled bodies with the interpreter on numbers and on values
// that must take the old path, and time the fast path against the old one.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const ops = require('../ops.ts');
const sampler = require('../sampler.ts');
const valueLib = require('../value.ts');
const profile = require('../sampler-profile-compile.ts');
const { SCALAR_OPS } = require('../sampler-eval-compile.ts');

const ARITH_OPS = sampler._internal.ARITH_OPS;
const EDGE = [
  0, -0, 1, -1, 2, -3, 3, 7, -7, 0.5, -0.5, 2.5, -2.5, 0.25, 0.75, 1.5,
  Number.MIN_VALUE, -Number.MIN_VALUE, 2.2250738585072014e-308,
  Number.MAX_VALUE, -Number.MAX_VALUE, Infinity, -Infinity, NaN,
];

function tuples(arity: number, vals: any[]): any[][] {
  if (arity === 1) return vals.map((a) => [a]);
  const inner = tuples(arity - 1, vals);
  const out: any[][] = [];
  for (const a of vals) for (const t of inner) out.push([a, ...t]);
  return out;
}

function outcome(fn: () => any) {
  try { return { value: fn() }; } catch (e: any) { return { threw: e.message }; }
}

function sameOutcome(a: any, b: any) {
  if ('threw' in a || 'threw' in b) return a.threw === b.threw;
  if (a.value && typeof a.value === 'object') return JSON.stringify(a.value) === JSON.stringify(b.value);
  return Object.is(a.value, b.value);
}

// The profile path's own reference: ops.dispatch for a declared op, else ARITH_OPS.
const reference = (op: string, args: any[]) =>
  (ops.isDeclared(op) === true ? ops.dispatch(op, args) : ARITH_OPS[op](...args));

for (const [op, arity] of Object.entries(ops.REAL_SCALAR_PRIM_ARITY) as [string, number][]) {
  test(`SCALAR_OPS.${op} equals the profile compiler's reference on numeric edge inputs`, () => {
    for (const args of tuples(arity, EDGE)) {
      const want = outcome(() => reference(op, args));
      const got = outcome(() => SCALAR_OPS[op](...args));
      assert.ok(sameOutcome(got, want), `${op}(${args.join(', ')})`);
    }
  });
}

// Bodies op(a, b, c) compiled once and evaluated per point must equal the
// interpreter. Mixed points put a boolean, a rank-0 Value or a complex value
// in one slot, which must take the old path.
test('a compiled scalar call equals the interpreter on numbers and on non-number points', () => {
  const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
  const names = ['a', 'b', 'c'];
  const others: any[] = [true, false, valueLib.scalar(2.5), { re: 1, im: -2 }];
  for (const [op, arity] of Object.entries(ops.REAL_SCALAR_PRIM_ARITY) as [string, number][]) {
    const ir = { kind: 'call', op, args: names.slice(0, arity).map(ref) };
    const prog = sampler.compileProfileBody(ir);
    const points = tuples(arity, [0.5, -2.5, 3, NaN, Infinity]);
    for (const o of others) for (let k = 0; k < arity; k++) points.push(points[0].map((v, j) => (j === k ? o : v)));
    for (const pt of points) {
      const env: any = {};
      pt.forEach((v, j) => { env[names[j]] = v; });
      prog.nextPoint();
      const got = outcome(() => prog.evalPoint(env));
      const want = outcome(() => sampler.evaluateExpr(ir, env));
      assert.ok(sameOutcome(got, want), `${op}(${pt.map(String).join(', ')}): ${JSON.stringify(got)} vs ${JSON.stringify(want)}`);
    }
  }
});

// The scalar MC inverse of simple-transport as a profile body: 20000 points
// swept over x. The fast path must beat the old dispatch path. batserv02
// measured a 2000-point profileN sweep at 7.35 -> 3.55 ms, and this loop runs
// 3.6-3.9x faster; the gate is lower so shared CI runners hold it.
test('perf: the scalar fast path makes a profile of the MC inverse at least 2x faster', () => {
  const { processSource, orchestrator } = require('../index.ts');
  const mcr = require('../mc-recipe.ts');
  const src = fs.readFileSync(path.resolve(__dirname, 'fixtures/simple-transport1.flatppl'), 'utf8');
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  const expand = (nm: any) => orchestrator.expandMeasure(nm, { derivations: built.derivations, bindings: built.bindings });
  const mc = mcr.buildMcMarginalForm(expand('zs'), built.bindings, expand).args[0];
  const env: any = { pars: { a: 0.1, b: 0.3, mu: 1.0 }, sigma: 0.2, [mc.outName]: 3.1 };
  const P = 20000;
  const sweep = (prog: any) => {
    let acc = 0;
    for (let i = 0; i < P; i++) { env.x = env[mc.retainedRef] = 0.3 + 1.7 * i / P; prog.nextPoint(); acc += prog.evalPoint(env); }
    return acc;
  };
  profile._setScalarFastPath(false);
  const slow = sampler.compileProfileBody(mc.inverseIR);
  profile._setScalarFastPath(true);
  const fast = sampler.compileProfileBody(mc.inverseIR);
  assert.ok(Object.is(sweep(fast), sweep(slow)), 'same sum');
  const hr = () => Number(process.hrtime.bigint()) / 1e6;
  const time = (prog: any) => { const t = hr(); sweep(prog); return hr() - t; };
  const on: number[] = [], off: number[] = [];
  for (let k = 0; k < 7; k++) { off.push(time(slow)); on.push(time(fast)); }
  const med = (a: number[]) => a.slice().sort((x, y) => x - y)[a.length >> 1];
  process.stderr.write(`  profile MC inverse x${P}: old path ${med(off).toFixed(1)}ms, fast path ${med(on).toFixed(1)}ms (${(med(off) / med(on)).toFixed(2)}x)\n`);
  assert.ok(med(on) * 2 <= med(off), `fast path (${med(on).toFixed(1)}ms) must be ≥2x faster than the old path (${med(off).toFixed(1)}ms)`);
});
