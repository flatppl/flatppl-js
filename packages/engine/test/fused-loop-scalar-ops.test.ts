'use strict';

// The fused-loop compiler (sampler-eval-compile.ts) runs on its own scalar op
// table, SCALAR_OPS, instead of the shared ARITH_OPS entries: those also take
// Values, arrays, complex numbers and records, so V8 could not inline them in
// the generated loop, and ARITH_OPS (128+ keys) sits in dictionary mode. Each
// SCALAR_OPS entry must equal the scalar arm of its ARITH_OPS entry bit for
// bit. These tests check that for every op the compiler accepts, on edge
// inputs: ±0, ±Infinity, NaN, subnormals, the largest finite values,
// integers vs non-integers (div, mod, round, integer), and booleans, which
// reach the loop through folded constants and comparison results.

const { test } = require('node:test');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');
const path = require('node:path');

const compile = require('../sampler-eval-compile.ts');
const sampler = require('../sampler.ts');
const batched = require('../sampler-eval-batched.ts');
const ops = require('../ops.ts');

const ARITH_OPS = sampler._internal.ARITH_OPS;
const SCALAR_OPS = compile.SCALAR_OPS;

const EDGE: any[] = [
  0, -0, 1, -1, 2, -3, 3, 7, -7, 0.5, -0.5, 2.5, -2.5, 0.25, 0.75, 1.5,
  Number.MIN_VALUE, -Number.MIN_VALUE, 2.2250738585072014e-308,
  Number.MAX_VALUE, -Number.MAX_VALUE, Infinity, -Infinity, NaN, true, false,
];

function outcome(fn: any, args: any[]) {
  try { return { value: fn(...args) }; } catch (e: any) { return { threw: e.message }; }
}

function same(a: any, b: any) {
  if ('threw' in a || 'threw' in b) return a.threw === b.threw;
  return Object.is(a.value, b.value);
}

function tuples(arity: number): any[][] {
  if (arity === 1) return EDGE.map((a) => [a]);
  const inner = tuples(arity - 1);
  const out: any[][] = [];
  for (const a of EDGE) for (const t of inner) out.push([a, ...t]);
  return out;
}

test('SCALAR_OPS covers exactly the ops the compiler accepts, at their arities', () => {
  assert.deepEqual(Object.keys(SCALAR_OPS).sort(), Object.keys(ops.REAL_SCALAR_PRIM_ARITY).sort());
  for (const [op, arity] of Object.entries(ops.REAL_SCALAR_PRIM_ARITY)) {
    assert.equal(SCALAR_OPS[op].length, arity, op);
  }
});

for (const [op, arity] of Object.entries(ops.REAL_SCALAR_PRIM_ARITY) as [string, number][]) {
  test(`SCALAR_OPS.${op} equals the scalar arm of ARITH_OPS.${op} on edge inputs`, () => {
    for (const args of tuples(arity)) {
      const want = outcome(ARITH_OPS[op], args);
      const got = outcome(SCALAR_OPS[op], args);
      assert.ok(same(got, want), `${op}(${args.map(String).join(', ')}): `
        + `${JSON.stringify(got, (_, v) => (typeof v === 'number' ? String(v) : v))} vs `
        + `${JSON.stringify(want, (_, v) => (typeof v === 'number' ? String(v) : v))}`);
    }
  });
}

// The generated loops splice SCALAR_INLINE bodies in place of SCALAR_OPS
// calls, so each spliced body must equal its SCALAR_OPS entry.
for (const op of Object.keys(compile.SCALAR_INLINE)) {
  test(`the spliced body of ${op} equals SCALAR_OPS.${op} on edge inputs`, () => {
    const arity = (ops.REAL_SCALAR_PRIM_ARITY as any)[op];
    assert.ok(arity >= 1, op + ' is a compiled op');
    const names = ['x', 'y', 'z'].slice(0, arity);
    // eslint-disable-next-line no-new-func
    const spliced = new Function(...names, 'return ' + compile._scalarSource(op, names, 'OPS') + ';');
    assert.doesNotMatch(spliced.toString(), /OPS\./, op + ' is spliced, not called');
    for (const args of tuples(arity)) {
      const want = outcome(SCALAR_OPS[op], args);
      const got = outcome(spliced, args);
      assert.ok(same(got, want), `${op}(${args.map(String).join(', ')}): ${String(got.value)} vs ${String(want.value)}`);
    }
  });
}

// The same inputs through the generated loop (per-atom Float64Array refs)
// and through the interpreter: equal element for element.
test('every compiled op equals the interpreter on edge inputs', () => {
  const nums = EDGE.filter((v) => typeof v === 'number');
  const ref = (name: string) => ({ kind: 'ref', ns: 'self', name });
  for (const [op, arity] of Object.entries(ops.REAL_SCALAR_PRIM_ARITY) as [string, number][]) {
    if (op === 'integer' || op === 'boolean') continue;   // throw on most inputs; covered above
    const rows = tuples(arity).filter((t) => t.every((v) => typeof v === 'number'));
    const N = rows.length;
    const refArrays: any = {};
    const names = ['x', 'y', 'z'].slice(0, arity);
    names.forEach((n, k) => { refArrays[n] = Float64Array.from(rows.map((r) => r[k])); });
    const ir = { kind: 'call', op, args: names.map(ref) };
    assert.ok(compile.compilePlan(ir, new Set(names)), `${op} compiles`);
    batched._setCompileEvalN(false);
    const off = sampler.evaluateExprN(ir, refArrays, N, {}, undefined);
    batched._setCompileEvalN(true);
    const on = sampler.evaluateExprN(ir, refArrays, N, {}, undefined);
    for (let i = 0; i < N; i++) {
      assert.ok(Object.is(on[i], off[i]), `${op}(${rows[i].join(', ')}): compiled ${on[i]} vs interpreted ${off[i]}`);
    }
    assert.ok(nums.length > 0);
  }
  batched._setCompileEvalN(true);
});

// V8 keeps an object literal with 128 or more properties in dictionary mode,
// where each `OPS.<op>(…)` in the loop is a hash lookup. That cliff cost the
// loop 43 ms when ARITH_OPS crossed it. Keep the loop's table fast.
test('SCALAR_OPS keeps fast properties', () => {
  const file = path.resolve(__dirname, '..', 'sampler-eval-compile.ts');
  const out = execFileSync(process.execPath, ['--allow-natives-syntax', '-e',
    `const m = require(${JSON.stringify(file)}); process.stdout.write(String(%HasFastProperties(m.SCALAR_OPS)))`],
  { encoding: 'utf8' });
  assert.equal(out, 'true');
});
