// Tests for `broadcasted(<scalar_op>)` engine primitives (engine-concepts
// §20.1 follow-up, 2026-05-28). The `_broadcastLogical` fast-path in
// ops-declarations.ts recognises `broadcast(<scalar_op>, args)` and the
// equivalent dotted-operator / `op.(…)` lowered shapes, dispatching
// through value-ops' batched-elementwise impls (mulElem / divElem /
// powElem / expElem / logElem / …).
//
// What's specifically pinned here:
//   - `A .* B` and `broadcasted(mul)(A, B)` for fixed-phase vectors and
//     matrices yield elementwise products (NOT spec-`mul` matrix
//     product).
//   - Scalar-broadcast cases (rank-0 × rank-N) work.
//   - `exp.(A)` / `log.(A)` for any-rank flat arrays.
//   - Spec §04 singleton-axis expansion ([2,3] .* [1,3], addaxes-
//     aligned operands) dispatches on the SAME fast path (stride-0
//     reads in the value-ops elementwise impls; the shared
//     `_broadcastOutShape` combiner owns the equal-or-1 rule).
//   - Inputs the fast path must NOT claim still ride the cold
//     `_broadcastApply` path: nested-vector (Ref-wrap) args,
//     outerRank-tagged vec-of-vec Values (per-cell sees the inner
//     vector WHOLE — §04 outer-axis semantics), incompatible ranks/
//     sizes (so the cold path's spec diagnostics fire).

const test = require('node:test');
const assert = require('node:assert');
const { processSource } = require('../index.ts');
const sampler = require('../sampler.ts');

// Evaluate a single binding's RHS to a JS-friendly value. Pre-evaluates
// all upstream bindings into the env so refs resolve correctly. Only
// works for fixed-phase programs (no `elementof` / `draw`).
function evalBinding(src: string, name: string): any {
  const proc = processSource(src);
  const env: Record<string, any> = {};
  for (const [n, b] of proc.loweredModule.bindings) {
    if (n === name) {
      return sampler.evaluateExpr(b.rhs, env);
    }
    env[n] = sampler.evaluateExpr(b.rhs, env);
  }
  throw new Error('no binding ' + name);
}

function toArray(v: any): number[] {
  if (v && v.data instanceof Float64Array) return Array.from(v.data);
  if (v instanceof Float64Array) return Array.from(v);
  if (Array.isArray(v)) return v.slice();
  throw new Error('toArray: unsupported shape');
}

// =====================================================================
// Binary multiplicative — fast path replaces what would otherwise
// throw / cold-loop.
// =====================================================================

test('broadcasted(mul) on length-3 vectors: elementwise (not matmul)', () => {
  const r = evalBinding(`
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
Y = broadcasted(mul)(A, B)
`, 'Y');
  assert.deepEqual(toArray(r), [4, 10, 18]);
});

test('A .* B for fixed length-3 vectors: elementwise via fast-path', () => {
  const r = evalBinding(`
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
Y = A .* B
`, 'Y');
  assert.deepEqual(toArray(r), [4, 10, 18]);
});

test('A .* B for two 2×2 matrices: elementwise (Hadamard), NOT matmul', () => {
  // Spec-mul on rank-2 × rank-2 is matrix product. The dotted form
  // `A .* B` is broadcast(mul, A, B) = elementwise per-cell mul. The
  // fast-path engine primitive for broadcasted(mul) computes the
  // Hadamard product.
  const r = evalBinding(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[5.0, 6.0], [7.0, 8.0]])
Y = A .* B
`, 'Y');
  // Hadamard: [[1*5, 2*6], [3*7, 4*8]] = [[5, 12], [21, 32]].
  assert.deepEqual(r.shape, [2, 2]);
  assert.deepEqual(toArray(r), [5, 12, 21, 32]);
});

test('A ./ B for fixed vectors: elementwise', () => {
  const r = evalBinding(`
A = [10.0, 20.0, 30.0]
B = [2.0, 4.0, 5.0]
Y = A ./ B
`, 'Y');
  assert.deepEqual(toArray(r), [5, 5, 6]);
});

test('A .^ B for fixed vectors: elementwise', () => {
  const r = evalBinding(`
A = [2.0, 3.0, 4.0]
B = [3.0, 2.0, 0.5]
Y = A .^ B
`, 'Y');
  assert.deepEqual(toArray(r), [8, 9, 2]);
});

// =====================================================================
// Scalar broadcast (rank-0 × rank-N) — covered by value-ops elementwise
// factory's scalarBroadcastBinop path.
// =====================================================================

test('scalar .* vector: rank-0 × rank-N elementwise', () => {
  const r = evalBinding(`
s = 2.0
v = [1.0, 2.0, 3.0]
Y = s .* v
`, 'Y');
  assert.deepEqual(toArray(r), [2, 4, 6]);
});

test('vector ./ scalar: rank-N × rank-0 elementwise', () => {
  const r = evalBinding(`
v = [4.0, 8.0, 12.0]
s = 4.0
Y = v ./ s
`, 'Y');
  assert.deepEqual(toArray(r), [1, 2, 3]);
});

// =====================================================================
// Unary scalar maths — engine primitives for `broadcasted(<unary>)`.
// =====================================================================

test('exp.(A) for fixed length-3 vector', () => {
  const r = evalBinding(`
A = [0.0, 1.0, 2.0]
Y = exp.(A)
`, 'Y');
  const out = toArray(r);
  assert.ok(Math.abs(out[0] - 1) < 1e-12);
  assert.ok(Math.abs(out[1] - Math.E) < 1e-12);
  assert.ok(Math.abs(out[2] - Math.E * Math.E) < 1e-12);
});

test('log.(A) for fixed length-3 vector', () => {
  const r = evalBinding(`
A = [1.0, 2.718281828459045, 7.389056098930650]
Y = log.(A)
`, 'Y');
  const out = toArray(r);
  assert.ok(Math.abs(out[0] - 0) < 1e-12);
  assert.ok(Math.abs(out[1] - 1) < 1e-12);
  assert.ok(Math.abs(out[2] - 2) < 1e-12);
});

test('sqrt.(A) for fixed length-3 vector', () => {
  const r = evalBinding(`
A = [4.0, 9.0, 16.0]
Y = sqrt.(A)
`, 'Y');
  assert.deepEqual(toArray(r), [2, 3, 4]);
});

test('abs.(A) elementwise', () => {
  const r = evalBinding(`
A = [-1.0, 2.0, -3.5]
Y = abs.(A)
`, 'Y');
  assert.deepEqual(toArray(r), [1, 2, 3.5]);
});

// =====================================================================
// Inverse direction: spec-mul (matrix product) for the same shapes
// unchanged. The fast path doesn't intercept this — spec-mul stays
// matrix-semantic.
// =====================================================================

test('regression: spec-mul on rank-2 × rank-2 is STILL matmul', () => {
  // The fast path only intercepts `broadcast(...)` calls. Direct
  // `mul(A, B)` (the spec form `A * B` lowers to) bypasses the fast
  // path entirely.
  const r = evalBinding(`
A = rowstack([[1.0, 2.0], [3.0, 4.0]])
B = rowstack([[5.0, 6.0], [7.0, 8.0]])
Y = A * B
`, 'Y');
  // A*B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
  assert.deepEqual(r.shape, [2, 2]);
  assert.deepEqual(toArray(r), [19, 22, 43, 50]);
});

// =====================================================================
// Spec §04 singleton-axis expansion — now ON the fast path (stride-0
// reads in value-ops; previously fell through to the per-cell cold
// path). Results are pinned identical to the cold path's.
// =====================================================================

test('singleton-axis broadcast: [3] .* [1] expands by repetition (spec §04)', () => {
  const r = evalBinding(`
A = [10.0, 20.0, 30.0]
B = [2.0]
Y = A .* B
`, 'Y');
  // [10*2, 20*2, 30*2]
  const out = toArray(r);
  assert.equal(out.length, 3);
  assert.deepEqual(out, [20, 40, 60]);
});

test('addaxes row-broadcast (NumPy-style): [2,3] .+ [1,3]', () => {
  // Spec §04's own alignment idiom: `addaxes(b, 1, 0)` turns the
  // length-3 vector into a [1,3] operand whose singleton leading axis
  // repeats across A's rows.
  const r = evalBinding(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = [10.0, 20.0, 30.0]
Y = A .+ addaxes(b, 1, 0)
`, 'Y');
  assert.deepEqual(r.shape, [2, 3]);
  assert.deepEqual(toArray(r), [11, 22, 33, 14, 25, 36]);
});

test('addaxes column-broadcast (Julia-style): [2,3] .+ [2,1]', () => {
  const r = evalBinding(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = [100.0, 200.0]
Y = A .+ addaxes(b, 0, 1)
`, 'Y');
  assert.deepEqual(r.shape, [2, 3]);
  assert.deepEqual(toArray(r), [101, 102, 103, 204, 205, 206]);
});

test('singleton expansion on both operands: [2,1] .* [1,3]', () => {
  const r = evalBinding(`
a = [2.0, 3.0]
b = [10.0, 100.0, 1000.0]
Y = addaxes(a, 0, 1) .* addaxes(b, 1, 0)
`, 'Y');
  // Outer product via singleton broadcast: out[i,j] = a[i]*b[j].
  assert.deepEqual(r.shape, [2, 3]);
  assert.deepEqual(toArray(r), [20, 200, 2000, 30, 300, 3000]);
});

test('ifelse.(cond, then, else) with a singleton branch', () => {
  // ifelseElem rides the same combiner: the size-1 `then` operand
  // repeats along the condition's axis.
  const r = evalBinding(`
c = [true, false, true]
t = [7.0]
e = [1.0, 2.0, 3.0]
Y = ifelse.(c, t, e)
`, 'Y');
  assert.deepEqual(toArray(r), [7, 2, 7]);
});

// =====================================================================
// Fast-vs-cold routing pins — the fast path must claim exactly the
// flat-Value singleton/equal-shape cases and NOTHING with nested-
// vector semantics. Pinned by counting `_broadcastApply` (cold) calls
// through the monkeypatch seam (node:test runs each file in its own
// process, so the patch can't leak).
// =====================================================================

const valueLib = require('../value.ts');

function routeOf(fn: () => any): { path: string; result?: any; error?: string } {
  const internal = sampler._internal;
  const orig = internal._broadcastApply;
  let cold = 0;
  internal._broadcastApply = function (...a: any[]) { cold++; return orig.apply(null, a); };
  try {
    const result = fn();
    return { path: cold > 0 ? 'cold' : 'fast', result };
  } catch (e: any) {
    return { path: cold > 0 ? 'cold' : 'fast', error: e.message };
  } finally {
    internal._broadcastApply = orig;
  }
}

test('routing: singleton-expansion broadcasts dispatch FAST', () => {
  const r = routeOf(() => evalBinding(`
A = rowstack([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = [10.0, 20.0, 30.0]
Y = A .+ addaxes(b, 1, 0)
`, 'Y'));
  assert.equal(r.path, 'fast');
  assert.deepEqual(toArray(r.result), [11, 22, 33, 14, 25, 36]);
});

test('routing: flat-scalar JS-array args coerce and dispatch FAST', () => {
  // Args reaching the broadcast as plain JS number arrays (not
  // Values) coerce via asValue on the fast path.
  const ref = (n: string) => ({ kind: 'ref', ns: 'self', name: n });
  const fnOf = { kind: 'call', op: 'functionof', params: ['a', 'b'],
    body: { kind: 'call', op: 'add', args: [ref('a'), ref('b')] } };
  const ir = { kind: 'call', op: 'broadcast',
    args: [fnOf, { kind: 'lit', value: [1, 2, 3] }, { kind: 'lit', value: [10] }] };
  const r = routeOf(() => sampler.evaluateExpr(ir, {}));
  assert.equal(r.path, 'fast');
  assert.deepEqual(toArray(r.result), [11, 12, 13]);
});

test('routing: nested-vector (Ref-wrap) args stay COLD with per-cell semantics', () => {
  // `[C]`-style hold-constant idiom: the JS-array layer is one loop
  // axis; per cell the body sees the inner VECTOR whole. Must not be
  // claimed by the elementwise fast path.
  const C = valueLib.asValue([1, 2, 3]);
  const ref = (n: string) => ({ kind: 'ref', ns: 'self', name: n });
  const fnOf = { kind: 'call', op: 'functionof', params: ['a', 'b'],
    body: { kind: 'call', op: 'add', args: [ref('a'), ref('b')] } };
  const ir = { kind: 'call', op: 'broadcast',
    args: [fnOf, ref('wrapped'), { kind: 'lit', value: [10, 20] }] };
  const r = routeOf(() => sampler.evaluateExpr(ir, { wrapped: [C, C] }));
  assert.equal(r.path, 'cold');
  // Two cells, each add(C, scalar): [[11,12,13],[21,22,23]].
  assert.deepEqual(r.result.shape, [2, 3]);
  assert.deepEqual(toArray(r.result), [11, 12, 13, 21, 22, 23]);
});

test('routing: outerRank-tagged vec-of-vec + matrix REFUSES (spec §04 outer axes)', () => {
  // A vec-of-vec Value (outerRank=1, storage shape [2,3]) has ONE
  // outer axis whose cells are 3-vectors; a [2,3] matrix has two.
  // Pre-widening the fast path compared storage shapes and silently
  // elementwise-added — semantically wrong. The gate now bails to the
  // cold path, whose classifier throws the §04 outer-axis error.
  const vov = valueLib.asValue([[1, 2, 3], [4, 5, 6]]);
  assert.equal(vov.outerRank, 1, 'precondition: asValue tags vec-of-vec');
  const ref = (n: string) => ({ kind: 'ref', ns: 'self', name: n });
  const fnOf = { kind: 'call', op: 'functionof', params: ['a', 'b'],
    body: { kind: 'call', op: 'add', args: [ref('a'), ref('b')] } };
  const mat = { kind: 'call', op: 'rowstack',
    args: [{ kind: 'lit', value: [[1, 1, 1], [1, 1, 1]] }] };
  const ir = { kind: 'call', op: 'broadcast', args: [fnOf, ref('vov'), mat] };
  const r = routeOf(() => sampler.evaluateExpr(ir, { vov }));
  assert.equal(r.path, 'cold');
  assert.match(r.error || '', /same number of axes/);
});

test('routing: incompatible sizes fall back COLD for the spec diagnostic', () => {
  const r = routeOf(() => evalBinding(`
A = [1.0, 2.0, 3.0]
B = [1.0, 2.0]
Y = A .+ B
`, 'Y'));
  assert.equal(r.path, 'cold');
  assert.match(r.error || '', /incompatible sizes|equal or 1/);
});

// =====================================================================
// ifelseElem direct units — the three-arg elementwise impl now rides
// the shared `_broadcastOutShape` combiner (singleton expansion) and
// handles complex BRANCHES (planar buffers picked per cell; a real
// branch contributes im = 0). The condition must be real.
// =====================================================================

const vo = require('../value-ops.ts');

test('ifelseElem: singleton then-branch expands along the condition axis', () => {
  const r = vo.ifelseElem(
    valueLib.asValue([1, 0, 1]),
    valueLib.asValue([7]),
    valueLib.asValue([10, 20, 30]));
  assert.deepEqual(r.shape, [3]);
  assert.deepEqual(Array.from(r.data), [7, 20, 7]);
});

test('ifelseElem: singleton condition selects whole branches', () => {
  const rThen = vo.ifelseElem(
    valueLib.asValue([1]),
    valueLib.asValue([1, 2, 3]),
    valueLib.asValue([9, 9, 9]));
  assert.deepEqual(Array.from(rThen.data), [1, 2, 3]);
  const rElse = vo.ifelseElem(
    valueLib.asValue([0]),
    valueLib.asValue([1, 2, 3]),
    valueLib.asValue([9, 9, 9]));
  assert.deepEqual(Array.from(rElse.data), [9, 9, 9]);
});

test('ifelseElem: rank-2 condition with [2,1] singleton else', () => {
  const cond = { shape: [2, 3], data: new Float64Array([1, 1, 0, 0, 1, 0]) };
  const thn = { shape: [2, 3], data: new Float64Array([1, 2, 3, 4, 5, 6]) };
  const els = { shape: [2, 1], data: new Float64Array([100, 200]) };
  const r = vo.ifelseElem(cond, thn, els);
  assert.deepEqual(r.shape, [2, 3]);
  assert.deepEqual(Array.from(r.data), [1, 2, 100, 200, 5, 200]);
});

test('ifelseElem: complex branches select planar cells; real branch gets im 0', () => {
  const z = vo.complexElem(valueLib.asValue([1, 2]), valueLib.asValue([3, 4]));
  const r = vo.ifelseElem(valueLib.asValue([1, 0]), z, valueLib.asValue([9, 9]));
  assert.equal(r.dtype, 'complex');
  assert.deepEqual(Array.from(r.data), [1, 9]);
  assert.deepEqual(Array.from(r.im), [3, 0]);
});

test('ifelseElem: complex condition refuses', () => {
  const z = vo.complexElem(valueLib.asValue([1, 0]), valueLib.asValue([0, 1]));
  assert.throws(
    () => vo.ifelseElem(z, valueLib.asValue([1, 2]), valueLib.asValue([3, 4])),
    /real-valued/);
});

test('ifelseElem: incompatible non-singleton shapes still refuse', () => {
  assert.throws(
    () => vo.ifelseElem(
      valueLib.asValue([1, 0]),
      valueLib.asValue([1, 2, 3]),
      valueLib.asValue([4, 5])),
    /mismatch/);
});
