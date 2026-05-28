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
//   - Shape mismatch + singleton expansion correctly fall through to
//     the cold `_broadcastApply` path (the existing tests there are
//     unaffected; here we just confirm singleton-expansion still
//     works).

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
// Cold-path fallthrough — singleton-axis expansion is NOT handled by
// the fast path; falls through to `_broadcastApply` which does the
// spec §04 size-1 expansion.
// =====================================================================

test('singleton-axis broadcast falls through to cold path (spec §04)', () => {
  // [3] .* [1] — spec broadcast expands the size-1 axis by repetition.
  // The fast path detects shape mismatch and returns null; cold path
  // produces [3].
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
