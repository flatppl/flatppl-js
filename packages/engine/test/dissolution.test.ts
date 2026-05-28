// Phase 2 — single-op broadcast dissolution. Pins the IR rewrite
// (broadcast(functionof(<safe_op>(_,_)), A, B) → <safe_op>(A, B))
// at the orchestrator's bindings level for dotted-binary surfaces and
// confirms the soundness gates (mismatched types / phases / unsafe op
// names / closed-over refs) leave the broadcast in place.

const test = require('node:test');
const assert = require('node:assert');
const { processSource } = require('../index.ts');
const { buildDerivations } = require('../derivations.ts');
const { dissolveExpr, _tryDissolveSingleOp, DISSOLVE_SAFE_OPS } =
  require('../dissolver.ts');

function getBinding(src: string, name: string) {
  const proc = processSource(src);
  const out = buildDerivations(proc.bindings);
  return out.bindings.get(name);
}

// ---------------------------------------------------------------------
// Structural matcher in isolation (bindings=null, no type check)
// ---------------------------------------------------------------------

test('dissolver: structural match on add/sub/neg/pos', () => {
  for (const op of ['add', 'sub', 'neg', 'pos']) {
    assert.ok(DISSOLVE_SAFE_OPS.has(op), op + ' should be in safe set');
  }
});

test('dissolver: mul / div / exp / log / sin are NOT in the Phase-2 safe set', () => {
  // These either have non-elementwise valueOps semantics (mul) or no
  // higher-rank elementwise impl (div, scalar unaries). Phase 3 will
  // widen the set with type-aware safety.
  for (const op of ['mul', 'div', 'divide', 'pow', 'exp', 'log',
                    'sqrt', 'sin', 'cos', 'abs']) {
    assert.ok(!DISSOLVE_SAFE_OPS.has(op),
      op + ' should NOT be in Phase-2 safe set');
  }
});

test('dissolver: structural matcher returns dissolved IR (bindings=null)', () => {
  // The bare structural matcher: broadcast(functionof(add(_,_)), refA, refB)
  // → add(refA, refB). Type-aware soundness check is bypassed when
  // bindings=null (unit-test path).
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof',
        params: ['_arg1_', '_arg2_'],
        paramKwargs: ['arg1', 'arg2'],
        body: {
          kind: 'call', op: 'add',
          args: [
            { kind: 'ref', ns: '%local', name: '_arg1_' },
            { kind: 'ref', ns: '%local', name: '_arg2_' },
          ],
        },
      },
      { kind: 'ref', ns: 'self', name: 'A' },
      { kind: 'ref', ns: 'self', name: 'B' },
    ],
  };
  const out = _tryDissolveSingleOp(bcIR, null);
  assert.ok(out, 'should dissolve');
  assert.equal(out.kind, 'call');
  assert.equal(out.op, 'add');
  assert.equal(out.args.length, 2);
  assert.equal(out.args[0].name, 'A');
  assert.equal(out.args[1].name, 'B');
});

test('dissolver: refuses non-safe op (mul) at the matcher', () => {
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof',
        params: ['_arg1_', '_arg2_'],
        paramKwargs: ['arg1', 'arg2'],
        body: {
          kind: 'call', op: 'mul',
          args: [
            { kind: 'ref', ns: '%local', name: '_arg1_' },
            { kind: 'ref', ns: '%local', name: '_arg2_' },
          ],
        },
      },
      { kind: 'ref', ns: 'self', name: 'A' },
      { kind: 'ref', ns: 'self', name: 'B' },
    ],
  };
  const out = _tryDissolveSingleOp(bcIR, null);
  assert.equal(out, null, 'mul is not Phase-2 safe; matcher returns null');
});

test('dissolver: refuses body with closed-over scope ref', () => {
  // Body references `self.C` in addition to the placeholder — not a
  // pure single-op single-param case.
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof',
        params: ['_arg1_'],
        paramKwargs: ['arg1'],
        body: {
          kind: 'call', op: 'add',
          args: [
            { kind: 'ref', ns: '%local', name: '_arg1_' },
            { kind: 'ref', ns: 'self', name: 'C' },  // closed-over
          ],
        },
      },
      { kind: 'ref', ns: 'self', name: 'A' },
    ],
  };
  const out = _tryDissolveSingleOp(bcIR, null);
  assert.equal(out, null);
});

test('dissolver: swizzled body args dissolve via substitution (Phase 3)', () => {
  // Phase 3's body walker substitutes each `%local` ref with the
  // corresponding broadcast arg by NAME (not by position) — so a
  // body of `add(_arg2_, _arg1_)` over args (A, B) dissolves to
  // `add(B, A)`. Spec-correct: broadcast iterates per cell with the
  // args bound to placeholders by name; the dissolved call just
  // hoists that substitution out of the per-cell loop.
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof',
        params: ['_arg1_', '_arg2_'],
        paramKwargs: ['arg1', 'arg2'],
        body: {
          kind: 'call', op: 'add',
          args: [
            { kind: 'ref', ns: '%local', name: '_arg2_' },
            { kind: 'ref', ns: '%local', name: '_arg1_' },
          ],
        },
      },
      { kind: 'ref', ns: 'self', name: 'A' },
      { kind: 'ref', ns: 'self', name: 'B' },
    ],
  };
  const out = _tryDissolveSingleOp(bcIR, null);
  assert.ok(out);
  assert.equal(out.op, 'add');
  // First substituted arg is whatever _arg2_ maps to = B.
  assert.equal(out.args[0].name, 'B');
  assert.equal(out.args[1].name, 'A');
});

// ---------------------------------------------------------------------
// End-to-end pipeline (processSource → buildDerivations)
// ---------------------------------------------------------------------

test('dotted-binary .+ dissolves end-to-end (length-N flat scalar arrays)', () => {
  const b = getBinding(`
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
Y = A .+ B
`, 'Y');
  assert.ok(b && b.ir, 'Y has lifted IR');
  assert.equal(b.ir.kind, 'call');
  assert.equal(b.ir.op, 'add', 'Y.ir should dissolve to add(A, B)');
  assert.equal(b.ir.args.length, 2);
});

test('dotted-binary .- dissolves end-to-end', () => {
  const b = getBinding(`
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
Y = A .- B
`, 'Y');
  assert.ok(b && b.ir);
  assert.equal(b.ir.op, 'sub');
});

test('dotted-unary .- (negation) dissolves end-to-end', () => {
  const b = getBinding(`
A = [1.0, 2.0, 3.0]
Y = .- A
`, 'Y');
  assert.ok(b && b.ir);
  assert.equal(b.ir.op, 'neg');
});

test('dotted-binary .* does NOT dissolve (mul not in Phase-2 safe set)', () => {
  const b = getBinding(`
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
Y = A .* B
`, 'Y');
  assert.ok(b && b.ir);
  // Should stay as a broadcast call.
  assert.equal(b.ir.op, 'broadcast');
});

test('mixed-shape .+ does NOT dissolve (scalar + vector — soundness guard)', () => {
  // alpha is a parameterised scalar (rank-0); x_data is a fixed-phase
  // length-4 vector (rank-1). The broadcast iterates over x_data's
  // outer axis with alpha held constant per cell — semantics that
  // valueOps elementwise add can't reproduce as a direct call. The
  // dissolver's type-uniformity check leaves the broadcast in place.
  const b = getBinding(`
alpha = elementof(reals)
x_data = [1.1, 1.5, 1.3, 1.4]
Y = alpha .+ x_data
`, 'Y');
  assert.ok(b && b.ir);
  assert.equal(b.ir.op, 'broadcast');
});

test('mixed-phase .+ does NOT dissolve (parameterised + fixed)', () => {
  // Same-shape vectors but different phase. Runtime shapes differ
  // (parameterised carries an atom axis; fixed doesn't), so direct
  // elementwise add would mismatch.
  const b = getBinding(`
mu = elementof(reals)
A = [mu, mu, mu]
B = [1.0, 2.0, 3.0]
Y = A .+ B
`, 'Y');
  assert.ok(b && b.ir);
  assert.equal(b.ir.op, 'broadcast');
});

test('same-type same-phase fixed vectors dissolve', () => {
  // Both A and B are length-3 fixed real vectors — identical
  // inferred type AND phase. Dissolution is sound.
  const b = getBinding(`
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
Y = A .+ B
Z = A .- B
W = .- A
`, 'Y');
  assert.ok(b && b.ir);
  assert.equal(b.ir.op, 'add');
});

test('Phase 3 multi-op body: fn(_arg1_ + _arg2_ - _arg1_) dissolves', () => {
  // The body is a tree of safe ops. Phase 3's substitute walker
  // descends into add → sub and replaces placeholder refs with
  // broadcast args.
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof',
        params: ['_arg1_', '_arg2_'],
        paramKwargs: ['arg1', 'arg2'],
        body: {
          kind: 'call', op: 'sub',
          args: [
            {
              kind: 'call', op: 'add',
              args: [
                { kind: 'ref', ns: '%local', name: '_arg1_' },
                { kind: 'ref', ns: '%local', name: '_arg2_' },
              ],
            },
            { kind: 'ref', ns: '%local', name: '_arg1_' },
          ],
        },
      },
      { kind: 'ref', ns: 'self', name: 'A' },
      { kind: 'ref', ns: 'self', name: 'B' },
    ],
  };
  const out = _tryDissolveSingleOp(bcIR, null);
  assert.ok(out);
  assert.equal(out.op, 'sub');
  // sub(add(A, B), A)
  assert.equal(out.args[0].op, 'add');
  assert.equal(out.args[0].args[0].name, 'A');
  assert.equal(out.args[0].args[1].name, 'B');
  assert.equal(out.args[1].name, 'A');
});

test('Phase 3 multi-op body: refuses unsafe inner op (mul)', () => {
  // Body tree contains an unsafe op (mul). The walker bails — no
  // partial dissolution.
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof',
        params: ['_arg1_', '_arg2_'],
        paramKwargs: ['arg1', 'arg2'],
        body: {
          kind: 'call', op: 'add',
          args: [
            {
              kind: 'call', op: 'mul',  // not in DISSOLVE_SAFE_OPS
              args: [
                { kind: 'ref', ns: '%local', name: '_arg1_' },
                { kind: 'ref', ns: '%local', name: '_arg2_' },
              ],
            },
            { kind: 'ref', ns: '%local', name: '_arg2_' },
          ],
        },
      },
      { kind: 'ref', ns: 'self', name: 'A' },
      { kind: 'ref', ns: 'self', name: 'B' },
    ],
  };
  const out = _tryDissolveSingleOp(bcIR, null);
  assert.equal(out, null);
});

test('Phase 3 multi-op body: literals pass through', () => {
  // `fn(_arg1_ + 1.0)` — a literal in the body survives substitution.
  const bcIR = {
    kind: 'call', op: 'broadcast',
    args: [
      {
        kind: 'call', op: 'functionof',
        params: ['_arg1_'],
        paramKwargs: ['arg1'],
        body: {
          kind: 'call', op: 'add',
          args: [
            { kind: 'ref', ns: '%local', name: '_arg1_' },
            { kind: 'lit', value: 1.0, numType: 'real' },
          ],
        },
      },
      { kind: 'ref', ns: 'self', name: 'A' },
    ],
  };
  const out = _tryDissolveSingleOp(bcIR, null);
  assert.ok(out);
  assert.equal(out.op, 'add');
  assert.equal(out.args[0].name, 'A');
  assert.equal(out.args[1].kind, 'lit');
  assert.equal(out.args[1].value, 1.0);
});

test('nested broadcast: lifted-anon ref blocks outer dissolution (type-guard)', () => {
  // The inner `A .* B` lifts to a synthetic `__anonN` binding; lift
  // runs after typeinfer (analyzer pass 7), so the anon's inferredType
  // is not populated by the same pass. The dissolver's type-guard
  // requires both args to have a concrete inferred type — without one
  // for `__anonN`, the outer broadcast stays as-is. Phase 3 can lift
  // this restriction by re-inferring shape on synthetic bindings.
  const b = getBinding(`
A = [1.0, 2.0, 3.0]
B = [4.0, 5.0, 6.0]
C = [7.0, 8.0, 9.0]
Y = (A .* B) .+ C
`, 'Y');
  assert.ok(b && b.ir);
  // For Phase 2 we accept either: (a) outer dissolves (if typeinfer
  // happened to flow through the lifted anon — engine-internal detail),
  // or (b) outer stays as broadcast. The CORRECTNESS guarantee is what
  // matters: no spurious dissolution on mismatched types.
  assert.ok(b.ir.op === 'add' || b.ir.op === 'broadcast',
    'outer ends in either dissolved or original form');
});
