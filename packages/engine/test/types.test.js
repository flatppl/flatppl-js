'use strict';

// Unit tests for engine/types.js — the FlatPIR type representation
// and built-in signature registry.
//
// Coverage:
//   - Type constructors produce the expected shapes
//   - equal / show / isMeasure / isValue
//   - unify on concrete types, type variables, deferred/any
//   - signatureOf yields fresh variables per call site
//   - A representative cross-section of registered built-in signatures

const { test } = require('node:test');
const assert = require('node:assert/strict');

const T = require('../types');

// =====================================================================
// Constructors and constants
// =====================================================================

test('constructors: scalar / measure / array / record / tuple / tvar', () => {
  assert.deepEqual(T.scalar('real'),   { kind: 'scalar', prim: 'real' });
  assert.deepEqual(T.measure(T.REAL),  { kind: 'measure', domain: { kind: 'scalar', prim: 'real' } });
  assert.deepEqual(T.array(1, [3], T.REAL),
    { kind: 'array', rank: 1, shape: [3], elem: { kind: 'scalar', prim: 'real' } });
  assert.deepEqual(T.record({ x: T.REAL }),
    { kind: 'record', fields: { x: { kind: 'scalar', prim: 'real' } } });
  assert.deepEqual(T.tuple([T.REAL, T.INTEGER]),
    { kind: 'tuple', elems: [{ kind: 'scalar', prim: 'real' }, { kind: 'scalar', prim: 'integer' }] });
  assert.deepEqual(T.tvar('T'), { kind: 'var', id: 'T' });
});

test('REAL / INTEGER / BOOLEAN / COMPLEX / STRING are stable singletons', () => {
  assert.equal(T.REAL.prim, 'real');
  assert.equal(T.INTEGER.prim, 'integer');
  assert.equal(T.BOOLEAN.prim, 'boolean');
  assert.equal(T.COMPLEX.prim, 'complex');
  assert.equal(T.STRING.prim, 'string');
});

// =====================================================================
// Equality
// =====================================================================

test('equal: identical primitives', () => {
  assert.ok(T.equal(T.REAL, T.scalar('real')));
  assert.ok(!T.equal(T.REAL, T.INTEGER));
});

test('equal: nested measure types', () => {
  assert.ok(T.equal(T.measure(T.REAL), T.measure(T.scalar('real'))));
  assert.ok(!T.equal(T.measure(T.REAL), T.measure(T.INTEGER)));
});

test('equal: array dim and element', () => {
  assert.ok(T.equal(T.array(1, [3], T.REAL), T.array(1, [3], T.REAL)));
  assert.ok(!T.equal(T.array(1, [3], T.REAL), T.array(1, [4], T.REAL)));
  assert.ok(!T.equal(T.array(1, [3], T.REAL), T.array(2, [3, 3], T.REAL)));
});

test('equal: record fields are order-sensitive (per spec)', () => {
  // §sec:valuetypes: "Field order is part of the record's identity"
  assert.ok(!T.equal(T.record({ a: T.REAL, b: T.INTEGER }),
                     T.record({ b: T.INTEGER, a: T.REAL })));
});

// =====================================================================
// show
// =====================================================================

test('show: produces user-facing diagnostic-friendly strings', () => {
  assert.equal(T.show(T.REAL), 'real');
  // Concrete domains use "measure over X"; unresolved variables
  // collapse to bare "measure" so users don't see implementation noise.
  assert.equal(T.show(T.measure(T.REAL)),  'measure over real');
  assert.equal(T.show(T.measure(T.tvar('T'))), 'measure');
  // Arrays render with shape if every dim is concrete; %dynamic dims
  // suppress the shape annotation.
  assert.equal(T.show(T.array(1, [3], T.REAL)),         'array of real (length 3)');
  assert.equal(T.show(T.array(1, ['%dynamic'], T.REAL)), 'array of real');
  assert.equal(T.show(T.array(2, [3, 4], T.REAL)),      '2d array of real (shape 3×4)');
  assert.equal(T.show(T.record({ x: T.REAL, y: T.INTEGER })),
               'record with fields x: real, y: integer');
  assert.equal(T.show(T.tuple([T.REAL, T.INTEGER])),    'tuple (real, integer)');
  assert.equal(T.show(T.deferred()), 'deferred');
  assert.equal(T.show(T.failed('cycle')), 'failed (cycle)');
  // Free type variables render as "any" — their internal IDs would
  // be confusing to users.
  assert.equal(T.show(T.tvar('T')),    'any');
  assert.equal(T.show(T.tvar('T_42')), 'any');
});

// =====================================================================
// isMeasure / isValue
// =====================================================================

test('isMeasure / isValue: predicate sanity', () => {
  assert.ok(T.isMeasure(T.measure(T.REAL)));
  assert.ok(!T.isMeasure(T.REAL));
  assert.ok(T.isValue(T.REAL));
  assert.ok(T.isValue(T.array(1, [3], T.REAL)));
  assert.ok(!T.isValue(T.measure(T.REAL)));
  // Deferred / any default to "could be value" so we don't emit
  // spurious errors before inference completes.
  assert.ok(T.isValue(T.deferred()));
  assert.ok(T.isValue(T.any()));
});

// =====================================================================
// Unification
// =====================================================================

test('unify: identical concrete types succeed with empty subst', () => {
  const s = T.unify(T.REAL, T.REAL, new Map());
  assert.ok(s);
  assert.equal(s.size, 0);
});

test('unify: directional promotion per spec §03 embedding ladder', () => {
  // Spec §03 canonical embeddings are one-directional:
  //   booleans ⊂ integers ⊂ reals → complexes.
  // The engine convention: unify(expected, actual) — actual can fit
  // where expected is requested iff actual is at most as "wide" as
  // expected. So promoting actual upward succeeds; demoting fails.

  // expected REAL, actual INTEGER → integer fits in real (succeeds).
  assert.ok(T.unify(T.REAL, T.INTEGER, new Map()));
  // expected COMPLEX, actual REAL → real embeds in complex (succeeds).
  assert.ok(T.unify(T.COMPLEX, T.REAL, new Map()));
  // expected INTEGER, actual BOOLEAN → boolean fits in integer (succeeds).
  assert.ok(T.unify(T.INTEGER, T.BOOLEAN, new Map()));

  // Demotions are NOT allowed per spec: a real does not generally fit
  // where an integer is expected, etc.
  assert.equal(T.unify(T.INTEGER, T.REAL,    new Map()), null);
  assert.equal(T.unify(T.REAL,    T.COMPLEX, new Map()), null);
  assert.equal(T.unify(T.BOOLEAN, T.INTEGER, new Map()), null);
  // Complex → integer is a two-step demotion; also forbidden.
  assert.equal(T.unify(T.REAL,    T.COMPLEX, new Map()), null);

  // String is outside the numeric tower — never promotes.
  assert.equal(T.unify(T.STRING, T.REAL, new Map()), null);
  assert.equal(T.unify(T.REAL, T.STRING, new Map()), null);
});

test('unify: variable binds to concrete type', () => {
  const v = T.tvar('T');
  const s = T.unify(v, T.REAL, new Map());
  assert.ok(s);
  assert.ok(T.equal(s.get('T'), T.REAL));
});

test('unify: variables on both sides eventually bind through transitivity', () => {
  const a = T.tvar('A'), b = T.tvar('B');
  let s = T.unify(a, b, new Map());
  assert.ok(s);
  s = T.unify(a, T.REAL, s);
  assert.ok(s);
  // After resolving through the chain, both end up at REAL.
  assert.ok(T.equal(T.substitute(a, s), T.REAL));
  assert.ok(T.equal(T.substitute(b, s), T.REAL));
});

test('unify: occurs check rejects T = measure<T>', () => {
  const v = T.tvar('T');
  assert.equal(T.unify(v, T.measure(v), new Map()), null);
});

test('unify: deferred succeeds against any concrete type without binding', () => {
  // %deferred means "we will fill this in later" — unification doesn't
  // commit it to the concrete side, which is correct since the deferred
  // slot may resolve to something else when its turn comes.
  const s = T.unify(T.deferred(), T.measure(T.REAL), new Map());
  assert.ok(s);
});

test('unify: failed never unifies', () => {
  assert.equal(T.unify(T.failed('upstream'), T.REAL, new Map()), null);
  assert.equal(T.unify(T.REAL, T.failed('upstream'), new Map()), null);
});

test('unify: nested measure with type variable', () => {
  const v = T.tvar('T');
  const s = T.unify(T.measure(v), T.measure(T.REAL), new Map());
  assert.ok(s);
  assert.ok(T.equal(s.get('T'), T.REAL));
});

test('unify: array dims and elements respect their respective rules', () => {
  // %dynamic dims accept anything.
  assert.ok(T.unify(T.array(1, [3], T.REAL),
                    T.array(1, ['%dynamic'], T.REAL),
                    new Map()));
  // Element types follow the same lenient scalar rule used at the top
  // level: an integer-element array unifies with a real-element array
  // (canonical embedding integers ⊂ reals).
  assert.ok(T.unify(T.array(1, [3], T.REAL),
                    T.array(1, [3], T.INTEGER),
                    new Map()));
  // Outside-the-numeric-tower element mismatch still fails.
  assert.equal(T.unify(T.array(1, [3], T.REAL),
                       T.array(1, [3], T.STRING),
                       new Map()), null);
  // Concrete-dim mismatch still fails.
  assert.equal(T.unify(T.array(1, [3], T.REAL),
                       T.array(1, [4], T.REAL),
                       new Map()), null);
});

// =====================================================================
// Substitute
// =====================================================================

test('substitute: applies binding through nested types', () => {
  const v = T.tvar('T');
  const s = new Map([['T', T.REAL]]);
  assert.ok(T.equal(T.substitute(v, s), T.REAL));
  assert.ok(T.equal(T.substitute(T.measure(v), s), T.measure(T.REAL)));
  assert.ok(T.equal(T.substitute(T.array(1, [3], v), s),
                    T.array(1, [3], T.REAL)));
});

test('substitute: leaves unbound variables in place', () => {
  const v = T.tvar('U');
  const s = new Map([['T', T.REAL]]);
  assert.deepEqual(T.substitute(v, s), v);
});

// =====================================================================
// signatureOf
// =====================================================================

test('signatureOf: Normal has the expected kwargs / result shape', () => {
  const sig = T.signatureOf('Normal');
  assert.deepEqual(sig.args, null);
  assert.ok(T.equal(sig.kwargs.mu, T.REAL));
  assert.ok(T.equal(sig.kwargs.sigma, T.REAL));
  assert.ok(T.equal(sig.result, T.measure(T.REAL)));
});

test('signatureOf: weighted is polymorphic — (real, measure<T>) → measure<T>', () => {
  const sig = T.signatureOf('weighted');
  assert.equal(sig.args.length, 2);
  assert.ok(T.equal(sig.args[0], T.REAL));
  assert.equal(sig.args[1].kind, 'measure');
  assert.equal(sig.args[1].domain.kind, 'var');
  // result variable is the SAME variable as args[1].domain (per
  // signature), so unifying args[1] against measure<real> at a call
  // site also concretises the result.
  assert.equal(sig.result.domain.id, sig.args[1].domain.id);
});

test('signatureOf: each call yields fresh variable IDs', () => {
  // Two call sites of weighted must not share unification state.
  const a = T.signatureOf('weighted');
  const b = T.signatureOf('weighted');
  assert.notEqual(a.args[1].domain.id, b.args[1].domain.id);
});

test('signatureOf: superpose marks variadic', () => {
  const sig = T.signatureOf('superpose');
  assert.equal(sig.variadic, 'positional');
});

test('signatureOf: draw extracts the value type from a measure', () => {
  const sig = T.signatureOf('draw');
  // draw: (measure<T>) → T — args[0] and result share their var.
  assert.equal(sig.args[0].domain.id, sig.result.id);
});

test('signatureOf: unknown op returns null', () => {
  assert.equal(T.signatureOf('not_a_real_op_xyz'), null);
});

test('hasSignature: true for built-ins, false for unknowns', () => {
  assert.ok(T.hasSignature('Normal'));
  assert.ok(T.hasSignature('weighted'));
  assert.ok(!T.hasSignature('not_a_real_op_xyz'));
});

// =====================================================================
// Cross-cut: instantiation + unification at a call site
// =====================================================================

test('integration: weighted(real, measure<real>) infers result = measure<real>', () => {
  // Mirrors what the inference pass will do at every call site.
  const sig = T.signatureOf('weighted');
  let s = T.unify(sig.args[0], T.REAL, new Map());
  s = T.unify(sig.args[1], T.measure(T.REAL), s);
  const result = T.substitute(sig.result, s);
  assert.ok(T.equal(result, T.measure(T.REAL)));
});

// =====================================================================
// unifyArith — shape-polymorphic arithmetic (scalar/array/broadcast)
// =====================================================================

test('unifyArith: scalar + scalar (numeric promotion)', () => {
  const r = T.unifyArith(T.REAL, T.INTEGER, new Map());
  assert.ok(r);
  assert.ok(T.equal(r.result, T.REAL));      // wider type wins
});

test('unifyArith: array + array (matching concrete shape)', () => {
  const r = T.unifyArith(T.array(1, [3], T.REAL),
                          T.array(1, [3], T.INTEGER),
                          new Map());
  assert.ok(r);
  assert.equal(r.result.kind, 'array');
  assert.deepEqual(r.result.shape, [3]);
  assert.ok(T.equal(r.result.elem, T.REAL));
});

test('unifyArith: array + array with %dynamic dim resolves to the concrete dim', () => {
  const r = T.unifyArith(T.array(1, [3], T.REAL),
                          T.array(1, ['%dynamic'], T.REAL),
                          new Map());
  assert.ok(r);
  assert.deepEqual(r.result.shape, [3]);
});

test('unifyArith: shape mismatch (different concrete dims)', () => {
  assert.equal(
    T.unifyArith(T.array(1, [3], T.REAL), T.array(1, [4], T.REAL), new Map()),
    null);
});

test('unifyArith: broadcast — scalar with array', () => {
  const r1 = T.unifyArith(T.REAL, T.array(1, [3], T.REAL), new Map());
  assert.ok(r1);
  assert.deepEqual(r1.result.shape, [3]);
  // Symmetric.
  const r2 = T.unifyArith(T.array(1, [3], T.REAL), T.REAL, new Map());
  assert.ok(r2);
  assert.deepEqual(r2.result.shape, [3]);
});

test('unifyArith: measure operands are not numeric', () => {
  assert.equal(T.unifyArith(T.measure(T.REAL), T.REAL, new Map()), null);
});

test('integration: weighted(measure<real>, measure<real>) is a type error', () => {
  // Mirrors the failing case the user reported:
  //   invalid1_dist = weighted(theta2_dist, theta1_dist)
  // arg 0 is a measure where a value is expected → unify fails.
  const sig = T.signatureOf('weighted');
  const s = T.unify(sig.args[0], T.measure(T.REAL), new Map());
  assert.equal(s, null);
});

test('integration: weighted(real, real) is a type error', () => {
  // The other failing case:
  //   invalid2_dist = weighted(theta2, theta1)
  // arg 1 is a value where a measure is expected → unify fails.
  const sig = T.signatureOf('weighted');
  let s = T.unify(sig.args[0], T.REAL, new Map());
  s = T.unify(sig.args[1], T.REAL, s);
  assert.equal(s, null);
});
