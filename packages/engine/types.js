'use strict';

// FlatPIR type representation, helpers, and built-in signature registry.
//
// This module mirrors §sec:flatpir of the FlatPPL design spec.
// It is the single source of truth for "what kind of thing is this
// expression?" — analyzer.js, orchestrator.js, and the visualizer all
// query this module rather than each maintaining their own ad-hoc
// classification (was: isMeasureExpr, MEASURE_PRODUCING, the
// orchestrator's argSignature, etc.).
//
// Scope (this commit, Phase 1)
// ============================
// Implemented categories:
//   %deferred, (%failed reason), %any, (%scalar prim), (%array rank shape elem),
//   (%record fields), (%tuple elems), (%measure domain), and a polymorphic
//   type variable used inside built-in signatures.
//
// Deferred until features land:
//   %kernel, %function, %likelihood, %table, %module — added when the
//   corresponding language constructs become first-class in the engine.
//
// What's NOT here (per spec §sec:flatpir):
//   * Set membership (posreals, unitinterval, etc.). Those live structurally
//     in the expression as `elementof`/`Lebesgue(support=…)` calls and are
//     consulted at runtime, not in the type system.
//   * Refinement types (sign, range). Out of scope by design.
//   * Phase information. Already lives separately in analyzer.computePhases.
//
// Interface
// =========
// The exports below split into three groups:
//   * Type constructors: deferred(), failed(), any(), scalar(), array(),
//     record(), tuple(), measure(), tvar(). Plus the constants REAL,
//     INTEGER, BOOLEAN, COMPLEX, STRING.
//   * Type operations: equal(), substitute(), unify(), show(), isMeasure(),
//     isValue().
//   * Built-in signatures: signatureOf(opName) returns a fresh-instantiated
//     signature object {args, kwargs, result, variadic} that callers use to
//     check argument types and infer the result type at a call site.

// =====================================================================
// Type constructors
//
// Every type carries a `kind` discriminator. Plain objects rather than
// classes — easier to JSON-serialize for FlatPIR roundtripping and
// cheaper to allocate in the inner inference loop.
// =====================================================================

function deferred()     { return { kind: 'deferred' }; }
function failed(reason) { return { kind: 'failed', reason }; }
function any()          { return { kind: 'any' }; }

/** Scalar type. `prim` ∈ {'real','integer','boolean','complex','string'}. */
function scalar(prim)   { return { kind: 'scalar', prim }; }

/** Array type. `rank` is a positive integer literal. `shape` is an array
 *  of length `rank`; each entry is a positive integer or '%dynamic'.
 *  `elem` is the element type (a Type, often scalar). */
function array(rank, shape, elem) { return { kind: 'array', rank, shape, elem }; }

/** Record type. `fields` is a plain object mapping field name → Type. */
function record(fields) { return { kind: 'record', fields }; }

/** Tuple type. `elems` is an array of element Types (length ≥ 2 per spec). */
function tuple(elems)   { return { kind: 'tuple', elems }; }

/** Closed measure over a value domain. */
function measure(domain) { return { kind: 'measure', domain }; }

/** Type variable, used inside polymorphic signatures (e.g. weighted's T).
 *  `id` is a string identifier; instantiation gives every signature a
 *  fresh batch of variables so two call sites can't accidentally share. */
function tvar(id)       { return { kind: 'var', id }; }

/** User-defined function. `inputs` is an array of `{name, type}` for
 *  the function's parameters; `result` is the body's inferred type.
 *
 *  FlatPIR canonical form is `(%function (%inputs <name>...))` —
 *  carrying parameter names only, with the result type recomputed at
 *  each call site by traversing the body. We carry `result` as an
 *  engine-internal extension because for our purposes the body's
 *  result type is fixed at definition time (we don't yet support
 *  per-call-site polymorphism that varies with input types). When/if
 *  we add full polymorphism, this stays type-checked: `result` becomes
 *  the *generic* result of the body, and call-site inference
 *  specialises it. */
function funcType(inputs, result)   { return { kind: 'function', inputs, result }; }

/** User-defined transition kernel. Same shape as a function but the
 *  result type is always a measure (per spec §sec:functionof-measure:
 *  a functionof with a measure body is a kernel). Named to avoid
 *  collision with the `fn` and `kernelof` built-in surface forms. */
function kernelType(inputs, result) { return { kind: 'kernel',   inputs, result }; }

// Convenience constants for the most common scalar types. Use these
// rather than re-allocating scalar('real') everywhere — equality is
// structural so the savings are micro, but it reads better.
const REAL    = scalar('real');
const INTEGER = scalar('integer');
const BOOLEAN = scalar('boolean');
const COMPLEX = scalar('complex');
const STRING  = scalar('string');

// =====================================================================
// Type operations
// =====================================================================

/**
 * Structural type equality. Recursively compares the discriminator and
 * all child types. NB: shape entries '%dynamic' compare by string.
 */
function equal(a, b) {
  if (a === b) return true;
  if (!a || !b) return false;
  if (a.kind !== b.kind) return false;
  switch (a.kind) {
    case 'deferred':
    case 'any':
      return true;
    case 'failed':
      return a.reason === b.reason;
    case 'scalar':
      return a.prim === b.prim;
    case 'array':
      if (a.rank !== b.rank) return false;
      if (a.shape.length !== b.shape.length) return false;
      for (let i = 0; i < a.shape.length; i++) if (a.shape[i] !== b.shape[i]) return false;
      return equal(a.elem, b.elem);
    case 'record': {
      const ka = Object.keys(a.fields), kb = Object.keys(b.fields);
      if (ka.length !== kb.length) return false;
      // Field order matters per spec §sec:valuetypes.
      for (let i = 0; i < ka.length; i++) {
        if (ka[i] !== kb[i]) return false;
        if (!equal(a.fields[ka[i]], b.fields[kb[i]])) return false;
      }
      return true;
    }
    case 'tuple':
      if (a.elems.length !== b.elems.length) return false;
      for (let i = 0; i < a.elems.length; i++) if (!equal(a.elems[i], b.elems[i])) return false;
      return true;
    case 'measure':
      return equal(a.domain, b.domain);
    case 'function':
    case 'kernel': {
      if (a.inputs.length !== b.inputs.length) return false;
      for (let i = 0; i < a.inputs.length; i++) {
        if (a.inputs[i].name !== b.inputs[i].name) return false;
        if (!equal(a.inputs[i].type, b.inputs[i].type)) return false;
      }
      return equal(a.result, b.result);
    }
    case 'var':
      return a.id === b.id;
  }
  return false;
}

/**
 * Substitute type variables in `t` according to `subst` (a Map<id, Type>).
 * Returns a new type tree without the substituted variables, leaving any
 * unresolved variables in place. Used after unification to materialise a
 * call's result type from its signature.
 */
function substitute(t, subst) {
  if (!t) return t;
  if (t.kind === 'var') {
    const v = subst.get(t.id);
    return v ? substitute(v, subst) : t;
  }
  if (t.kind === 'array')   return array(t.rank, t.shape.slice(), substitute(t.elem, subst));
  if (t.kind === 'measure') return measure(substitute(t.domain, subst));
  if (t.kind === 'tuple')   return tuple(t.elems.map(e => substitute(e, subst)));
  if (t.kind === 'record') {
    const out = {};
    for (const k in t.fields) out[k] = substitute(t.fields[k], subst);
    return record(out);
  }
  if (t.kind === 'function' || t.kind === 'kernel') {
    return { kind: t.kind,
      inputs: t.inputs.map(i => ({ name: i.name, type: substitute(i.type, subst) })),
      result: substitute(t.result, subst) };
  }
  return t;
}

/**
 * Unify two types. Returns a new substitution (extending `subst`, which is
 * a Map<id, Type>) or null on failure. Symmetric, with the special case
 * that %deferred and %any unify with anything (deferred = "we'll fill
 * this in later"; any = "no constraint per spec"). %failed never unifies
 * — once inference fails on a subterm, every dependent unification fails
 * too, propagating the error.
 */
function unify(a, b, subst) {
  if (!subst) subst = new Map();
  a = walk(a, subst);
  b = walk(b, subst);
  if (a.kind === 'failed' || b.kind === 'failed') return null;
  if (a.kind === 'deferred' || b.kind === 'deferred') return subst;
  if (a.kind === 'any' || b.kind === 'any') return subst;
  if (a.kind === 'var') return bind(a.id, b, subst);
  if (b.kind === 'var') return bind(b.id, a, subst);
  if (a.kind !== b.kind) return null;
  switch (a.kind) {
    case 'scalar':
      // Strict equality, OR canonical promotion either direction.
      // §sec:valuetypes: "booleans ⊂ integers ⊂ reals" with a canonical
      // embedding into complexes. So `Normal(mu=0, sigma=1)` (integer
      // literals) unifies cleanly against the (real, real) kwarg
      // signature, and `equal(b, 1)` (boolean vs integer) doesn't
      // emit a spurious error. Subtyping is intentionally collapsed
      // into unification here — keeps the inference machinery simple
      // and matches the spec's "may use these embeddings implicitly".
      if (a.prim === b.prim) return subst;
      if (canPromote(a.prim, b.prim) || canPromote(b.prim, a.prim)) return subst;
      return null;
    case 'measure':
      return unify(a.domain, b.domain, subst);
    case 'array': {
      if (a.rank !== b.rank) return null;
      if (a.shape.length !== b.shape.length) return null;
      for (let i = 0; i < a.shape.length; i++) {
        // %dynamic unifies with any concrete dim (and with another %dynamic).
        if (a.shape[i] === '%dynamic' || b.shape[i] === '%dynamic') continue;
        if (a.shape[i] !== b.shape[i]) return null;
      }
      return unify(a.elem, b.elem, subst);
    }
    case 'tuple': {
      if (a.elems.length !== b.elems.length) return null;
      let s = subst;
      for (let i = 0; i < a.elems.length; i++) {
        s = unify(a.elems[i], b.elems[i], s);
        if (s == null) return null;
      }
      return s;
    }
    case 'record': {
      const ka = Object.keys(a.fields), kb = Object.keys(b.fields);
      if (ka.length !== kb.length) return null;
      let s = subst;
      for (let i = 0; i < ka.length; i++) {
        if (ka[i] !== kb[i]) return null;
        s = unify(a.fields[ka[i]], b.fields[kb[i]], s);
        if (s == null) return null;
      }
      return s;
    }
  }
  return null;
}

// Scalar promotion lattice (least → most permissive). Used by unify's
// scalar case to admit canonical embeddings (booleans ⊂ integers ⊂
// reals → complexes) without a separate subtype-check pass.
const SCALAR_RANK = { boolean: 0, integer: 1, real: 2, complex: 3 };
function canPromote(from, to) {
  return SCALAR_RANK[from] != null && SCALAR_RANK[to] != null
    && SCALAR_RANK[from] <= SCALAR_RANK[to];
}

// Walk a type one level — replace it with its substitution if it's a
// variable that's been bound. Stops at the first non-variable.
function walk(t, subst) {
  while (t && t.kind === 'var' && subst.has(t.id)) t = subst.get(t.id);
  return t;
}

// Bind a type variable to a concrete type, with an occurs check to avoid
// constructing infinite types.
function bind(id, t, subst) {
  if (t.kind === 'var' && t.id === id) return subst;
  if (occurs(id, t, subst)) return null;
  const next = new Map(subst);
  next.set(id, t);
  return next;
}

function occurs(id, t, subst) {
  t = walk(t, subst);
  if (t.kind === 'var') return t.id === id;
  if (t.kind === 'measure') return occurs(id, t.domain, subst);
  if (t.kind === 'array') return occurs(id, t.elem, subst);
  if (t.kind === 'tuple') return t.elems.some(e => occurs(id, e, subst));
  if (t.kind === 'record') return Object.values(t.fields).some(f => occurs(id, f, subst));
  return false;
}

/**
 * Render a type in human-readable form for user-facing diagnostics.
 *
 * The output uses plain English rather than the angle-bracket
 * "measure<T>" / "array<1,[3],T>" syntax — FlatPPL surface doesn't
 * have parametric type syntax, and users seeing "measure<'T_3>" tend
 * to read it as a literal foreign string. So:
 *
 *   measure (free domain)        → "measure"
 *   measure over real            → "measure over real"
 *   array<1, [3], real>          → "array of real (length 3)"
 *   array<1, [%dynamic], real>   → "array of real"
 *   array<2, [3, 4], real>       → "2d array of real (shape 3×4)"
 *   record with named fields     → "record with fields a: real, b: integer"
 *   tuple<real, integer>         → "tuple (real, integer)"
 *   free type variable           → "any" (the freshness counter has
 *                                   no meaning to users; an unresolved
 *                                   variable means "we couldn't pin
 *                                   this down" which reads as "any")
 *
 * Round-trip-faithful FlatPIR rendering (canonical S-expressions) is
 * a separate concern and would belong in a dedicated `showSExpr`
 * helper if needed.
 */
function show(t) {
  if (!t) return '<null>';
  switch (t.kind) {
    case 'deferred': return 'deferred';
    case 'failed':   return 'failed (' + t.reason + ')';
    case 'any':      return 'any';
    case 'scalar':   return t.prim;
    case 'array':    return showArray(t);
    case 'record':   return showRecord(t);
    case 'tuple':    return 'tuple (' + t.elems.map(show).join(', ') + ')';
    case 'measure':  return showMeasure(t);
    case 'function': return showCallable('function', t);
    case 'kernel':   return showCallable('kernel',   t);
    case 'var':      return 'any';  // unresolved → user-facing "any"
  }
  return '<unknown>';
}

function showMeasure(t) {
  // Drop the domain when it's an unresolved type variable — "measure"
  // reads better than "measure over any". Keep it whenever the
  // domain is concrete or compound.
  if (t.domain && t.domain.kind === 'var') return 'measure';
  return 'measure over ' + show(t.domain);
}

function showArray(t) {
  const concrete = t.shape.every(d => d !== '%dynamic');
  const elem = show(t.elem);
  if (t.rank === 1) {
    return concrete ? 'array of ' + elem + ' (length ' + t.shape[0] + ')'
                    : 'array of ' + elem;
  }
  return concrete ? t.rank + 'd array of ' + elem + ' (shape ' + t.shape.join('×') + ')'
                  : t.rank + 'd array of ' + elem;
}

function showRecord(t) {
  const ks = Object.keys(t.fields);
  if (ks.length === 0) return 'record';
  return 'record with fields ' + ks.map(k => k + ': ' + show(t.fields[k])).join(', ');
}

function showCallable(label, t) {
  // "function f(par: real) → real" reads naturally next to a user
  // identifier (the function's name surfaces from context). Kernels
  // get the same shape with the result type as a measure.
  const params = t.inputs.map(i => i.name + ': ' + show(i.type)).join(', ');
  return label + '(' + params + ') → ' + show(t.result);
}

/** Whether `t` is a measure type (or one that resolves to a measure). */
function isMeasure(t) {
  return t != null && t.kind === 'measure';
}

/**
 * Whether `t` is a value type — anything that can sit on the right of
 * a `weighted` or be returned by `draw`. Currently: scalars, arrays,
 * records, tuples (whose elements are themselves value-typed). The
 * spec calls these "value types" in §sec:valuetypes.
 *
 * %deferred and %any are accepted because in those cases inference
 * hasn't classified the type yet — we don't want to emit a spurious
 * "not a value" error before the type is known.
 */
function isValue(t) {
  if (t == null) return false;
  switch (t.kind) {
    case 'scalar':
    case 'array':
    case 'record':
    case 'tuple':
    case 'deferred':
    case 'any':
      return true;
  }
  return false;
}

/** Whether `t` is a user-defined callable (function or kernel). */
function isCallable(t) {
  return t != null && (t.kind === 'function' || t.kind === 'kernel');
}

/**
 * Unify two types under "numeric" arithmetic rules. Used by binary
 * arithmetic ops (add, sub, mul, div, comparisons, …) to handle the
 * three legitimate cases:
 *
 *   1. scalar ⊕ scalar           → scalar (with promotion through unify)
 *   2. array<R,S,T> ⊕ array<R,S,T'> → array<R,S, T⊔T'> (elementwise; shape match)
 *   3. scalar T ⊕ array<R,S,T'> → array<R,S, T⊔T'>   (broadcast)
 *      array<R,S,T> ⊕ scalar T'  → array<R,S, T⊔T'>   (broadcast)
 *
 * Returns `{ result, subst }` on success or `null` on incompatible
 * shapes. The caller wraps this in its diagnostic emission. We don't
 * unify "tuple", "record", "measure" etc. — those aren't numeric.
 */
function unifyArith(a, b, subst) {
  if (!subst) subst = new Map();
  a = walk(a, subst);
  b = walk(b, subst);
  // Lenient on inferenceless types: deferred / any / failed flow
  // through, matching unify's behaviour for non-arith ops.
  if (a.kind === 'failed' || b.kind === 'failed') return null;
  if (a.kind === 'deferred' || a.kind === 'any')  return { result: b, subst };
  if (b.kind === 'deferred' || b.kind === 'any')  return { result: a, subst };

  // Both scalars — fall through to the numeric-promotion path of unify.
  if (a.kind === 'scalar' && b.kind === 'scalar') {
    const s2 = unify(a, b, subst);
    if (s2 == null) return null;
    // Pick the "wider" scalar so 1 + 1.0 lands on real.
    const aRank = SCALAR_RANK[a.prim], bRank = SCALAR_RANK[b.prim];
    const result = (aRank != null && bRank != null && aRank >= bRank) ? a : b;
    return { result, subst: s2 };
  }
  // Both arrays — shapes must match (with %dynamic flexibility);
  // element types unify as numerics.
  if (a.kind === 'array' && b.kind === 'array') {
    if (a.rank !== b.rank) return null;
    if (a.shape.length !== b.shape.length) return null;
    const shape = [];
    for (let i = 0; i < a.shape.length; i++) {
      const ai = a.shape[i], bi = b.shape[i];
      if (ai === '%dynamic' || bi === '%dynamic') {
        shape.push(ai === '%dynamic' ? bi : ai);
      } else if (ai !== bi) {
        return null;
      } else {
        shape.push(ai);
      }
    }
    const elemR = unifyArith(a.elem, b.elem, subst);
    if (elemR == null) return null;
    return { result: array(a.rank, shape, elemR.result), subst: elemR.subst };
  }
  // Broadcast: scalar with array → result is the array with element
  // type unified with the scalar.
  if (a.kind === 'scalar' && b.kind === 'array') {
    const elemR = unifyArith(a, b.elem, subst);
    if (elemR == null) return null;
    return { result: array(b.rank, b.shape.slice(), elemR.result), subst: elemR.subst };
  }
  if (a.kind === 'array' && b.kind === 'scalar') {
    const elemR = unifyArith(a.elem, b, subst);
    if (elemR == null) return null;
    return { result: array(a.rank, a.shape.slice(), elemR.result), subst: elemR.subst };
  }
  return null;
}

// =====================================================================
// Built-in signature registry
//
// Each signature describes:
//   args      — array of expected positional-arg Types (may contain
//               type variables); `null` for ops that take only kwargs.
//   kwargs    — plain object {name: Type} for required kwargs (may be
//               empty); names not in the object are flagged as unknown.
//   result    — return Type
//   variadic  — if set, args[args.length-1] (or kwargs key '*') is
//               repeatable: one of 'positional' or 'kwargs'.
//
// Variables ('T') are LOCAL to a signature; calling signatureOf() returns
// a fresh instantiation so each call site has independent variables.
// =====================================================================

// Helper for distribution constructors — every parameterised real-valued
// scalar distribution has the same signature shape (kwargs are values,
// result is a real-valued measure). Reduces repetition below.
function realDistKwargs(kwargs) {
  return { args: null, kwargs, result: measure(REAL) };
}
function intDistKwargs(kwargs) {
  return { args: null, kwargs, result: measure(INTEGER) };
}
function boolDistKwargs(kwargs) {
  return { args: null, kwargs, result: measure(BOOLEAN) };
}

// Signatures are stored as factory functions so each call to
// signatureOf() yields a freshly-allocated signature object with
// freshly-allocated type variables. Sharing a single signature object
// across call sites would let one site's unification leak into another.
const SIGNATURE_FACTORIES = {
  // ---- Distributions -----------------------------------------------
  // Scalar continuous: kwargs all real-typed, result is measure<real>.
  Normal:            () => realDistKwargs({ mu: REAL, sigma: REAL }),
  LogNormal:         () => realDistKwargs({ mu: REAL, sigma: REAL }),
  Cauchy:            () => realDistKwargs({ loc: REAL, scale: REAL }),
  StudentT:          () => realDistKwargs({ nu: REAL }),
  Logistic:          () => realDistKwargs({ loc: REAL, scale: REAL }),
  Exponential:       () => realDistKwargs({ rate: REAL }),
  Gamma:             () => realDistKwargs({ alpha: REAL, theta: REAL }),
  InverseGamma:      () => realDistKwargs({ alpha: REAL, theta: REAL }),
  Weibull:           () => realDistKwargs({ alpha: REAL, theta: REAL }),
  Beta:              () => realDistKwargs({ alpha: REAL, beta: REAL }),
  Uniform:           () => realDistKwargs({ min: REAL, max: REAL }),
  GeneralizedNormal: () => realDistKwargs({ mu: REAL, alpha: REAL, beta: REAL }),
  // Scalar discrete.
  Bernoulli: () => boolDistKwargs({ p: REAL }),
  Binomial:  () => intDistKwargs({ n: INTEGER, p: REAL }),
  Poisson:   () => intDistKwargs({ rate: REAL }),
  // Categorical is over integer atoms (categories indexed 1..K).
  Categorical: () => intDistKwargs({ probs: array(1, ['%dynamic'], REAL) }),
  // Fundamental measures over reals/integers — argument-less in the
  // common form; support sets are advisory and don't affect type.
  Lebesgue:  () => ({ args: [], kwargs: {}, result: measure(REAL) }),
  Counting:  () => ({ args: [], kwargs: {}, result: measure(INTEGER) }),
  Dirac:     () => ({ args: [tvar('T')], kwargs: {}, result: measure(tvar('T')) }),

  // ---- Measure algebra ---------------------------------------------
  weighted:    () => ({ args: [REAL,    measure(tvar('T'))], kwargs: {}, result: measure(tvar('T')) }),
  logweighted: () => ({ args: [REAL,    measure(tvar('T'))], kwargs: {}, result: measure(tvar('T')) }),
  normalize:   () => ({ args: [measure(tvar('T'))], kwargs: {}, result: measure(tvar('T')) }),
  superpose:   () => ({ args: [measure(tvar('T'))], kwargs: {}, result: measure(tvar('T')), variadic: 'positional' }),

  // ---- Stochastic / law extraction ---------------------------------
  draw:    () => ({ args: [measure(tvar('T'))], kwargs: {}, result: tvar('T') }),
  // lawof's argument is value-typed in the spec (the law of a value);
  // identity law `lawof(draw(m)) ≡ m` is checked structurally rather
  // than baked into the signature.
  lawof:   () => ({ args: [tvar('T')], kwargs: {}, result: measure(tvar('T')) }),

  // ---- Inputs / boundaries -----------------------------------------
  // elementof's argument is a set name (a built-in symbol like `reals`).
  // The result type follows the set's structural category. For now we
  // collapse every set to scalar real / integer / boolean — refinement
  // (posreals etc.) is structural-category-real per spec.
  // Special-cased in checkCall (because the arg is a bare set symbol,
  // not a typed expression).
  elementof: () => ({ args: [any()], kwargs: {}, result: any(), special: 'elementof' }),

  // ---- Composition (variadic) --------------------------------------
  // joint takes a kwargs-record of measures and produces a measure of
  // records; iid takes a measure and a count and produces a measure of
  // arrays. Both keep deferred result types when components don't
  // resolve cleanly.
  joint: () => ({ args: [], kwargs: {}, result: deferred(), special: 'joint' }),
  iid:   () => ({ args: [measure(tvar('T')), INTEGER], kwargs: {},
                  result: measure(array(1, ['%dynamic'], tvar('T'))) }),

  // ---- Constructors ------------------------------------------------
  record: () => ({ args: [], kwargs: {}, result: deferred(), special: 'record' }),
  vector: () => ({ args: [tvar('T')], kwargs: {}, result: array(1, ['%dynamic'], tvar('T')),
                   variadic: 'positional' }),
  tuple:  () => ({ args: [], kwargs: {}, result: deferred(), special: 'tuple' }),

  // ---- Arithmetic & predicates -------------------------------------
  // For now we treat arithmetic as 'real-or-integer' polymorphic over
  // a single argument type; promotion happens at runtime, not in the
  // type system. Real ops return real; we'd need a refined "numeric"
  // class to track integer-only chains.
  add:    () => arith2(),
  sub:    () => arith2(),
  mul:    () => arith2(),
  div:    () => ({ args: [REAL, REAL], kwargs: {}, result: REAL }),  // div always real
  divide: () => ({ args: [REAL, REAL], kwargs: {}, result: REAL }),
  mod:    () => arith2(),
  pow:    () => arith2(),
  neg:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  pos:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  abs:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  abs2:   () => ({ args: [REAL], kwargs: {}, result: REAL }),
  exp:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  log:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  log10:  () => ({ args: [REAL], kwargs: {}, result: REAL }),
  sqrt:   () => ({ args: [REAL], kwargs: {}, result: REAL }),
  sin:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  cos:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  tan:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  floor:  () => ({ args: [REAL], kwargs: {}, result: INTEGER }),
  ceil:   () => ({ args: [REAL], kwargs: {}, result: INTEGER }),
  round:  () => ({ args: [REAL], kwargs: {}, result: INTEGER }),
  // Predicates → boolean.
  equal:   () => ({ args: [tvar('T'), tvar('T')], kwargs: {}, result: BOOLEAN }),
  unequal: () => ({ args: [tvar('T'), tvar('T')], kwargs: {}, result: BOOLEAN }),
  lt:      () => ({ args: [REAL, REAL], kwargs: {}, result: BOOLEAN }),
  le:      () => ({ args: [REAL, REAL], kwargs: {}, result: BOOLEAN }),
  gt:      () => ({ args: [REAL, REAL], kwargs: {}, result: BOOLEAN }),
  ge:      () => ({ args: [REAL, REAL], kwargs: {}, result: BOOLEAN }),
  isfinite:() => ({ args: [REAL], kwargs: {}, result: BOOLEAN }),
  isinf:   () => ({ args: [REAL], kwargs: {}, result: BOOLEAN }),
  isnan:   () => ({ args: [REAL], kwargs: {}, result: BOOLEAN }),
  iszero:  () => ({ args: [REAL], kwargs: {}, result: BOOLEAN }),

  // ---- Reductions / norms ------------------------------------------
  sum:    () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  mean:   () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  prod:   () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  length: () => ({ args: [array(1, ['%dynamic'], any())], kwargs: {}, result: INTEGER }),
};

function arith2() { return { args: [REAL, REAL], kwargs: {}, result: REAL }; }

/**
 * Look up a built-in op's signature and return a freshly-instantiated
 * copy. Returns null for unknown names — callers treat unknown-op as
 * "deferred result, no arg checks".
 */
// Module-level counter for fresh type-variable IDs. Persists across
// signatureOf calls so two consecutive sites get distinct variables
// (call A's 'T_0' must not unify with call B's 'T_0').
let FRESH_COUNTER = 0;

function signatureOf(opName) {
  const f = SIGNATURE_FACTORIES[opName];
  if (!f) return null;
  // Re-key the signature's type variables so two call sites can't share
  // unification state. The module-level FRESH_COUNTER guarantees
  // cross-call uniqueness; the per-signature `sub` map keeps the
  // signature's internal sharing intact (so 'T' in args and 'T' in
  // result still bind to the same fresh variable).
  const sig = f();
  const sub = new Map();
  function fresh(t) {
    if (!t) return t;
    if (t.kind === 'var') {
      let m = sub.get(t.id);
      if (!m) {
        m = tvar(t.id + '_' + (FRESH_COUNTER++));
        sub.set(t.id, m);
      }
      return m;
    }
    if (t.kind === 'measure') return measure(fresh(t.domain));
    if (t.kind === 'array')   return array(t.rank, t.shape.slice(), fresh(t.elem));
    if (t.kind === 'tuple')   return tuple(t.elems.map(fresh));
    if (t.kind === 'record') {
      const out = {};
      for (const k in t.fields) out[k] = fresh(t.fields[k]);
      return record(out);
    }
    return t;
  }
  const out = {
    args:     sig.args ? sig.args.map(fresh) : sig.args,
    kwargs:   {},
    result:   fresh(sig.result),
    variadic: sig.variadic || null,
    special:  sig.special  || null,
  };
  for (const k in sig.kwargs) out.kwargs[k] = fresh(sig.kwargs[k]);
  return out;
}

/** Whether this op has a known signature. */
function hasSignature(opName) {
  return Object.prototype.hasOwnProperty.call(SIGNATURE_FACTORIES, opName);
}

module.exports = {
  // Constructors
  deferred, failed, any, scalar, array, record, tuple, measure, tvar,
  funcType, kernelType,
  REAL, INTEGER, BOOLEAN, COMPLEX, STRING,
  // Operations
  equal, substitute, unify, unifyArith, show, isMeasure, isValue, isCallable,
  // Signatures
  signatureOf, hasSignature,
};
