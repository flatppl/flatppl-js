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
// Scope
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
function failed(reason: any) { return { kind: 'failed', reason }; }
function any()          { return { kind: 'any' }; }

/** Scalar type. `prim` ∈ {'real','integer','boolean','complex','string'}. */
function scalar(prim: any)   { return { kind: 'scalar', prim }; }

/** Array type. `rank` is a positive integer literal. `shape` is an array
 *  of length `rank`; each entry is a positive integer or '%dynamic'.
 *  `elem` is the element type (a Type, often scalar). */
function array(rank: any, shape: any, elem: any) { return { kind: 'array', rank, shape, elem }; }

/** Transposed-vector type (spec §03; FlatPIR `%tvector`).
 *  Distinct from `array(1, [length], elem)`: the spec formalises
 *  transposed vectors as a separate value type from non-transposed
 *  ones, distinguished by orientation. Used so that the return type of
 *  `*` is inferable at the type level:
 *    `transposed_vector(n) * vector(n) → scalar`         (inner product)
 *    `vector(m) * transposed_vector(n) → matrix(m, n)`   (outer product)
 *    `transposed_vector(n) * matrix(n, p) → transposed_vector(p)`
 *    `transpose(vector(n)) → transposed_vector(n)`       (and inverse)
 *
 *  At runtime the orientation is captured by the Klein-4 tag on Value
 *  (engine/value.js); type-level we use this distinct constructor. */
function tvector(length: any, elem: any) { return { kind: 'tvector', length, elem }; }

/** Record type. `fields` is a plain object mapping field name → Type. */
function record(fields: any) { return { kind: 'record', fields }; }

/** Tuple type. `elems` is an array of element Types (length ≥ 2 per spec). */
function tuple(elems: any)   { return { kind: 'tuple', elems }; }

/** Closed measure over a value domain. */
function measure(domain: any) { return { kind: 'measure', domain }; }

/** Type variable, used inside polymorphic signatures (e.g. weighted's T).
 *  `id` is a string identifier; instantiation gives every signature a
 *  fresh batch of variables so two call sites can't accidentally share. */
function tvar(id: any)       { return { kind: 'var', id }; }

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
function funcType(inputs: any, result: any)   { return { kind: 'function', inputs, result }; }

/** User-defined transition kernel. Same shape as a function but the
 *  result type is always a measure (per spec §sec:functionof-measure:
 *  a functionof with a measure body is a kernel). Named to avoid
 *  collision with the `fn` and `kernelof` built-in surface forms. */
function kernelType(inputs: any, result: any) { return { kind: 'kernel',   inputs, result }; }

/**
 * Opaque RNG-state type (per spec §sec:random / §03 `rngstates`):
 *  rng = rnginit(seed)             — produces an rngstate
 *  value, rng' = rand(rng, m)      — threads rngstate through draws
 * Equality is structural-by-kind only (no internal shape exposed),
 * matching the spec's "algorithm-dependent opaque values" guarantee.
 * Engines back this with their own RNG state representation; the JS
 * engine uses the Philox state from rng.js.
 */
function rngstate() { return { kind: 'rngstate' }; }

// Convenience constants for the most common scalar types. Use these
// rather than re-allocating scalar('real') everywhere — equality is
// structural so the savings are micro, but it reads better.
const REAL    = scalar('real');
const INTEGER = scalar('integer');
const BOOLEAN = scalar('boolean');
const COMPLEX = scalar('complex');
const STRING  = scalar('string');
const RNGSTATE = rngstate();

// =====================================================================
// Type operations
// =====================================================================

/**
 * Structural type equality. Recursively compares the discriminator and
 * all child types. NB: shape entries '%dynamic' compare by string.
 */
function equal(a: any, b: any): boolean {
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
    case 'tvector':
      if (a.length !== b.length) return false;
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
    case 'rngstate':
      // Opaque — kind match alone is sufficient.
      return true;
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
function substitute(t: any, subst: Map<any, any>): any {
  if (!t) return t;
  if (t.kind === 'var') {
    const v = subst.get(t.id);
    return v ? substitute(v, subst) : t;
  }
  if (t.kind === 'array')   return array(t.rank, t.shape.slice(), substitute(t.elem, subst));
  if (t.kind === 'tvector') return tvector(t.length, substitute(t.elem, subst));
  if (t.kind === 'measure') return measure(substitute(t.domain, subst));
  if (t.kind === 'tuple')   return tuple(t.elems.map((e: any) => substitute(e, subst)));
  if (t.kind === 'record') {
    const out: Record<string, any> = {};
    for (const k in t.fields) out[k] = substitute(t.fields[k], subst);
    return record(out);
  }
  if (t.kind === 'function' || t.kind === 'kernel') {
    return { kind: t.kind,
      inputs: t.inputs.map((i: any) => ({ name: i.name, type: substitute(i.type, subst) })),
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
function unify(a: any, b: any, subst: any): any {
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
      // Spec §03 promotion ladder: booleans ⊂ integers ⊂ reals, with
      // reals embedded into complexes. The embeddings are ONE-
      // DIRECTIONAL — a boolean can be used where an integer is
      // expected, but not the other way; a real can be used where a
      // complex is expected, but a complex cannot be used where a
      // real is expected.
      //
      // The unify convention in the engine: `a` is the SIGNATURE-side
      // (expected) type, `b` is the actual/inferred type. The check
      // is `canPromote(actual, expected)` — does the actual fit where
      // the signature wants?
      //
      // Symmetric numeric joins (e.g. `int + real → real`) belong to
      // `unifyArith`, which calls unify in explicit wider/narrower
      // order so the directional check applies correctly.
      if (a.prim === b.prim) return subst;
      if (canPromote(b.prim, a.prim)) return subst;
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
    case 'tvector': {
      // Same %dynamic-length convention as array.
      if (a.length !== '%dynamic' && b.length !== '%dynamic'
          && a.length !== b.length) return null;
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
    case 'rngstate':
      // Opaque — both sides are rngstate, no further structure to
      // unify. Same kind already matches above; this case is here for
      // clarity / future extension.
      return subst;
  }
  return null;
}

// Scalar promotion lattice (least → most permissive). Used by unify's
// scalar case to admit canonical embeddings (booleans ⊂ integers ⊂
// reals → complexes) without a separate subtype-check pass.
const SCALAR_RANK: Record<string, number> = { boolean: 0, integer: 1, real: 2, complex: 3 };
function canPromote(from: any, to: any) {
  return SCALAR_RANK[from] != null && SCALAR_RANK[to] != null
    && SCALAR_RANK[from] <= SCALAR_RANK[to];
}

// Walk a type one level — replace it with its substitution if it's a
// variable that's been bound. Stops at the first non-variable.
function walk(t: any, subst: any) {
  while (t && t.kind === 'var' && subst.has(t.id)) t = subst.get(t.id);
  return t;
}

// Bind a type variable to a concrete type, with an occurs check to avoid
// constructing infinite types.
function bind(id: any, t: any, subst: any) {
  if (t.kind === 'var' && t.id === id) return subst;
  if (occurs(id, t, subst)) return null;
  const next = new Map(subst);
  next.set(id, t);
  return next;
}

function occurs(id: any, t: any, subst: any): boolean {
  t = walk(t, subst);
  if (t.kind === 'var') return t.id === id;
  if (t.kind === 'measure') return occurs(id, t.domain, subst);
  if (t.kind === 'array') return occurs(id, t.elem, subst);
  if (t.kind === 'tuple') return t.elems.some((e: any) => occurs(id, e, subst));
  if (t.kind === 'record') return Object.values(t.fields).some((f: any) => occurs(id, f, subst));
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
function show(t: any): string {
  if (!t) return '<null>';
  switch (t.kind) {
    case 'deferred': return 'deferred';
    case 'failed':   return 'failed (' + t.reason + ')';
    case 'any':      return 'any';
    case 'scalar':   return t.prim;
    case 'array':    return showArray(t);
    case 'tvector':  return 'transposed vector of ' + show(t.elem)
                            + (t.length !== '%dynamic' ? ' (length ' + t.length + ')' : '');
    case 'record':   return showRecord(t);
    case 'tuple':    return 'tuple (' + t.elems.map(show).join(', ') + ')';
    case 'measure':  return showMeasure(t);
    case 'rngstate': return 'rngstate';
    case 'function': return showCallable('function', t);
    case 'kernel':   return showCallable('kernel',   t);
    case 'var':      return 'any';  // unresolved → user-facing "any"
  }
  return '<unknown>';
}

function showMeasure(t: any): string {
  // Drop the domain when it's an unresolved type variable — "measure"
  // reads better than "measure over any". Keep it whenever the
  // domain is concrete or compound.
  if (t.domain && t.domain.kind === 'var') return 'measure';
  return 'measure over ' + show(t.domain);
}

function showArray(t: any): string {
  const concrete = t.shape.every((d: any) => d !== '%dynamic');
  const elem: any = show(t.elem);
  if (t.rank === 1) {
    return concrete ? 'array of ' + elem + ' (length ' + t.shape[0] + ')'
                    : 'array of ' + elem;
  }
  return concrete ? t.rank + 'd array of ' + elem + ' (shape ' + t.shape.join('×') + ')'
                  : t.rank + 'd array of ' + elem;
}

function showRecord(t: any): string {
  const ks = Object.keys(t.fields);
  if (ks.length === 0) return 'record';
  return 'record with fields ' + ks.map((k: string) => k + ': ' + show(t.fields[k])).join(', ');
}

function showCallable(label: string, t: any): string {
  // "function f(par: real) → real" reads naturally next to a user
  // identifier (the function's name surfaces from context). Kernels
  // get the same shape with the result type as a measure.
  const params = t.inputs.map((i: any) => i.name + ': ' + show(i.type)).join(', ');
  return label + '(' + params + ') → ' + show(t.result);
}

/** Whether `t` is a measure type (or one that resolves to a measure). */
function isMeasure(t: any) {
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
function isValue(t: any) {
  if (t == null) return false;
  switch (t.kind) {
    case 'scalar':
    case 'array':
    case 'tvector':
    case 'record':
    case 'tuple':
    case 'deferred':
    case 'any':
      return true;
  }
  return false;
}

/** Whether `t` is a user-defined callable (function or kernel). */
function isCallable(t: any) {
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
function unifyArith(a: any, b: any, subst: any): any {
  if (!subst) subst = new Map();
  a = walk(a, subst);
  b = walk(b, subst);
  // Lenient on inferenceless types: deferred / any / failed flow
  // through, matching unify's behaviour for non-arith ops.
  if (a.kind === 'failed' || b.kind === 'failed') return null;
  if (a.kind === 'deferred' || a.kind === 'any')  return { result: b, subst };
  if (b.kind === 'deferred' || b.kind === 'any')  return { result: a, subst };

  // Both scalars — numeric promotion to the wider type (join in the
  // canonical-embedding lattice).
  if (a.kind === 'scalar' && b.kind === 'scalar') {
    const aRank = SCALAR_RANK[a.prim], bRank = SCALAR_RANK[b.prim];
    // Pick the wider as the expected, narrower as actual — that's
    // the direction in which unify's directional canPromote check
    // succeeds. The join becomes the result type.
    let wider: any, narrower: any;
    if (aRank != null && bRank != null && aRank >= bRank) { wider = a; narrower = b; }
    else if (aRank != null && bRank != null)              { wider = b; narrower = a; }
    else if (aRank != null) { wider = a; narrower = b; }
    else                    { wider = b; narrower = a; }
    const s2 = unify(wider, narrower, subst);
    if (s2 == null) return null;
    return { result: wider, subst: s2 };
  }
  // Both arrays — shapes must match (with %dynamic flexibility);
  // element types unify as numerics.
  if (a.kind === 'array' && b.kind === 'array') {
    if (a.rank !== b.rank) return null;
    if (a.shape.length !== b.shape.length) return null;
    const shape: any[] = [];
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
    const elemR: any = unifyArith(a.elem, b.elem, subst);
    if (elemR == null) return null;
    return { result: array(a.rank, shape, elemR.result), subst: elemR.subst };
  }
  // Broadcast: scalar with array → result is the array with element
  // type unified with the scalar.
  if (a.kind === 'scalar' && b.kind === 'array') {
    const elemR: any = unifyArith(a, b.elem, subst);
    if (elemR == null) return null;
    return { result: array(b.rank, b.shape.slice(), elemR.result), subst: elemR.subst };
  }
  if (a.kind === 'array' && b.kind === 'scalar') {
    const elemR: any = unifyArith(a.elem, b, subst);
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
function realDistKwargs(kwargs: any) {
  return { args: null, kwargs, result: measure(REAL) };
}
function intDistKwargs(kwargs: any) {
  return { args: null, kwargs, result: measure(INTEGER) };
}
function boolDistKwargs(kwargs: any) {
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
  // Spec-canonical kwarg names per §08-distributions.md. The runtime
  // `sampler.REGISTRY` may use stdlib's internal names; its `aliases` map
  // bridges spec names back to stdlib names. Adding a distribution: take
  // the spec kwarg names verbatim here, and translate to stdlib in
  // sampler.js's REGISTRY entry.
  Cauchy:            () => realDistKwargs({ location: REAL, scale: REAL }),
  StudentT:          () => realDistKwargs({ nu: REAL }),
  Logistic:          () => realDistKwargs({ mu: REAL, s: REAL }),
  Exponential:       () => realDistKwargs({ rate: REAL }),
  Gamma:             () => realDistKwargs({ shape: REAL, rate: REAL }),
  InverseGamma:      () => realDistKwargs({ shape: REAL, scale: REAL }),
  Weibull:           () => realDistKwargs({ shape: REAL, scale: REAL }),
  Beta:              () => realDistKwargs({ alpha: REAL, beta: REAL }),
  Uniform:           () => realDistKwargs({ support: any() }),
  GeneralizedNormal: () => realDistKwargs({ mean: REAL, alpha: REAL, beta: REAL }),
  // Scalar discrete.
  Bernoulli: () => boolDistKwargs({ p: REAL }),
  Binomial:  () => intDistKwargs({ n: INTEGER, p: REAL }),
  Poisson:   () => intDistKwargs({ rate: REAL }),
  // Categorical is over integer atoms (categories indexed 1..K).
  Categorical:  () => intDistKwargs({ p: array(1, ['%dynamic'], REAL) }),
  Categorical0: () => intDistKwargs({ p: array(1, ['%dynamic'], REAL) }),
  // Fundamental reference measures (spec §06). The optional `support = S`
  // kwarg parameterises on the set the measure lives on; default is `reals`
  // for Lebesgue and `integers` for Counting. The result's measure-domain
  // tracks the support's value-type — `Lebesgue(support = cartpow(reals,
  // n))` is `measure(array(real, n))`. Resolved by the `lebesgue` /
  // `counting` special handlers in typeinfer.js.
  Lebesgue:  () => ({ args: null, kwargs: { support: any() }, result: measure(REAL),    special: 'lebesgue' }),
  Counting:  () => ({ args: null, kwargs: { support: any() }, result: measure(INTEGER), special: 'counting' }),
  // Dirac accepts either `Dirac(x)` or `Dirac(value = x)` per spec §04
  // (built-ins admit both positional and keyword forms with identical
  // semantics). Same shape as the other distributions: `args: null`
  // skips the positional arity check (kwargs path is the typed contract).
  Dirac:     () => ({ args: null, kwargs: { value: tvar('T') }, result: measure(tvar('T')) }),

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

  // tuple_get(tuple_expr, slot_lit) — engine-internal projection emitted
  // by the analyzer's multi-LHS rewriter. Result type depends on the
  // literal slot value, so the static result is `any()` here and the
  // real work is in typeinfer.inferTupleGet (which reads the slot
  // literal and projects the tuple's element type).
  tuple_get: () => ({ args: [any(), INTEGER], kwargs: {}, result: any(), special: 'tuple_get' }),

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
  // cat(...) — structural concatenation (spec §07). All-scalar args
  // produce a vector, all-vector args produce a concatenated vector,
  // all-record args produce a merged record. The static type system
  // can't easily express the all-three-shapes-with-shape-class-uniformity
  // discipline, so we mark cat as `special` and defer the result type
  // to the typeinfer pass for now. The runtime check in
  // sampler.ARITH_OPS catches kind-mix violations.
  cat:    () => ({ args: [any()], kwargs: {}, result: deferred(),
                   variadic: 'positional', special: 'cat' }),
  tuple:  () => ({ args: [], kwargs: {}, result: deferred(), special: 'tuple' }),
  // fixed(x) — identity-typed marker (spec §03 value types). Carries
  // no runtime semantics beyond passing the wrapped value through; the
  // 'fixed' op survives in the IR so tooling can recognize it as a
  // "held constant" hint (e.g. preset-point UI excludes fixed kwargs
  // from the x-axis sweep selector). Single distinguished input;
  // result type equals the input's type.
  fixed:  () => ({ args: [tvar('T')], kwargs: {}, result: tvar('T') }),

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
  floor:  () => ({ args: [REAL], kwargs: {}, result: INTEGER }),
  ceil:   () => ({ args: [REAL], kwargs: {}, result: INTEGER }),
  round:  () => ({ args: [REAL], kwargs: {}, result: INTEGER }),
  // Binary min/max — variadic `maximum`/`minimum` over arrays live
  // elsewhere; these are the binary scalar form.
  min:    () => ({ args: [REAL, REAL], kwargs: {}, result: REAL }),
  max:    () => ({ args: [REAL, REAL], kwargs: {}, result: REAL }),
  // Gamma function family and logit / probit link functions.
  gamma:     () => ({ args: [REAL], kwargs: {}, result: REAL }),
  loggamma:  () => ({ args: [REAL], kwargs: {}, result: REAL }),
  // Complex arithmetic primitives (spec §03 / §07).
  //   complex(re, im) :: (real, real) → complex
  //   real(z)         :: complex → real    (also identity on real)
  //   imag(z)         :: complex → real    (zero on real)
  //   conj(z)         :: complex → complex (identity on real)
  //   cis(theta)      :: real → complex    (e^{iθ})
  // Real ↔ complex unify via the canonical embedding (spec §03's
  // booleans ⊂ integers ⊂ reals embedded in complexes) so call sites
  // mixing scalar primitives don't emit spurious type errors.
  complex: () => ({ args: [REAL, REAL], kwargs: {}, result: COMPLEX }),
  real:    () => ({ args: [COMPLEX],    kwargs: {}, result: REAL }),
  imag:    () => ({ args: [COMPLEX],    kwargs: {}, result: REAL }),
  conj:    () => ({ args: [COMPLEX],    kwargs: {}, result: COMPLEX }),
  cis:     () => ({ args: [REAL],       kwargs: {}, result: COMPLEX }),
  logit:     () => ({ args: [REAL], kwargs: {}, result: REAL }),
  invlogit:  () => ({ args: [REAL], kwargs: {}, result: REAL }),
  probit:    () => ({ args: [REAL], kwargs: {}, result: REAL }),
  invprobit: () => ({ args: [REAL], kwargs: {}, result: REAL }),
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

  // ---- Approximation functions (spec §07) ---------------------------
  // Pure value functions. polynomial / bernstein take a real-array of
  // coefficients and a scalar evaluation point. stepwise takes an
  // edge vector (n+1 reals), a value vector (n reals), and an
  // evaluation point. All three return REAL.
  polynomial: () => ({
    args:   [array(1, ['%dynamic'], REAL), REAL],
    kwargs: { coefficients: array(1, ['%dynamic'], REAL), x: REAL },
    result: REAL,
  }),
  bernstein:  () => ({
    args:   [array(1, ['%dynamic'], REAL), REAL],
    kwargs: { coefficients: array(1, ['%dynamic'], REAL), x: REAL },
    result: REAL,
  }),
  stepwise:   () => ({
    args:   [array(1, ['%dynamic'], REAL), array(1, ['%dynamic'], REAL), REAL],
    kwargs: { edges: array(1, ['%dynamic'], REAL),
              values: array(1, ['%dynamic'], REAL), x: REAL },
    result: REAL,
  }),
  // Array constructors (spec §07). All variadic in shape; result type
  // stays dynamic since the shape arity varies. fill is generic over
  // element type; zeros/ones default to real; eye/onehot are reals.
  fill:   () => ({ args: [tvar('T')], kwargs: {},
                   result: deferred(), variadic: 'positional' }),
  zeros:  () => ({ args: [INTEGER], kwargs: {},
                   result: deferred(), variadic: 'positional' }),
  ones:   () => ({ args: [INTEGER], kwargs: {},
                   result: deferred(), variadic: 'positional' }),
  eye:    () => ({ args: [INTEGER], kwargs: { n: INTEGER },
                   result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  onehot: () => ({ args: [INTEGER, INTEGER], kwargs: { i: INTEGER, n: INTEGER },
                   result: array(1, ['%dynamic'], REAL) }),
  // Scalar restrictors (spec §07). Identity at runtime; type system
  // pins the canonical scalar category.
  boolean: () => ({ args: [any()], kwargs: {}, result: BOOLEAN }),
  integer: () => ({ args: [any()], kwargs: {}, result: INTEGER }),

  // Linear algebra (spec §07). Matrices are 2-D real arrays; vectors
  // are 1-D real arrays. Most results stay deferred — the static
  // type system can't easily express n×n preservation across
  // transpose / inv etc. without a square-matrix shape variable.
  // transpose / adjoint accept matrix OR vector OR tvector per spec
  // §07: "transpose | A | A^T | vectors, matrices". Result shape
  // depends on input shape — handled by inferTransposeAdjoint in
  // typeinfer.js (the static signature here is a permissive fallback
  // for callers that don't dispatch on `special`).
  transpose:      () => ({ args: [any()], kwargs: {},
                           result: deferred(), special: 'transpose' }),
  adjoint:        () => ({ args: [any()], kwargs: {},
                           result: deferred(), special: 'adjoint' }),
  trace:          () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL)],
                           kwargs: {}, result: REAL }),
  diagmat:        () => ({ args: [array(1, ['%dynamic'], REAL)],
                           kwargs: {},
                           result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  self_outer:     () => ({ args: [array(1, ['%dynamic'], REAL)],
                           kwargs: {},
                           result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  det:            () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL)],
                           kwargs: {}, result: REAL }),
  logabsdet:      () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL)],
                           kwargs: {}, result: REAL }),
  inv:            () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL)],
                           kwargs: {},
                           result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  linsolve:       () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL), any()],
                           kwargs: {}, result: deferred() }),
  lower_cholesky: () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL)],
                           kwargs: {},
                           result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  row_gram:       () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL)],
                           kwargs: {},
                           result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  col_gram:       () => ({ args: [array(2, ['%dynamic', '%dynamic'], REAL)],
                           kwargs: {},
                           result: array(2, ['%dynamic', '%dynamic'], REAL) }),

  // linspace(from, to, n) — endpoint-inclusive evenly-spaced vector.
  // extlinspace adds ±∞ overflow edges. Both produce real vectors.
  linspace:    () => ({ args: [REAL, REAL, INTEGER], kwargs: {
                          from: REAL, to: REAL, n: INTEGER },
                        result: array(1, ['%dynamic'], REAL) }),
  extlinspace: () => ({ args: [REAL, REAL, INTEGER], kwargs: {
                          from: REAL, to: REAL, n: INTEGER },
                        result: array(1, ['%dynamic'], REAL) }),
  // broadcast(f, x = A, ...) — handled specially. The integration
  // tests on fixtures using broadcast type-check before a sig was
  // added here; adding a strict sig regresses those, because the
  // first-arg type-recursion hits get_field on `standard_module`
  // refs that don't yet have full record-typing. Leave broadcast
  // out of SIGNATURE_FACTORIES for now; the value-evaluator handles
  // it at runtime via the dedicated case in sampler.evaluateCall.
  // TODO: add a proper sig once standard_module typing is wired
  // through get_field correctly.
  // reduce(f, xs) — left fold over xs with binary f. First element of
  // xs is the initial accumulator; result type follows xs's element.
  reduce: () => ({ args: [any(), array(1, ['%dynamic'], tvar('T'))],
                   kwargs: {}, result: tvar('T') }),
  // scan(f, init, xs) — left scan; produces a vector of intermediate
  // accumulator values, one per element of xs.
  scan:   () => ({ args: [any(), tvar('T'), array(1, ['%dynamic'], tvar('T'))],
                   kwargs: {},
                   result: array(1, ['%dynamic'], tvar('T')) }),
  // filter(predicate, data) — keep elements satisfying predicate.
  // The orchestrator rewrites this at lift time into a 3-arg shape
  // with the predicate's body embedded, so type-checking sees the
  // original surface (predicate, data) form. Result is dynamic-length.
  filter: () => ({ args: [any(), array(1, ['%dynamic'], tvar('T'))],
                   kwargs: {},
                   result: array(1, ['%dynamic'], tvar('T')) }),
  // partition(xs, spec) — split a vector into a vector of sub-vectors.
  // Static return type kept dynamic (the inner shape depends on spec).
  partition:   () => ({ args: [array(1, ['%dynamic'], any()), any()],
                        kwargs: {}, result: deferred() }),
  // reverse(xs) — same shape as input, elements reversed.
  reverse:     () => ({ args: [array(1, ['%dynamic'], tvar('T'))],
                        kwargs: {},
                        result: array(1, ['%dynamic'], tvar('T')) }),
  // addaxes(A, n_leading, n_trailing) — insert singular axes around A
  // (spec §07). Output rank = rank(A) + n_leading + n_trailing, so
  // the result type is dynamic (rank depends on the integer args),
  // same treatment as `array`. n_leading / n_trailing are fixed
  // non-negative integers.
  addaxes:     () => ({ args: [any(), INTEGER, INTEGER],
                        kwargs: {}, result: deferred() }),
  // array(data, size, dimorder) — n-D array from flat data + shape.
  // Result type stays dynamic (rank depends on size); the runtime
  // returns nested JS arrays of the appropriate shape.
  array: () => ({ args: [array(1, ['%dynamic'], REAL),
                         array(1, ['%dynamic'], INTEGER),
                         array(1, ['%dynamic'], INTEGER)],
                  kwargs: { data: array(1, ['%dynamic'], REAL),
                            size: array(1, ['%dynamic'], INTEGER),
                            dimorder: array(1, ['%dynamic'], INTEGER) },
                  result: deferred() }),
  // rowstack(vs) / colstack(vs) — vector-of-vectors → matrix. Inputs
  // must all have the same length (runtime check).
  rowstack:    () => ({ args: [array(1, ['%dynamic'],
                                     array(1, ['%dynamic'], REAL))],
                        kwargs: {},
                        result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  colstack:    () => ({ args: [array(1, ['%dynamic'],
                                     array(1, ['%dynamic'], REAL))],
                        kwargs: {},
                        result: array(2, ['%dynamic', '%dynamic'], REAL) }),
  // bincounts(bins, data) — count data points falling into bins. 1D
  // case: bins is a vector of edges, data is a vector of reals,
  // result is an integer-count vector of length n-1 (where n is the
  // edge count). Multi-D variants defer (record-of-edges shape).
  bincounts: () => ({
    args:   [array(1, ['%dynamic'], REAL), array(1, ['%dynamic'], REAL)],
    kwargs: { bins: array(1, ['%dynamic'], REAL),
              data: array(1, ['%dynamic'], REAL) },
    result: array(1, ['%dynamic'], INTEGER),
  }),
  // selectbins(edges, region, counts) — keep counts for bins whose
  // interval intersects `region`. The `region` argument is a set
  // expression; the sampler accepts literal interval(lo,hi) and the
  // named real sets (reals / posreals / nonnegreals / unitinterval).
  // Result length depends on region overlap — type stays dynamic.
  selectbins: () => ({
    args:   [array(1, ['%dynamic'], REAL), any(),
             array(1, ['%dynamic'], REAL)],
    kwargs: { edges: array(1, ['%dynamic'], REAL),
              region: any(),
              counts: array(1, ['%dynamic'], REAL) },
    result: array(1, ['%dynamic'], REAL),
  }),

  // ---- Logic / conditionals ----------------------------------------
  // Spec §07 — boolean ops. `lnot` is unary; the others are binary.
  // `ifelse(cond, a, b)` is the conditional expression — both branches
  // must unify (T, T → T) so the result type is a single T.
  land:   () => ({ args: [BOOLEAN, BOOLEAN], kwargs: {}, result: BOOLEAN }),
  lor:    () => ({ args: [BOOLEAN, BOOLEAN], kwargs: {}, result: BOOLEAN }),
  lxor:   () => ({ args: [BOOLEAN, BOOLEAN], kwargs: {}, result: BOOLEAN }),
  lnot:   () => ({ args: [BOOLEAN], kwargs: {}, result: BOOLEAN }),
  ifelse: () => ({ args: [BOOLEAN, tvar('T'), tvar('T')], kwargs: {}, result: tvar('T') }),

  // ---- Reductions / norms ------------------------------------------
  sum:     () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  mean:    () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  prod:    () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  lengthof: () => ({ args: [array(1, ['%dynamic'], any())], kwargs: {}, result: INTEGER }),
  // sizeof returns the shape vector of an array; length of the result
  // is the rank of the input. Static result-length is left dynamic
  // because the input rank isn't always statically known.
  sizeof:   () => ({ args: [array(1, ['%dynamic'], any())], kwargs: {}, result: array(1, ['%dynamic'], INTEGER) }),
  // Population variance (divisor N — not N-1). maximum/minimum on an
  // array; pluralised to avoid collision with the binary `min` / `max`
  // built-ins (per spec §07).
  var:     () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  maximum: () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  minimum: () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  // Norms and softmax family (spec §07). Single vector argument.
  // *norm / logsumexp produce a scalar; *unit / softmax / logsoftmax
  // produce a vector of the same length.
  l1norm:     () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  l2norm:     () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  logsumexp:  () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {}, result: REAL }),
  l1unit:     () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {},
                       result: array(1, ['%dynamic'], REAL) }),
  l2unit:     () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {},
                       result: array(1, ['%dynamic'], REAL) }),
  softmax:    () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {},
                       result: array(1, ['%dynamic'], REAL) }),
  logsoftmax: () => ({ args: [array(1, ['%dynamic'], REAL)], kwargs: {},
                       result: array(1, ['%dynamic'], REAL) }),

  // ---- RNG (per spec §sec:random) ----------------------------------
  // rnginit(seed) — seed is a byte-vector; result is an opaque rngstate.
  rnginit: () => ({ args: [array(1, ['%dynamic'], INTEGER)], kwargs: {}, result: RNGSTATE }),
  // rand(rstate, m) — draw one sample from m using rstate; returns
  // (value, new_rstate). The variate type is m's domain — we keep it
  // as a tvar('T') unified against measure(T)'s argument.
  rand: () => ({
    args:   [RNGSTATE, measure(tvar('T'))],
    kwargs: {},
    result: tuple([tvar('T'), RNGSTATE]),
  }),
  // rngstate(bytes) — reconstructs an rngstate from a byte serialisation.
  rngstate: () => ({ args: [array(1, ['%dynamic'], INTEGER)], kwargs: {}, result: RNGSTATE }),

  // ---- Density evaluation (per spec §sec:measure-algebra line 87-89,
  //      §sec:posterior line 433+, §sec:likelihoodof line 340+) -------
  // logdensityof(M, x) — log-density of M at x w.r.t. the implicit
  // reference measure. M may be a measure or a likelihood object;
  // x's type is M's domain.
  logdensityof: () => ({
    args: [measure(tvar('T')), tvar('T')],
    kwargs: {},
    result: REAL,
  }),
  // densityof(M, x) — exp(logdensityof(M, x)). Lowered to that in the
  // engine; first-class on the surface so spec-level density formulas
  // read naturally.
  densityof: () => ({
    args: [measure(tvar('T')), tvar('T')],
    kwargs: {},
    result: REAL,
  }),
  // totalmass(M) — scalar mass of a (possibly unnormalized) measure.
  // For closed-form M, this is a deterministic scalar; for measures
  // with parameterized weights it varies with the parameters. The
  // engine materialises it from each measure's tracked logTotalmass
  // (real-scale via exp) at sample time.
  totalmass: () => ({
    args: [measure(tvar('T'))],
    kwargs: {},
    result: REAL,
  }),
  // truncate(M, S) — restricts the support of measure M to set S.
  // Per spec §06: ν(A) = M(A ∩ S). Does NOT normalize, so the
  // resulting measure's totalmass shrinks to M(S). The set argument
  // is a bare set expression (built-in name like `posreals`, or a
  // call like `interval(0, inf)`); typed loosely here and validated
  // structurally by the orchestrator's classifier.
  truncate: () => ({
    args: [measure(tvar('T')), any()],
    kwargs: {},
    result: measure(tvar('T')),
  }),
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

function signatureOf(opName: string) {
  const f = (SIGNATURE_FACTORIES as any)[opName];
  if (!f) return null;
  // Re-key the signature's type variables so two call sites can't share
  // unification state. The module-level FRESH_COUNTER guarantees
  // cross-call uniqueness; the per-signature `sub` map keeps the
  // signature's internal sharing intact (so 'T' in args and 'T' in
  // result still bind to the same fresh variable).
  const sig = f();
  const sub = new Map();
  function fresh(t: any): any {
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
      const out: Record<string, any> = {};
      for (const k in t.fields) out[k] = fresh(t.fields[k]);
      return record(out);
    }
    return t;
  }
  const out: any = {
    args:     sig.args ? sig.args.map(fresh) : sig.args,
    kwargs:   {} as Record<string, any>,
    result:   fresh(sig.result),
    variadic: sig.variadic || null,
    special:  sig.special  || null,
  };
  for (const k in sig.kwargs) out.kwargs[k] = fresh(sig.kwargs[k]);
  return out;
}

/** Whether this op has a known signature. */
function hasSignature(opName: string) {
  return Object.prototype.hasOwnProperty.call(SIGNATURE_FACTORIES, opName);
}

module.exports = {
  // Constructors
  deferred, failed, any, scalar, array, tvector, record, tuple, measure, rngstate, tvar,
  funcType, kernelType,
  REAL, INTEGER, BOOLEAN, COMPLEX, STRING, RNGSTATE,
  // Operations
  equal, substitute, unify, unifyArith, show, isMeasure, isValue, isCallable,
  // Signatures
  signatureOf, hasSignature,
};
