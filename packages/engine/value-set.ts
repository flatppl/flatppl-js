'use strict';

// =====================================================================
// value-set.ts — statically known value sets (spec §03 / §11 valueset
// domain; engine-concepts §17.3)
// =====================================================================
//
// ONE owner for the value-set vocabulary, mirroring flatppl-rust
// `core::ty::ValueSet`. Consumed by typeinfer's valueset inference
// (producer), pir-sexpr's `%meta` emitter (the third slot), and the
// mass pass (`isBounded` drives the Lebesgue/Counting rule). A node's
// value set is the strongest statically known set containing its value
// (for a measure node, its support); the vocabulary is the existing §03
// set vocabulary — no new language. Conservative: the vocabulary is NOT
// intersection-closed, so `UNKNOWN` is always a sound fallback.
//
// Representation: atom sets are plain strings; the parameterised sets
// (`interval` / `stdsimplex` / `cartpow`) are tagged `{vs, …}` objects.
// `DEFERRED` (not yet inferred) and `UNKNOWN` (inferred, no constraint)
// are distinct, per spec §11.

const DEFERRED = 'deferred';
const UNKNOWN = 'unknown';
const REALS = 'reals';
const POSREALS = 'posreals';
const NONNEGREALS = 'nonnegreals';
const UNITINTERVAL = 'unitinterval';
const INTEGERS = 'integers';
const POSINTEGERS = 'posintegers';
const NONNEGINTEGERS = 'nonnegintegers';
const BOOLEANS = 'booleans';
const COMPLEXES = 'complexes';
const RNGSTATES = 'rngstates';
const ANYTHING = 'anything';

function interval(lo: number, hi: number): any { return { vs: 'interval', lo, hi }; }
function stdsimplex(n: any): any { return { vs: 'stdsimplex', n }; }     // n: number | '%dynamic'
function cartpow(elem: any, n: any): any { return { vs: 'cartpow', elem, n }; }

function equal(a: any, b: any): boolean {
  if (a === b) return true;
  if (!a || !b || typeof a === 'string' || typeof b === 'string') return false;
  if (a.vs !== b.vs) return false;
  if (a.vs === 'interval') return a.lo === b.lo && a.hi === b.hi;
  if (a.vs === 'stdsimplex') return a.n === b.n;
  if (a.vs === 'cartpow') return a.n === b.n && equal(a.elem, b.elem);
  return false;
}

// The natural extent of a type: the maximal set its values inhabit
// (spec §11 — a value-typed node's set is at least this). UNKNOWN for
// non-value types (callables, modules, type vars). Mirrors Rust
// `ValueSet::natural_of`.
function naturalOf(t: any): any {
  if (!t || typeof t !== 'object') return UNKNOWN;
  switch (t.kind) {
    case 'scalar':
      return t.prim === 'real' ? REALS
        : t.prim === 'integer' ? INTEGERS
        : t.prim === 'boolean' ? BOOLEANS
        : t.prim === 'complex' ? COMPLEXES : UNKNOWN;
    case 'array': {
      // Nested arrays (vec-of-vec) carry no flat set vocabulary entry;
      // only rank-1 arrays of scalars map to a cartpow.
      if (Array.isArray(t.shape) && t.shape.length === 1) {
        const inner = naturalOf(t.elem);
        return inner === UNKNOWN ? UNKNOWN : cartpow(inner, t.shape[0]);
      }
      return UNKNOWN;
    }
    case 'measure': return naturalOf(t.domain);
    case 'rngstate': return RNGSTATES;
    case 'any': return ANYTHING;
    default: return UNKNOWN;
  }
}

// Is the set bounded? true / false / null (not statically known).
// Drives the Lebesgue/Counting and truncate total-mass rules. Mirrors
// Rust `ValueSet::is_bounded`.
function isBounded(vs: any): boolean | null {
  if (vs === UNITINTERVAL || vs === BOOLEANS) return true;
  if (vs === REALS || vs === POSREALS || vs === NONNEGREALS
    || vs === INTEGERS || vs === POSINTEGERS || vs === NONNEGINTEGERS
    || vs === COMPLEXES) return false;
  if (vs === DEFERRED || vs === UNKNOWN || vs === ANYTHING || vs === RNGSTATES) return null;
  if (vs && vs.vs === 'stdsimplex') return true;
  if (vs && vs.vs === 'interval') {
    return Number.isFinite(vs.lo) && Number.isFinite(vs.hi);
  }
  if (vs && vs.vs === 'cartpow') {
    return vs.n === '%dynamic' ? null : isBounded(vs.elem);
  }
  return null;
}

// Conservative subset check: true means `a ⊆ b` is PROVEN; false means
// unproven (NOT disproven). Mirrors Rust `ValueSet::subset_of`.
function subsetOf(a: any, b: any): boolean {
  if (equal(a, b)) return a !== DEFERRED && a !== UNKNOWN;
  if (b === ANYTHING) return a !== DEFERRED && a !== UNKNOWN;
  // Named-set lattice.
  if ((a === POSREALS || a === NONNEGREALS || a === UNITINTERVAL) && b === REALS) return true;
  if ((a === POSREALS || a === UNITINTERVAL) && b === NONNEGREALS) return true;
  if ((a === POSINTEGERS || a === NONNEGINTEGERS) && (b === INTEGERS || b === REALS)) return true;
  if (a === POSINTEGERS && (b === NONNEGINTEGERS || b === POSREALS || b === NONNEGREALS)) return true;
  if (a === NONNEGINTEGERS && b === NONNEGREALS) return true;
  if (a === INTEGERS && b === REALS) return true;
  if (a && a.vs === 'interval') {
    if (b === REALS) return true;
    if (b === NONNEGREALS) return a.lo >= 0.0;
    if (b === POSREALS) return a.lo > 0.0;
    if (b === UNITINTERVAL) return a.lo >= 0.0 && a.hi <= 1.0;
    return false;
  }
  if (a && a.vs === 'stdsimplex' && b && b.vs === 'cartpow') {
    return (a.n === b.n || b.n === '%dynamic') && subsetOf(UNITINTERVAL, b.elem);
  }
  if (a && a.vs === 'cartpow' && b && b.vs === 'cartpow') {
    return (a.n === b.n || b.n === '%dynamic') && subsetOf(a.elem, b.elem);
  }
  return false;
}

// Render a real for the `%mass`/valueset slot, matching Rust
// `render_real` (append `.0` to a bare integer-looking value).
function _renderReal(r: number): string {
  if (r === Infinity) return 'inf';
  if (r === -Infinity) return '(neg inf)';
  const s = String(r);
  return (/[.eE]/.test(s) || s.includes('inf') || s.includes('NaN')) ? s : s + '.0';
}
function _renderDim(n: any): string { return n === '%dynamic' ? '%dynamic' : String(n); }

// Render a value set to its `%meta` slot form (a §03 set expression).
function toSexpr(vs: any): string {
  if (vs === undefined || vs === DEFERRED) return '%deferred';
  if (vs === UNKNOWN) return '%unknown';
  if (typeof vs === 'string') return vs;     // the named atom sets
  if (vs.vs === 'stdsimplex') return '(stdsimplex ' + _renderDim(vs.n) + ')';
  if (vs.vs === 'interval') return '(interval ' + _renderReal(vs.lo) + ' ' + _renderReal(vs.hi) + ')';
  if (vs.vs === 'cartpow') return '(cartpow ' + toSexpr(vs.elem) + ' ' + _renderDim(vs.n) + ')';
  return '%unknown';
}

// A predefined set constant name → its ValueSet.
function constSetValueset(name: string): any {
  switch (name) {
    case 'reals': return REALS;
    case 'posreals': return POSREALS;
    case 'nonnegreals': return NONNEGREALS;
    case 'unitinterval': return UNITINTERVAL;
    case 'integers': return INTEGERS;
    case 'posintegers': return POSINTEGERS;
    case 'nonnegintegers': return NONNEGINTEGERS;
    case 'booleans': return BOOLEANS;
    case 'complexes': return COMPLEXES;
    case 'rngstates': return RNGSTATES;
    case 'anything': return ANYTHING;
    default: return UNKNOWN;
  }
}

// A set EXPRESSION (an `elementof` / `truncate` / reference-measure
// argument) read structurally into a ValueSet. `resolveDim(ir)` →
// number | null resolves a dimension argument (caller supplies a
// const-eval-capable resolver; defaults to literal-only). Mirrors Rust
// `set_expr_valueset`.
function setExprValueset(setIR: any, resolveDim?: (ir: any) => any): any {
  if (!setIR) return UNKNOWN;
  if ((setIR.kind === 'const' || setIR.kind === 'ref') && typeof setIR.name === 'string') {
    return constSetValueset(setIR.name);
  }
  if (setIR.kind !== 'call') return UNKNOWN;
  const args = setIR.args || [];
  switch (setIR.op) {
    case 'interval': {
      const lo = _resolveBound(args[0]);
      const hi = _resolveBound(args[1]);
      return (lo == null || hi == null) ? UNKNOWN : interval(lo, hi);
    }
    case 'stdsimplex': {
      const n = _dim(args[0], resolveDim);
      return stdsimplex(n);
    }
    case 'cartpow': {
      const elem = setExprValueset(args[0], resolveDim);
      if (elem === UNKNOWN) return UNKNOWN;
      return cartpow(elem, _dim(args[1], resolveDim));
    }
    // cartprod has no flat-vocabulary entry (record/tuple of sets).
    default: return UNKNOWN;
  }
}

function _dim(ir: any, resolveDim?: (ir: any) => any): any {
  if (ir && ir.kind === 'lit' && Number.isInteger(ir.value)) return ir.value;
  if (resolveDim) { const v = resolveDim(ir); if (v != null) return v; }
  return '%dynamic';
}

// Resolve an interval bound to a number (literal, `inf`, `neg(inf)`).
function _resolveBound(ir: any): number | null {
  if (!ir) return null;
  if (ir.kind === 'lit' && typeof ir.value === 'number') return ir.value;
  if ((ir.kind === 'const' || ir.kind === 'ref') && ir.name === 'inf') return Infinity;
  if (ir.kind === 'call' && ir.op === 'neg' && ir.args && ir.args.length === 1) {
    const inner = _resolveBound(ir.args[0]);
    return inner == null ? null : -inner;
  }
  return null;
}

// The value set of a scalar literal (spec §11 producer).
function literalValueset(litNode: any): any {
  if (!litNode || litNode.kind !== 'lit') return UNKNOWN;
  const v = litNode.value;
  if (typeof v === 'boolean') return BOOLEANS;
  if (typeof v === 'number') {
    if (Number.isInteger(v) && litNode.numType !== 'real') {
      return v > 0 ? POSINTEGERS : v === 0 ? NONNEGINTEGERS : INTEGERS;
    }
    return interval(v, v);
  }
  return UNKNOWN;
}

// A predefined value constant → its set.
function constValueset(name: string): any {
  if (name === 'pi' || name === 'inf') return POSREALS;
  if (name === 'im') return COMPLEXES;
  return UNKNOWN;
}

module.exports = {
  DEFERRED, UNKNOWN, REALS, POSREALS, NONNEGREALS, UNITINTERVAL,
  INTEGERS, POSINTEGERS, NONNEGINTEGERS, BOOLEANS, COMPLEXES, RNGSTATES, ANYTHING,
  interval, stdsimplex, cartpow,
  equal, naturalOf, isBounded, subsetOf, toSexpr,
  constSetValueset, setExprValueset, literalValueset, constValueset,
};
