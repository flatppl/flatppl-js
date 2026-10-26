'use strict';

// =====================================================================
// operator-domains.ts — the §07 "Operator-equivalent functions" domains
// =====================================================================
//
// Spec §07 "Operator-equivalent functions" gives every arithmetic and
// comparison function a Domains cell:
//
//   add, sub : scalars or arrays of same shape (real or complex)
//   mul      : scalars, matrix-matrix, matrix-vector, scalar-matrix, scalar-vector,
//              transposed-vector–vector, vector–transposed-vector,
//              transposed-vector–matrix
//   divide   : scalars, array-scalar, transposed-vector–scalar (real or complex)
//   neg      : scalars or arrays (real or complex)
//   pow      : scalars (real or complex)
//   equal, unequal : integers, booleans, strings
//   lt, le, gt, ge : reals
//
// §05 states the same for the infix spellings ("No implicit operator
// broadcasting"), and §03 admits booleans and integers wherever reals are
// admitted (`booleans ⊂ integers ⊂ reals`).
//
// This is the ONE statement of those domains. The typed call of every head
// (the infix operator, the named function, and the per-cell call of a
// `broadcast` over the head) asks `operatorDomainViolation`, so the table
// cannot drift between spellings. Operand types the engine has not resolved
// (`any`, `deferred`, a type variable) are admitted: nothing is proven wrong.

const T = require('./types.ts');

/** The heads this module judges. */
const OPERATOR_DOMAIN_OPS = new Set([
  'add', 'sub', 'mul', 'divide', 'neg', 'pow',
  'equal', 'unequal', 'lt', 'le', 'gt', 'ge',
]);

const NUMERIC = new Set(['boolean', 'integer', 'real', 'complex']);
const DISCRETE = new Set(['boolean', 'integer', 'string']);
const REAL_VALUED = new Set(['boolean', 'integer', 'real']);

// Result types the engine guessed from an operand it had not resolved (`_ + 1` is typed as the
// literal's scalar). A guess proves nothing, so it is judged like an unresolved type.
const guessed = new WeakSet<object>();

/** A copy of `t` that is judged as unresolved wherever it later appears as an operand. */
function asGuess(t: any): any {
  const copy = { ...t };
  guessed.add(copy);
  return copy;
}

function unresolved(t: any): boolean {
  return !t || ['any', 'deferred', 'var'].includes(t.kind) || guessed.has(t);
}

function isScalar(t: any, prims: Set<string>): boolean {
  return t?.kind === 'scalar' && prims.has(t.prim);
}

/** The scalar leaf of an array or transposed vector, through any nesting. */
function leaf(t: any): any {
  return t?.kind === 'array' || t?.kind === 'tvector' ? leaf(t.elem) : t;
}

/** An array whose scalar cells are all numeric (any rank). */
function isNumericArray(t: any): boolean {
  return t?.kind === 'array' && (isScalar(leaf(t), NUMERIC) || unresolved(leaf(t)));
}

/** The extents of an array, nested arrays included. */
function extents(t: any): any[] {
  if (t?.kind === 'array') return [...t.shape, ...extents(t.elem)];
  if (t?.kind === 'tvector') return [t.length, ...extents(t.elem)];
  return [];
}

const sameExtent = (a: any, b: any) => a === '%dynamic' || b === '%dynamic' || a === b;

function sameShape(a: any, b: any): boolean {
  const x = extents(a), y = extents(b);
  return x.length === y.length && x.every((n, i) => sameExtent(n, y[i]));
}

/** `rank` of a numeric array of scalars (a vector is 1, a matrix 2), else 0. */
function scalarArrayRank(t: any): number {
  return t?.kind === 'array' && t.elem?.kind !== 'array' && isNumericArray(t) ? t.rank : 0;
}

type Kind = 'scalar' | 'vector' | 'matrix' | 'tvector' | 'other';

function kindOf(t: any): Kind {
  if (isScalar(t, NUMERIC)) return 'scalar';
  if (t?.kind === 'tvector' && (isScalar(leaf(t), NUMERIC) || unresolved(leaf(t)))) return 'tvector';
  const rank = scalarArrayRank(t);
  return rank === 1 ? 'vector' : rank === 2 ? 'matrix' : 'other';
}

/** The extent along `axis` of a vector, matrix or transposed vector, or null when not static. */
function extentOf(t: any, axis: number): number | null {
  const n = t?.kind === 'tvector' ? (axis === 0 ? t.length : null) : t?.shape?.[axis];
  return typeof n === 'number' ? n : null;
}

const SPEC = 'spec §07 "Operator-equivalent functions"';

// The arithmetic rows take real or complex arrays, and §03 does not read a vector of vectors as
// a matrix, so an array of arrays lies outside each of them.
const ARITHMETIC = new Set(['add', 'sub', 'mul', 'divide', 'neg', 'pow']);

function isArrayOfArrays(t: any): boolean {
  return (t?.kind === 'array' || t?.kind === 'tvector')
    && (t.elem?.kind === 'array' || t.elem?.kind === 'tvector');
}

function describe(types: any[]): string {
  return types.map(t => T.show(t)).join(' and ');
}

/**
 * Why the operand types lie outside the Domains cell of operator-equivalent `op`, or null when
 * they are inside it (or not yet resolved). The text names the spec's own wording.
 */
function operatorDomainViolation(op: string, types: any[]): string | null {
  if (!OPERATOR_DOMAIN_OPS.has(op) || types.some(unresolved)) return null;
  const [a, b] = types;
  const nested = ARITHMETIC.has(op) ? types.find(isArrayOfArrays) : undefined;
  if (nested) {
    return op + ': operand ' + T.show(nested) + ' is an array of arrays, outside the domain ' + SPEC
      + ' gives `' + op + '`; `rowstack(…)` makes a matrix of a vector of vectors (spec §03), and '
      + 'the elementwise form `' + op + '.(…)` applies `' + op + '` to each element (spec §04)';
  }
  switch (op) {
    case 'add':
    case 'sub': {
      if (isScalar(a, NUMERIC) && isScalar(b, NUMERIC)) return null;
      if (isNumericArray(a) && isNumericArray(b)) {
        if (sameShape(a, b)) return null;
        return op + ': operand types ' + describe(types) + ' are not numerically compatible — ' + op
          + ' takes arrays of the same shape (' + SPEC + ')';
      }
      return op + ': operand types ' + describe(types) + ' are not numerically compatible — ' + op
        + ' takes scalars or arrays of the same shape (real or complex) (' + SPEC + '); the '
        + 'elementwise form is `' + op + '.(a, b)`';
    }
    case 'mul': {
      const ka = kindOf(a), kb = kindOf(b);
      const ok = (ka === 'scalar' && kb !== 'other' && kb !== 'tvector')
        || (kb === 'scalar' && (ka === 'vector' || ka === 'matrix'))
        || (ka === 'matrix' && (kb === 'matrix' || kb === 'vector'))
        || (ka === 'tvector' && (kb === 'vector' || kb === 'matrix'))
        || (ka === 'vector' && kb === 'tvector');
      if (!ok) {
        return 'mul: operand types ' + describe(types) + ' are not numerically compatible — mul takes '
          + 'scalars, matrix-matrix, matrix-vector, scalar-matrix, scalar-vector, '
          + 'transposed-vector–vector, vector–transposed-vector or transposed-vector–matrix ('
          + SPEC + '); the elementwise form is `mul.(a, b)`';
      }
      // The contracted extents must agree where they are known.
      const inner: Array<[number | null, number | null]> =
        ka === 'matrix' ? [[extentOf(a, 1), extentOf(b, 0)]]
        : ka === 'tvector' ? [[extentOf(a, 0), extentOf(b, 0)]] : [];
      for (const [x, y] of inner) {
        if (x != null && y != null && x !== y) {
          return 'mul: the contracted extents ' + x + ' and ' + y + ' differ for ' + describe(types)
            + ' (' + SPEC + ')';
        }
      }
      return null;
    }
    case 'divide': {
      const dividend = isScalar(a, NUMERIC) || isNumericArray(a) || kindOf(a) === 'tvector';
      if (dividend && isScalar(b, NUMERIC)) return null;
      if (dividend) {
        return 'divide: arg 2 (the divisor) expects a scalar, got ' + T.show(b) + ' (' + SPEC + ' gives `divide` the '
          + 'domains "scalars, array-scalar, transposed-vector–scalar", all with a scalar divisor); '
          + 'the elementwise form is `divide.(a, b)`';
      }
      return 'divide: operand types ' + describe(types) + ' are not numerically compatible — divide takes '
        + 'scalars, array-scalar or transposed-vector–scalar (real or complex) (' + SPEC + ')';
    }
    case 'neg': {
      if (isScalar(a, NUMERIC) || isNumericArray(a)) return null;
      return 'neg takes a scalar or an array (real or complex), got ' + T.show(a) + ' (' + SPEC + ')';
    }
    case 'pow': {
      if (isScalar(a, NUMERIC) && isScalar(b, NUMERIC)) return null;
      return 'pow: operand types ' + describe(types) + ' are not numerically compatible — pow takes '
        + 'scalars (real or complex) (' + SPEC + '); the elementwise form is `pow.(a, b)`';
    }
    case 'equal':
    case 'unequal': {
      const strings = isScalar(a, new Set(['string'])) || isScalar(b, new Set(['string']));
      const bothDiscrete = isScalar(a, DISCRETE) && isScalar(b, DISCRETE);
      const mixed = strings && !(isScalar(a, new Set(['string'])) && isScalar(b, new Set(['string'])));
      if (bothDiscrete && !mixed) return null;
      const offender = !isScalar(a, DISCRETE) ? a : !isScalar(b, DISCRETE) ? b : null;
      if (offender == null) {
        return op + ' compares a string only with a string, got ' + describe(types) + ' (' + SPEC + ')';
      }
      return op + ' takes integers, booleans and strings, but an operand has type ' + T.show(offender)
        + ' (' + SPEC + '): exact equality is restricted to discrete domains; compare real-valued '
        + 'quantities through `integer(x)`, `floor(x)`, `ceil(x)` or `round(x)`';
    }
    default: {
      // lt, le, gt, ge
      if (isScalar(a, REAL_VALUED) && isScalar(b, REAL_VALUED)) return null;
      const offender = !isScalar(a, REAL_VALUED) ? a : b;
      return op + ' takes reals, but an operand has type ' + T.show(offender) + ' (' + SPEC
        + '); the elementwise form is `' + op + '.(a, b)`';
    }
  }
}

module.exports = { OPERATOR_DOMAIN_OPS, operatorDomainViolation, kindOf, extentOf, asGuess, unresolved };
