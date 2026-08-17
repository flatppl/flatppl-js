'use strict';

// Spec parameter NAMES for the builtins whose signatures are positional.
//
// §04 "Calling conventions" defines auto-splatting as equivalence to the
// KEYWORD form — "`f(record(a = x, b = y, ...))` and `f(table(a = x, b = y,
// ...))` are equivalent to `f(a = x, b = y, ...)`" — over the general rule that
// "Arguments are bound to inputs by name, the order of the arguments is not
// relevant". A splat therefore needs the callee's argument NAMES, and §04 makes
// a mismatch a static error: "A call with field or column names that do not
// match the callable's argument names is a static error."
//
// `SIGNATURE_FACTORIES` stores a distribution's parameters as kwargs, which
// already carry names, but stores every §07 function row as a positional `args`
// list with none. With no names the splat had nothing to check, so a sole
// positional record or table could not bind at all — `atan2(table(y = ys, x =
// xs))` was rejected although §04 makes it valid.
//
// Each list is the row's **Arguments** column, in declared order, read from §07
// (and from §06:110-111 for `densityof` / `logdensityof`, spelled there as
// `densityof(M, x)` and `logdensityof(M, x)`). Extracted mechanically from the
// spec tables at flatppl-design `9e35262` rather than transcribed by hand.
//
// Cross-checked against flatppl-rust's catalogue at `4c3baec`, whose 139 lists a
// reviewer verified against the same tables: 123 of the lists here also exist
// there and all 123 agree exactly, with zero disagreements.
//
// NOT here, deliberately:
//   - variadic rows (`cat`, `vector`, `superpose`, `kchain`, `jointchain`, and
//     `get`/`get0`'s trailing selectors) — §04 gives their inputs no names to
//     bind, so they stay permissive. Recorded in TODO-flatppl-js.md.
//   - distributions and other kwarg-signature ops — already named.
//   - the nine §04 carve-out names, which take an aggregate WHOLE and so must
//     never splat. They are filtered at the call site rather than omitted here,
//     because their names are still wanted for an ordinary keyword call.

const BUILTIN_PARAM_NAMES: Record<string, string[]> = {
  abs:                  ['x'],
  abs2:                 ['x'],
  acos:                 ['x'],
  acosh:                ['x'],
  add:                  ['a', 'b'],
  addaxes:              ['A', 'n_leading', 'n_trailing'],
  adjoint:              ['A'],
  asin:                 ['x'],
  asinh:                ['x'],
  atan:                 ['x'],
  atan2:                ['y', 'x'],
  atanh:                ['x'],
  bandedmat:            ['v', 'rows'],
  blockdiagmat:         ['mats'],
  boolean:              ['x'],
  builtin_fromnormal:   ['kernel', 'kernel_input', 'z'],
  builtin_fromuniform:  ['kernel', 'kernel_input', 'u'],
  builtin_logdensityof: ['kernel', 'kernel_input', 'x'],
  builtin_tonormal:     ['kernel', 'kernel_input', 'x'],
  builtin_touniform:    ['kernel', 'kernel_input', 'x'],
  ceil:                 ['x'],
  cis:                  ['theta'],
  col_gram:             ['A'],
  colstack:             ['vs'],
  complex:              ['re', 'im'],
  conj:                 ['x'],
  conv:                 ['v', 'kernel'],
  cos:                  ['x'],
  cosh:                 ['x'],
  cross:                ['a', 'b'],
  crosscorr:            ['v', 'kernel'],
  cumprod:              ['xs'],
  cumsum:               ['xs'],
  densityof:            ['M', 'x'],
  det:                  ['A'],
  diagmat:              ['x'],
  div:                  ['a', 'b'],
  divide:               ['a', 'b'],
  equal:                ['a', 'b'],
  exp:                  ['x'],
  expm1:                ['x'],
  fill:                 ['x', 'size'],
  filter:               ['pred', 'data'],
  floor:                ['x'],
  gamma:                ['x'],
  ge:                   ['a', 'b'],
  gt:                   ['a', 'b'],
  identity:             ['x'],
  ifelse:               ['cond', 'a', 'b'],
  iid:                  ['M', 'size'],
  imag:                 ['x'],
  indicesof:            ['x'],
  indicesof0:           ['x'],
  integer:              ['x'],
  inv:                  ['A'],
  invlogit:             ['x'],
  invprobit:            ['x'],
  isfinite:             ['x'],
  isinf:                ['x'],
  isnan:                ['x'],
  iszero:               ['x'],
  joinblocks:           ['A'],
  l1norm:               ['v'],
  l1unit:               ['v'],
  l2norm:               ['v'],
  l2unit:               ['v'],
  land:                 ['a', 'b'],
  le:                   ['a', 'b'],
  lengthof:             ['x'],
  linsolve:             ['A', 'b'],
  lnot:                 ['a'],
  log:                  ['x'],
  log10:                ['x'],
  log1p:                ['x'],
  logabsdet:            ['A'],
  logdensityof:         ['M', 'x'],
  loggamma:             ['x'],
  logit:                ['p'],
  logsoftmax:           ['v'],
  logsumexp:            ['v'],
  logweighted:          ['logweight', 'base'],
  lor:                  ['a', 'b'],
  lower_cholesky:       ['A'],
  lt:                   ['a', 'b'],
  lxor:                 ['a', 'b'],
  max:                  ['a', 'b'],
  maximum:              ['xs'],
  mean:                 ['xs'],
  min:                  ['a', 'b'],
  minimum:              ['xs'],
  mod:                  ['a', 'b'],
  mul:                  ['a', 'b'],
  neg:                  ['x'],
  normalize:            ['M'],
  ones:                 ['size'],
  partition:            ['xs', 'spec'],
  pow:                  ['base', 'exponent'],
  probit:               ['p'],
  prod:                 ['xs'],
  pushfwd:              ['f', 'M'],
  quadform:             ['A', 'x'],
  rand:                 ['rstate', 'm'],
  real:                 ['x'],
  reverse:              ['xs'],
  rnginit:              ['rngseed'],
  rngstate:             ['bytes'],
  round:                ['x'],
  row_gram:             ['A'],
  rowstack:             ['vs'],
  self_outer:           ['x'],
  sin:                  ['x'],
  sinh:                 ['x'],
  sizeof:               ['x'],
  softmax:              ['v'],
  splitblocks:          ['A', 'blocksize'],
  sqrt:                 ['x'],
  std:                  ['xs'],
  sub:                  ['a', 'b'],
  sum:                  ['xs'],
  tan:                  ['x'],
  tanh:                 ['x'],
  tile:                 ['A', 'size'],
  totalmass:            ['M'],
  trace:                ['A'],
  transpose:            ['A'],
  truncate:             ['M', 'S'],
  unequal:              ['a', 'b'],
  var:                  ['xs'],
  weighted:             ['weight', 'base'],
  zeros:                ['size'],
};

/** Spec argument names for `opName`, or null when it declares none. */
function builtinParamNames(opName: string): string[] | null {
  return Object.prototype.hasOwnProperty.call(BUILTIN_PARAM_NAMES, opName)
    ? BUILTIN_PARAM_NAMES[opName]
    : null;
}

// §04's single-input carve-out (design#78): "A callable with exactly one input
// whose documented domain admits records or tables is exempt and receives a sole
// positional record or table whole, so that `sum(t)` and `lengthof(t)` reduce
// over the table rather than splatting."
//
// These twelve MUST NOT splat. `sum`'s only argument is named `xs`, so splatting
// `sum(table(mass = …, pt = …))` would compare `{mass, pt}` against `{xs}` and
// reject a call §07's "Table reductions" paragraph defines. They keep their
// entries above because an ordinary keyword call still binds by name.
//
// The set: `sum`, `mean`, `var`, `std` (table domain from §07's "Table
// reductions" paragraph, `std` by owner ruling), `lengthof`, `reverse` (Domains
// `vectors, tables`), `indicesof`, `indicesof0` (Domains `vectors, arrays,
// tables`), `identity` (Domains `any`), and `prod`, `maximum`, `minimum`
// (table domain ahead of spec: flatppl-design PR #79 extends §07's "Table
// reductions" paragraph to name these three, matching this engine's existing
// column-wise behaviour; owner-merge pending, per TODO-flatppl-js.md).
const SPLAT_EXEMPT_BUILTINS: ReadonlySet<string> = new Set([
  'sum', 'mean', 'var', 'std', 'lengthof', 'reverse',
  'indicesof', 'indicesof0', 'identity', 'prod', 'maximum', 'minimum',
]);

/** Whether §04's carve-out exempts `opName` from splatting. */
function isSplatExemptBuiltin(opName: string): boolean {
  return SPLAT_EXEMPT_BUILTINS.has(opName);
}

// The §04 name decision, in ONE place. `present` is the record's field names or
// the table's column names. Returns:
//   null                       — no splat applies to this op at all
//   { order }                  — the declared order the fields bind in
//   { unbindable, missing }     — §04 static error, with the offending names
//
// Both rewrite sites (typeinfer's lowered IR and the analyzer's AST, which feed
// type inference and the evaluator respectively) call this rather than
// re-deriving the rule, because two sites drifting is precisely how the
// surplus-name gap in `resolveParams` / `resolveParamsN` arose.
// Formerly held `prod`, `maximum`, `minimum` pending the owner ruling that
// flatppl-design PR #79 now encodes (unmerged); moved to the carve-out set
// above ahead of spec merge — see its comment. Empty for now; a future op
// with unsettled table-reduction status goes here, not straight into the
// carve-out set.
const SPLAT_HELD_TABLE_REDUCTIONS: ReadonlySet<string> = new Set([]);

function splatDecision(opName: string, present: string[], arity: number): any {
  if (isSplatExemptBuiltin(opName)) return null;
  if (SPLAT_HELD_TABLE_REDUCTIONS.has(opName)) return null;
  const declared = builtinParamNames(opName);
  if (!declared || declared.length !== arity) return null;
  const unbindable = present.filter((k) => declared.indexOf(k) === -1);
  const missing = declared.filter((k) => present.indexOf(k) === -1);
  if (unbindable.length > 0 || missing.length > 0) return { unbindable, missing, declared };
  return { order: declared.slice(), declared };
}

module.exports = {
  BUILTIN_PARAM_NAMES, builtinParamNames,
  SPLAT_EXEMPT_BUILTINS, isSplatExemptBuiltin,
  SPLAT_HELD_TABLE_REDUCTIONS, splatDecision,
};
