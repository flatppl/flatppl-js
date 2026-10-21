'use strict';

// Spec parameter NAMES for the builtins whose signatures are positional.
//
// §04 uses these names when a callable cannot accept one positional input
// and therefore binds a sole record/table by name. Unary and optional-unary
// callables consume the whole object; type/domain validation follows binding.
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

const BUILTIN_PARAM_NAMES: Record<string, string[]> = {
  Counting:             ['support'],
  Lebesgue:             ['support'],
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
  checked:              ['value', 'condition'],
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
  cummax:               ['xs'],
  cummin:               ['xs'],
  cumprod:              ['xs'],
  cumsum:               ['xs'],
  densityof:            ['M', 'x'],
  det:                  ['A'],
  diag:                 ['A', 'k'],
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
  lall:                 ['xs'],
  land:                 ['a', 'b'],
  lany:                 ['xs'],
  le:                   ['a', 'b'],
  lengthof:             ['x'],
  linfnorm:             ['v'],
  linsolve:             ['A', 'b'],
  lnot:                 ['a'],
  log:                  ['x'],
  log2:                 ['x'],
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
  median:               ['xs'],
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
  quantile:             ['xs', 'p'],
  rand:                 ['m', 'rstate'],
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

// The same interface decision serves the inferred IR and the executable AST.
// Unary eligibility wins before domain or field-name checks (§04).
function splatDecision(opName: string, present: string[], sig: any): any {
  const T = require('./types.ts');
  const iface = T.builtinInterface(opName, sig);
  if (!iface || T.acceptsPositional(iface, 1)) return null;
  return namedDecision(iface, present);
}

function namedDecision(iface: any, present: string[]): any {
  const declared = iface.inputs.map((p: any) => p.name);
  const unbindable = present.filter(k => !declared.includes(k));
  const missing = declared.slice(0, iface.minArgs).filter((k: string) => !present.includes(k));
  if (unbindable.length || missing.length) return { unbindable, missing, declared };
  return { order: declared.filter((k: string) => present.includes(k)), declared };
}

module.exports = { BUILTIN_PARAM_NAMES, builtinParamNames, splatDecision, namedDecision };
