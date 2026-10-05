'use strict';

// Portable pyhf helper definitions. Reuse the particle-physics interpolation
// implementation. This module adds names and signatures, not a JS fast path.
const T = require('./types.ts');

/** Unit-centered HistFactory code4, matching the FlatPPL reference definition:
 * normsys_factor(lo, hi, alpha) = hep.interp_poly6_exp(lo, 1.0, hi, alpha).
 * Ordinary call dispatch and broadcast handle argument binding and axes. */
function pyhfModule(
  interpPoly6Exp: (lo: number, nominal: number, hi: number, alpha: number) => number,
  interpPoly6Lin: (lo: number, nominal: number, hi: number, alpha: number) => number,
) {
  return {
    name: 'pyhf_helpers',
    compat: '0.1',
    bindings: new Map([
      ['normsys_factor', {
        kind: 'function' as const,
        sig: T.funcType([
          { name: 'lo', type: T.REAL },
          { name: 'hi', type: T.REAL },
          { name: 'alpha', type: T.REAL },
        ], T.REAL),
        impl: (lo: number, hi: number, alpha: number) => interpPoly6Exp(lo, 1, hi, alpha),
      }],
      ['histosys_shift', {
        kind: 'function' as const,
        sig: T.funcType([
          { name: 'lo', type: T.REAL },
          { name: 'nominal', type: T.REAL },
          { name: 'hi', type: T.REAL },
          { name: 'alpha', type: T.REAL },
        ], T.REAL),
        impl: (lo: number, nominal: number, hi: number, alpha: number) =>
          interpPoly6Lin(lo - nominal, 0, hi - nominal, alpha),
      }],
    ]),
  };
}

module.exports = { pyhfModule };
