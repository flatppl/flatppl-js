'use strict';

// distribution-param-ranks.ts — declared RANK of each §08 distribution
// parameter, in axes.
// =====================================================================
//
// §06's `ksuperpose` family rule measures a family argument against the
// parameter it feeds: "an argument's family axes are its leading axes in
// excess of the rank (number of axes) of the parameter it feeds, and any
// count other than one is a static error". That needs the DECLARED rank,
// which §08 gives in each entry's Parameters list — `MvNormal`'s `mu` is a
// "mean vector (array of reals, length n)" and its `cov` a "covariance
// matrix (n x n)", so their ranks are 1 and 2.
//
// Only the shaped rows are listed. Every other §08 parameter, and every §09
// module distribution's parameter, is a scalar — `paramRankOf` reports rank 0
// for a known distribution that is absent here, which is what keeps a
// two-axis argument over a scalar parameter a static error.
//
// A name this does not know (a reified kernel, a fundamental measure whose
// parameter is rank-polymorphic such as `Dirac`'s `value`) yields `null`,
// which the family rule reads as rank-polymorphic: the rank that leaves
// exactly one family axis.

// Parameters §06/§08 declare RANK-POLYMORPHIC. `Dirac(value)` takes a point of
// any type — §08 builds a categorical over arbitrary values as
// `normalize(ksuperpose(Dirac, p)(value = labels))` — so no fixed rank applies
// and the family rule takes the rank that leaves exactly one family axis.
const POLYMORPHIC: Record<string, string[]> = {
  Dirac: ['value'],
};

const RANKS: Record<string, Record<string, number>> = {
  MvNormal: { mu: 1, cov: 2 },
  Dirichlet: { alpha: 1 },
  Multinomial: { n: 0, p: 1 },
  Categorical: { p: 1 },
  Categorical0: { p: 1 },
  Wishart: { nu: 0, scale: 2 },
  InverseWishart: { nu: 0, scale: 2 },
};

// Declared rank of `dist`'s parameter `param`, or null when it is unknown.
// `isKnownDist` says whether `dist` is a distribution whose parameter list is
// declared, so an unlisted parameter of a listed distribution reads as scalar
// rather than as unknown.
function paramRankOf(dist: string | null, param: string | null,
                     isKnownDist: boolean): number | null {
  if (dist == null || param == null) return null;
  const poly = POLYMORPHIC[dist];
  if (poly && poly.indexOf(param) >= 0) return null;
  const row = RANKS[dist];
  if (row) {
    const r = row[param];
    return typeof r === 'number' ? r : (isKnownDist ? 0 : null);
  }
  return isKnownDist ? 0 : null;
}

module.exports = { paramRankOf, DISTRIBUTION_PARAM_RANKS: RANKS };
