'use strict';

// distribution-param-ranks.ts — declared RANK of each §08 distribution
// parameter, in axes.
// =====================================================================
//
// The signature and measure-expansion paths use these declared §08 ranks
// to distinguish scalar, vector and matrix parameters. Broadcasting itself
// iterates outer array axes; nested arrays supply vector/matrix cells.
// Dirac's point may have any rank.
const RANKS: Record<string, Record<string, number>> = {
  MvNormal: { mu: 1, cov: 2 },
  Dirichlet: { alpha: 1 },
  Multinomial: { n: 0, p: 1 },
  Categorical: { p: 1 },
  Categorical0: { p: 1 },
  Wishart: { nu: 0, scale: 2 },
  InverseWishart: { nu: 0, scale: 2 },
};

module.exports = { DISTRIBUTION_PARAM_RANKS: RANKS };
