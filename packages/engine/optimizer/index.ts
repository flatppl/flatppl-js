'use strict';

// =====================================================================
// optimizer/index.ts — public surface of the optimizer
// =====================================================================
//
// The "Find maximum" feature (TODO §"Plot optimizer + adaptive IS"). The
// optimizer is a pure, dependency-injected engine module: callers pass an
// async `evalCloud` (a batch of original-space points → fitness, MAXIMISED)
// and per-input value-set domains + plot-range scales; it returns a
// `ModeFit` (mode + Laplace covariance + diagnostics). The viewer wires
// `evalCloud` to the worker (evaluateN / logDensityN(pointsBatched)); the
// Laplace covariance is the reusable artefact the adaptive-IS step turns
// into an importance-sampling proposal.

const { optimize, registerOptimizer, OPTIMIZERS } = require('./optimize.ts');
const { makeCoords } = require('./coords.ts');
const { cmaes, mulberry32 } = require('./cmaes.ts');
const { polish, fdGradient, fdHessian, laplaceCovariance } = require('./polish.ts');
const { symEig, matSqrtAndInvSqrt } = require('./linalg.ts');

module.exports = {
  optimize, registerOptimizer, OPTIMIZERS,
  makeCoords, cmaes, mulberry32,
  polish, fdGradient, fdHessian, laplaceCovariance,
  symEig, matSqrtAndInvSqrt,
};
