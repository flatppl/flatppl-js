// packages/engine/mcmc-driver.ts
'use strict';
const rng = require('./rng.ts');
const sampler = require('./sampler.ts');
const diagnostics = require('./diagnostics.ts');

// Box-Muller gaussian noise from a U(0,1) prng. Shared proposal primitive
// (proposal noise need not match the engine's inverse-CDF sampler).
function gaussianNoise(prng: () => number): number {
  const u1 = Math.max(prng(), 1e-300), u2 = prng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

// Fixed-proposal batched Metropolis step. Each walker proposes
// y' = y + scale·L·z (z ~ N(0,I), L lower-triangular Cholesky factor), the whole
// sweep is scored in ONE logPosteriorBatch, then each walker accepts independently.
// RNG is drawn per walker in the scalar-path order (proposal noise, then the
// accept-uniform) so the prng stream is bit-identical to a per-walker scalar loop —
// one batched likelihood eval instead of nWalkers scalar ones. Shared by the mh
// (sample-phase) and RAM kernels; both run FIXED-proposal chains outside warmup.
function fixedProposalBatchStep(ensemble: Float64Array[], logp: Float64Array, mv: any, prng: () => number, L: Float64Array, scale: number, dim: number, z: Float64Array) {
  const nWalkers = ensemble.length;
  const props: Float64Array[] = new Array(nWalkers);
  const us = new Float64Array(nWalkers);
  for (let w = 0; w < nWalkers; w++) {
    const y = ensemble[w];
    for (let d = 0; d < dim; d++) z[d] = gaussianNoise(prng);
    const yProp = Float64Array.from(y);
    for (let i = 0; i < dim; i++) {
      let acc = 0;
      for (let k = 0; k <= i; k++) acc += L[i * dim + k] * z[k];
      yProp[i] = y[i] + scale * acc;
    }
    props[w] = yProp; us[w] = prng();
  }
  const lps = mv.logPosteriorBatch(props);
  let accepts = 0, proposals = 0;
  for (let w = 0; w < nWalkers; w++) {
    proposals++;
    if (Math.log(us[w] + 1e-300) < (lps[w] - logp[w])) {
      ensemble[w] = props[w]; logp[w] = lps[w]; accepts++;
    }
  }
  return { accepts, proposals };
}

// Ensemble MCMC driver shared by every kernel. Holds nWalkers positions in
// unconstrained ℝⁿ; each iteration delegates the move to kernel.step.
function runMcmc(mv: any, kernel: any, opts: any) {
  const dim = mv.dim;
  const nWalkers = opts.nWalkers ?? 4;
  const warmup = opts.warmup ?? 1000;
  const draws = opts.draws ?? 1000;
  const seed = opts.seed ?? 0;
  const initSpread = opts.initSpread ?? 1;

  const baseKey = rng.keyFromSeed(seed);
  const prng = sampler.makePhiloxPrngAdapter(rng.stateFromKey(baseKey[0], baseKey[1]));

  // Ensemble init. `opts.initPositions` (one unconstrained Float64Array per
  // walker) starts each walker from a PRIOR DRAW — in-region and overdispersed,
  // which matters when the prior isn't centred at the origin (e.g. mu~Normal(100,…))
  // or has a degenerate corner the origin sits near (Student-t nu). Without it,
  // fall back to a dispersed ball around the origin (dispersion is required for
  // emcee so identical walkers don't collapse the stretch move).
  const init: Float64Array[] | null = opts.initPositions
    || (typeof mv.initFromPrior === 'function' ? mv.initFromPrior(nWalkers, prng) : null);
  const ensemble = new Array(nWalkers);
  const logp = new Float64Array(nWalkers);
  for (let w = 0; w < nWalkers; w++) {
    let y: Float64Array;
    if (init && init[w]) {
      y = Float64Array.from(init[w]);
    } else {
      y = new Float64Array(dim);
      for (let d = 0; d < dim; d++) y[d] = initSpread * gaussianNoise(prng);
    }
    ensemble[w] = y;
    logp[w] = mv.logPosterior(y);
  }

  const adaptState = kernel.init ? kernel.init(nWalkers, dim, opts, mv) : {};
  // Frozen proposal from a prior warmup phase ({L, scale}). Running with warmup=0
  // then makes every step a fixed-proposal sampling step (no re-adaptation), so
  // independent chains can be distributed across workers after adaptation stops.
  if (opts.initAdapt) {
    if (opts.initAdapt.L) adaptState.L = opts.initAdapt.L;
    if (typeof opts.initAdapt.scale === 'number') adaptState.scale = opts.initAdapt.scale;
  }
  const collected = new Array(nWalkers);
  for (let w = 0; w < nWalkers; w++) collected[w] = [];
  let acceptTotal = 0, proposalTotal = 0;

  const onProgress = typeof opts.onProgress === 'function' ? opts.onProgress : null;
  const total = warmup + draws;
  const progStep = Math.max(1, Math.floor(total / 50));   // ~2% granularity
  for (let it = 0; it < total; it++) {
    const phase = it < warmup ? 'warmup' : 'sample';
    const r = kernel.step(ensemble, logp, mv, prng, adaptState, phase);
    acceptTotal += r.accepts; proposalTotal += r.proposals;
    if (onProgress && (it % progStep === 0 || it === total - 1)) onProgress((it + 1) / total, phase);
    if (it >= warmup) {
      for (let w = 0; w < nWalkers; w++) {
        const theta = mv.constrainAll(ensemble[w]);
        const row = new Float64Array(dim);
        for (let d = 0; d < dim; d++) row[d] = theta[mv.names[d]];
        collected[w].push(row);
      }
    }
  }

  const drawsByName: Record<string, Float64Array> = {}, walkersByName: Record<string, Float64Array[]> = {}, perParam: Record<string, any> = {};
  for (let d = 0; d < dim; d++) {
    const name = mv.names[d];
    const flat = new Float64Array(nWalkers * draws);
    const perWalker = new Array(nWalkers);
    let k = 0;
    for (let w = 0; w < nWalkers; w++) {
      const ws = new Float64Array(draws);
      for (let i = 0; i < draws; i++) { ws[i] = collected[w][i][d]; flat[k++] = ws[i]; }
      perWalker[w] = ws;
    }
    drawsByName[name] = flat;
    walkersByName[name] = perWalker;
    perParam[name] = { rHat: diagnostics.splitRHat(perWalker), essBulk: diagnostics.essBulk(perWalker) };
  }
  const acceptRate = proposalTotal > 0 ? acceptTotal / proposalTotal : 0;
  // endPositions = final unconstrained ensemble; a warmup-only run (draws=0)
  // returns it (with the tuned adaptState) so a later sampling phase on another
  // worker can resume the warmed chains under the frozen proposal.
  const endPositions = ensemble.map((y: Float64Array) => Float64Array.from(y));
  return { drawsByName, walkers: walkersByName, acceptRate, adaptState, endPositions, diagnostics: { acceptRate, perParam } };
}

module.exports = { runMcmc, gaussianNoise, fixedProposalBatchStep };
