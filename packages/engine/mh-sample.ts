// packages/engine/mh-sample.ts
'use strict';
const rng = require('./rng.ts');
const sampler = require('./sampler.ts');

function gaussian(prng: () => number): number {
  // Box-Muller for proposal noise (proposals need not match the engine's
  // inverse-CDF sampler; this stream is internal to MH).
  const u1 = Math.max(prng(), 1e-300), u2 = prng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function runChain(mv: any, y0: Float64Array, step: Float64Array, nWarmup: number, nDraws: number, prng: () => number, adapt: boolean) {
  const dim = mv.dim;
  let y = Float64Array.from(y0);
  let lp = mv.logPosterior(y);
  const out: Float64Array[] = [];               // kept draws (constrained), per iter as Float64Array(dim)
  let accepts = 0, total = 0;
  const target = dim === 1 ? 0.44 : 0.234;
  const total_iter = nWarmup + nDraws;
  for (let it = 0; it < total_iter; it++) {
    // Metropolis-within-Gibbs: one coordinate update per dimension per sweep.
    for (let d = 0; d < dim; d++) {
      const yProp = Float64Array.from(y);
      yProp[d] = y[d] + step[d] * gaussian(prng);
      const lpProp = mv.logPosterior(yProp);
      total++;
      const accepted = Math.log(prng() + 1e-300) < (lpProp - lp);
      if (accepted) { y = yProp; lp = lpProp; accepts++; }
      if (adapt && it < nWarmup) {
        // Robbins-Monro step-size adaptation toward target accept rate.
        const g = (accepted ? 1 : 0) - target;
        step[d] *= Math.exp(g / Math.sqrt(it + 1));
      }
    }
    if (it >= nWarmup) {
      const theta = mv.constrainAll(y);
      const row = new Float64Array(dim);
      for (let d = 0; d < dim; d++) row[d] = theta[mv.names[d]];
      out.push(row);
    }
  }
  return { out, acceptRate: accepts / total };
}

function mhSample(mv: any, opts: any) {
  const chains = opts.chains ?? 4;
  const warmup = opts.warmup ?? 1000;
  const draws = opts.draws ?? 1000;
  const seed = opts.seed ?? 0;
  const initStep = opts.initStep ?? 1;
  const baseKey = rng.keyFromSeed(seed);
  const chainKeys = rng.split(baseKey, chains);

  const perChain: Float64Array[][] = [];        // perChain[c] = array of Float64Array(dim) rows
  let acceptSum = 0;
  for (let c = 0; c < chains; c++) {
    const prng = sampler.makePhiloxPrngAdapter(rng.stateFromKey(chainKeys[c][0], chainKeys[c][1]));
    const y0 = new Float64Array(mv.dim);         // start at unconstrained origin
    const step = new Float64Array(mv.dim).fill(initStep);
    const r = runChain(mv, y0, step, warmup, draws, prng, true);
    perChain.push(r.out);
    acceptSum += r.acceptRate;
  }

  // Reshape to per-name flat draws + per-name per-chain arrays (for diagnostics).
  const drawsByName: any = {};
  const chainsByName: any = {};
  for (let d = 0; d < mv.dim; d++) {
    const name = mv.names[d];
    const flat = new Float64Array(chains * draws);
    const perChainArr: Float64Array[] = [];
    let k = 0;
    for (let c = 0; c < chains; c++) {
      const ch = new Float64Array(draws);
      for (let i = 0; i < draws; i++) { ch[i] = perChain[c][i][d]; flat[k++] = ch[i]; }
      perChainArr.push(ch);
    }
    drawsByName[name] = flat;
    chainsByName[name] = perChainArr;
  }
  return { drawsByName, chains: chainsByName, acceptRate: acceptSum / chains };
}

module.exports = { mhSample };
