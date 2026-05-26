'use strict';

// Per-kind materialiser handlers for multivariate distributions.
//
// Spec §08: MvNormal / Dirichlet / Multinomial / Wishart /
// InverseWishart / LKJ / LKJCholesky / BinnedPoissonProcess. Each
// produces a vector- or matrix-atom EmpiricalMeasure (Value
// shape=[N, ...intrinsic-dims], atom-major). All eight currently
// require atom-independent parameters — per-atom params (kernel-body
// hierarchies over a multivariate distribution) are a documented
// follow-up (TODO §08).
//
// Each handler returns a Promise<EmpiricalMeasure>. The shared
// plumbing (measureFromValue / nameSeed / collectRefArrays /
// prepareDensityRefs) lives in `./materialiser-shared.ts`.

import type {
  DerivationBinnedPoissonProcess,
  DerivationDirichlet,
  DerivationInverseWishart,
  DerivationLKJ,
  DerivationLKJCholesky,
  DerivationMultinomial,
  DerivationMvNormal,
  DerivationWishart,
  EmpiricalMeasure,
} from './engine-types';

const orchestrator = require('./orchestrator.ts');
const valueLib     = require('./value.ts');
const shared       = require('./materialiser-shared.ts');

const { nameSeed, measureFromValue } = shared;

// =====================================================================
// matMvNormal — atom-batched n-vector draws
// =====================================================================

function matMvNormal(name: string, d: DerivationMvNormal, ctx: any) {
  // MvNormal(mu, cov) — per spec §08, samples are n-vectors with
  // x ~ Normal_n(mu, cov). Implementation routes through the spec
  // equivalence  pushfwd(fn(mu + L*_), iid(Normal(0,1), n))  but
  // without the AST rewrite — we do the matvec / vector-add directly
  // via value-ops so the per-atom batched form short-circuits.
  //
  // Pipeline:
  //   1. Resolve mu (atom-indep vector, shape=[n]) and cov (shape=[n, n]).
  //   2. L = lower_cholesky(cov)         — one O(n³) call.
  //   3. Draw N atoms of n standard normals → shape=[N, n].
  //   4. result = mu + L * z (value-ops.mulN + addN)  → shape=[N, n].
  //
  // logTotalmass = 0 (normalized probability measure); n_eff = N.
  const valueOps = require('./value-ops.ts');
  const sampler  = require('./sampler.ts');
  const distIR = d.distIR;
  if (!distIR || !distIR.kwargs || !distIR.kwargs.mu || !distIR.kwargs.cov) {
    return Promise.reject(new Error('MvNormal: requires mu and cov kwargs'));
  }
  const muVal = orchestrator.resolveIRToValue(
    distIR.kwargs.mu, ctx.bindings, ctx.fixedValues);
  const covVal = orchestrator.resolveIRToValue(
    distIR.kwargs.cov, ctx.bindings, ctx.fixedValues);
  if (muVal == null) {
    return Promise.reject(new Error('MvNormal: cannot resolve mu (per-atom params deferred)'));
  }
  if (covVal == null) {
    return Promise.reject(new Error('MvNormal: cannot resolve cov (per-atom params deferred)'));
  }
  const muValue = valueLib.asValue(muVal);
  if (muValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'MvNormal: mu must be a vector, got shape=' + JSON.stringify(muValue.shape)));
  }
  const n = muValue.shape[0];
  const covValue = Array.isArray(covVal) && covVal.length > 0 && Array.isArray(covVal[0])
    ? valueOps._nestedToValue(covVal)
    : valueLib.asValue(covVal);
  if (covValue.shape.length !== 2 || covValue.shape[0] !== n || covValue.shape[1] !== n) {
    return Promise.reject(new Error(
      'MvNormal: cov must be ' + n + 'x' + n + ', got shape='
      + JSON.stringify(covValue.shape)));
  }
  let L;
  try {
    L = sampler._internal.ARITH_OPS.lower_cholesky(covValue);
  } catch (err) {
    return Promise.reject(new Error('MvNormal: ' + (err as any).message));
  }
  const N = ctx.sampleCount;
  const stdNormalIR = {
    kind: 'call', op: 'Normal',
    kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } },
  };
  return ctx.sendWorker({
    type: 'sampleN', ir: stdNormalIR, count: N, repeat: n,
    refArrays: {},
    seed: nameSeed(name, ctx.rootSeed),
  }).then((reply: any) => {
    const z = { shape: [N, n], data: reply.samples };
    const Lz = valueOps.mulN(L, z, N);
    const result = valueOps.addN(muValue, Lz, N);
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// matDirichlet — atom-batched probability-simplex draws
// =====================================================================
//
// Spec §08: Dirichlet(alpha = [a_1, ..., a_K]). Atom is a length-K
// probability vector summing to 1. Per-coord rep: g_k ~ Gamma(a_k, 1)
// independently across coords, then x = g / sum(g) (l1-normalize).
//
// Implementation: K worker calls (one per coord) each drawing N gamma
// variates; combine into atom-major shape=[N, K] + l1-normalize per
// atom.
//
// alpha must be atom-independent (literal vector or fixed-phase
// binding). Per-atom alpha (a vector that varies across N atoms) is
// deferred — it requires per-atom Gamma parameters, which the worker's
// sampleN doesn't broadcast through.
function matDirichlet(name: string, d: DerivationDirichlet, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const alphaIR = (distIR.kwargs && distIR.kwargs.alpha)
    || (distIR.args && distIR.args[0]);
  if (!alphaIR) {
    return Promise.reject(new Error('Dirichlet: requires alpha argument'));
  }
  const alphaVal = orchestrator.resolveIRToValue(alphaIR, ctx.bindings, ctx.fixedValues);
  if (alphaVal == null) {
    return Promise.reject(new Error('Dirichlet: cannot resolve alpha (per-atom alpha deferred)'));
  }
  const alphaValue = valueLib.asValue(alphaVal);
  if (alphaValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'Dirichlet: alpha must be a vector, got shape=' + JSON.stringify(alphaValue.shape)));
  }
  const K = alphaValue.shape[0];
  if (K === 0) {
    return Promise.reject(new Error('Dirichlet: alpha must be non-empty'));
  }
  for (let k = 0; k < K; k++) {
    if (!(alphaValue.data[k] > 0)) {
      return Promise.reject(new Error(
        'Dirichlet: alpha[' + k + '] = ' + alphaValue.data[k] + ' must be positive'));
    }
  }
  const N = ctx.sampleCount;
  const promises: Promise<any>[] = [];
  for (let k = 0; k < K; k++) {
    const gammaIR = {
      kind: 'call', op: 'Gamma',
      kwargs: {
        shape: { kind: 'lit', value: alphaValue.data[k] },
        rate:  { kind: 'lit', value: 1 },
      },
    };
    promises.push(ctx.sendWorker({
      type: 'sampleN', ir: gammaIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|' + k, ctx.rootSeed),
    }));
  }
  return Promise.all(promises).then((replies: any[]) => {
    const data = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        const v = replies[k].samples[i];
        data[i * K + k] = v;
        sum += v;
      }
      const inv = sum > 0 ? 1 / sum : 0;
      for (let k = 0; k < K; k++) data[i * K + k] *= inv;
    }
    const result: any = { shape: [N, K], data };
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// matMultinomial — atom-batched K-vector count draws
// =====================================================================
//
// Spec §08: Multinomial(n, p) — n iid Categorical(p) draws aggregated
// into a length-K count vector. Atom is shape=[K] non-negative integers
// summing to n. n and p must be atom-independent.
function matMultinomial(name: string, d: DerivationMultinomial, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const nIR = (distIR.kwargs && distIR.kwargs.n)
    || (distIR.args && distIR.args[0]);
  const pIR = (distIR.kwargs && distIR.kwargs.p)
    || (distIR.args && distIR.args[1]);
  if (!nIR || !pIR) {
    return Promise.reject(new Error('Multinomial: requires n and p arguments'));
  }
  const nVal = orchestrator.resolveIRToValue(nIR, ctx.bindings, ctx.fixedValues);
  const pVal = orchestrator.resolveIRToValue(pIR, ctx.bindings, ctx.fixedValues);
  if (typeof nVal !== 'number' || !Number.isInteger(nVal) || nVal < 0) {
    return Promise.reject(new Error('Multinomial: n must be a non-negative integer'));
  }
  const pValue = valueLib.asValue(pVal);
  if (pValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'Multinomial: p must be a vector, got shape=' + JSON.stringify(pValue.shape)));
  }
  const K = pValue.shape[0];
  if (K === 0) {
    return Promise.reject(new Error('Multinomial: p must be non-empty'));
  }
  let pSum = 0;
  for (let k = 0; k < K; k++) {
    if (!(pValue.data[k] >= 0)) {
      return Promise.reject(new Error(
        'Multinomial: p[' + k + '] = ' + pValue.data[k] + ' must be non-negative'));
    }
    pSum += pValue.data[k];
  }
  if (!(pSum > 0)) {
    return Promise.reject(new Error('Multinomial: p must sum to > 0'));
  }
  const N = ctx.sampleCount;
  const n = nVal | 0;
  const catIR: any = {
    kind: 'call', op: 'Categorical',
    kwargs: {
      p: {
        kind: 'call', op: 'vector',
        args: Array.from(pValue.data, (v) => ({ kind: 'lit', value: v })),
      },
    },
  };
  if (n === 0) {
    const data = new Float64Array(N * K);
    const result: any = { shape: [N, K], data };
    return Promise.resolve(measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    }));
  }
  return ctx.sendWorker({
    type: 'sampleN', ir: catIR, count: N * n,
    refArrays: {},
    seed: nameSeed(name, ctx.rootSeed),
  }).then((reply: any) => {
    const draws = reply.samples; // Float64Array(N*n), each in 1..K
    const data = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      const base = i * n;
      for (let j = 0; j < n; j++) {
        const cat = (draws[base + j] | 0) - 1;  // 1-indexed → 0-indexed
        if (cat >= 0 && cat < K) data[i * K + cat] += 1;
      }
    }
    const result: any = { shape: [N, K], data };
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// matWishart / matInverseWishart — atom-batched n×n SPD matrix draws
// via Bartlett decomposition
// =====================================================================
//
// Spec §08: Wishart(nu, scale) with scale ∈ SPD(n) and nu > n - 1.
// Bartlett decomposition: A is n×n lower-triangular with
//   A[i,i] ~ sqrt(Chi²_{nu - i})   (for 0-indexed slot i)
//   A[i,j] ~ Normal(0, 1)  for i > j
//   A[i,j] = 0             for i < j
// L = lower_cholesky(scale); W = (L A)(L A)^T.

function matWishart(name: string, d: DerivationWishart, ctx: any): Promise<EmpiricalMeasure> {
  return wishartCore(name, d.distIR, ctx, /*invert=*/false);
}

function matInverseWishart(name: string, d: DerivationInverseWishart, ctx: any): Promise<EmpiricalMeasure> {
  return wishartCore(name, d.distIR, ctx, /*invert=*/true);
}

function wishartCore(name: string, distIR: any, ctx: any, invert: boolean): Promise<EmpiricalMeasure> {
  const valueOps = require('./value-ops.ts');
  const sampler  = require('./sampler.ts');
  const opLabel = invert ? 'InverseWishart' : 'Wishart';
  const nuIR = (distIR.kwargs && distIR.kwargs.nu)
    || (distIR.args && distIR.args[0]);
  const scaleIR = (distIR.kwargs && distIR.kwargs.scale)
    || (distIR.args && distIR.args[1]);
  if (!nuIR || !scaleIR) {
    return Promise.reject(new Error(`${opLabel}: requires nu and scale arguments`));
  }
  const nuVal = orchestrator.resolveIRToValue(nuIR, ctx.bindings, ctx.fixedValues);
  if (typeof nuVal !== 'number') {
    return Promise.reject(new Error(`${opLabel}: nu must be a number`));
  }
  const scaleVal = orchestrator.resolveIRToValue(scaleIR, ctx.bindings, ctx.fixedValues);
  if (scaleVal == null) {
    return Promise.reject(new Error(`${opLabel}: cannot resolve scale (per-atom scale deferred)`));
  }
  const scaleValue = Array.isArray(scaleVal) && scaleVal.length > 0 && Array.isArray(scaleVal[0])
    ? valueOps._nestedToValue(scaleVal)
    : valueLib.asValue(scaleVal);
  if (scaleValue.shape.length !== 2 || scaleValue.shape[0] !== scaleValue.shape[1]) {
    return Promise.reject(new Error(
      `${opLabel}: scale must be a square matrix, got shape=` + JSON.stringify(scaleValue.shape)));
  }
  const n = scaleValue.shape[0];
  if (!(nuVal > n - 1)) {
    return Promise.reject(new Error(
      `${opLabel}: nu (${nuVal}) must be > n - 1 = ${n - 1}`));
  }
  let scaleForBartlett: any;
  let LS: any;
  try {
    if (invert) {
      scaleForBartlett = sampler._internal.ARITH_OPS.inv(scaleValue);
      LS = sampler._internal.ARITH_OPS.lower_cholesky(scaleForBartlett);
    } else {
      scaleForBartlett = scaleValue;
      LS = sampler._internal.ARITH_OPS.lower_cholesky(scaleValue);
    }
  } catch (err) {
    return Promise.reject(new Error(`${opLabel}: ` + (err as any).message));
  }
  const N = ctx.sampleCount;
  const diagPromises: Promise<any>[] = [];
  for (let i = 0; i < n; i++) {
    const gammaIR = {
      kind: 'call', op: 'Gamma',
      kwargs: {
        shape: { kind: 'lit', value: (nuVal - i) / 2 },
        rate:  { kind: 'lit', value: 0.5 },
      },
    };
    diagPromises.push(ctx.sendWorker({
      type: 'sampleN', ir: gammaIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|diag|' + i, ctx.rootSeed),
    }));
  }
  const offDiagPerAtom = n * (n - 1) / 2;
  const offDiagP = offDiagPerAtom > 0 ? ctx.sendWorker({
    type: 'sampleN',
    ir: { kind: 'call', op: 'Normal',
          kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } } },
    count: N * offDiagPerAtom,
    refArrays: {},
    seed: nameSeed(name + '|offdiag', ctx.rootSeed),
  }) : Promise.resolve({ samples: new Float64Array(0) });
  return Promise.all([Promise.all(diagPromises), offDiagP]).then(([diagReplies, offDiagReply]: any[]) => {
    const data = new Float64Array(N * n * n);
    const A = new Float64Array(n * n);
    const T = new Float64Array(n * n);
    const W = new Float64Array(n * n);
    const offDiagData = offDiagReply.samples;
    const LSdata = LS.data;
    for (let atom = 0; atom < N; atom++) {
      for (let i = 0; i < n * n; i++) A[i] = 0;
      for (let i = 0; i < n; i++) {
        A[i * n + i] = Math.sqrt(diagReplies[i].samples[atom]);
      }
      let oIdx = atom * offDiagPerAtom;
      for (let i = 1; i < n; i++) {
        for (let j = 0; j < i; j++) A[i * n + j] = offDiagData[oIdx++];
      }
      // T = LS * A.
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += LSdata[i * n + k] * A[k * n + j];
          T[i * n + j] = s;
        }
      }
      // W = T * T^T.
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          for (let k = 0; k < n; k++) s += T[i * n + k] * T[j * n + k];
          W[i * n + j] = s;
        }
      }
      const wBase = atom * n * n;
      if (invert) {
        const tinv = new Float64Array(n * n);
        for (let j = 0; j < n; j++) {
          for (let i = 0; i < n; i++) {
            if (i < j) { tinv[i * n + j] = 0; continue; }
            let s = (i === j) ? 1 : 0;
            for (let k = j; k < i; k++) s -= T[i * n + k] * tinv[k * n + j];
            tinv[i * n + j] = s / T[i * n + i];
          }
        }
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) {
            let s = 0;
            for (let k = 0; k < n; k++) s += tinv[k * n + i] * tinv[k * n + j];
            data[wBase + i * n + j] = s;
          }
        }
      } else {
        for (let i = 0; i < n; i++) {
          for (let j = 0; j < n; j++) data[wBase + i * n + j] = W[i * n + j];
        }
      }
    }
    const result: any = { shape: [N, n, n], data };
    return measureFromValue(result, {
      logWeights: null,
      logTotalmass: 0,
      n_eff: N,
    });
  });
}

// =====================================================================
// matLKJCholesky — atom-batched n×n lower-triangular Cholesky factor
// =====================================================================
//
// Spec §08: LKJCholesky(n, eta) — onion-procedure sampler.
function matLKJCholesky(name: string, d: DerivationLKJCholesky, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const nIR   = (distIR.kwargs && distIR.kwargs.n)   || (distIR.args && distIR.args[0]);
  const etaIR = (distIR.kwargs && distIR.kwargs.eta) || (distIR.args && distIR.args[1]);
  if (!nIR || !etaIR) {
    return Promise.reject(new Error('LKJCholesky: requires n and eta arguments'));
  }
  const nVal   = orchestrator.resolveIRToValue(nIR,   ctx.bindings, ctx.fixedValues);
  const etaVal = orchestrator.resolveIRToValue(etaIR, ctx.bindings, ctx.fixedValues);
  if (typeof nVal !== 'number' || !Number.isInteger(nVal) || nVal < 1) {
    return Promise.reject(new Error('LKJCholesky: n must be a positive integer'));
  }
  if (typeof etaVal !== 'number' || !(etaVal > 0)) {
    return Promise.reject(new Error('LKJCholesky: eta must be a positive number'));
  }
  const n   = nVal | 0;
  const eta = etaVal;
  const N   = ctx.sampleCount;
  if (n === 1) {
    const data = new Float64Array(N);
    for (let i = 0; i < N; i++) data[i] = 1;
    return Promise.resolve(measureFromValue(
      { shape: [N, 1, 1], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N }));
  }
  const betaPromises: Promise<any>[] = [];
  for (let i = 1; i < n; i++) {
    const a = eta + (n - i - 1) / 2;
    const b = i / 2;
    const betaIR = {
      kind: 'call', op: 'Beta',
      kwargs: { alpha: { kind: 'lit', value: a },
                beta:  { kind: 'lit', value: b } },
    };
    betaPromises.push(ctx.sendWorker({
      type: 'sampleN', ir: betaIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|beta|' + i, ctx.rootSeed),
    }));
  }
  const totalNormals = N * (n * (n - 1) / 2);
  const normalP = ctx.sendWorker({
    type: 'sampleN',
    ir: { kind: 'call', op: 'Normal',
          kwargs: { mu: { kind: 'lit', value: 0 }, sigma: { kind: 'lit', value: 1 } } },
    count: totalNormals,
    refArrays: {},
    seed: nameSeed(name + '|norm', ctx.rootSeed),
  });
  return Promise.all([Promise.all(betaPromises), normalP]).then(([betaReplies, normalReply]: any[]) => {
    const data = new Float64Array(N * n * n);
    const normals = normalReply.samples;
    let nIdx = 0;
    for (let atom = 0; atom < N; atom++) {
      const base = atom * n * n;
      data[base] = 1;
      for (let i = 1; i < n; i++) {
        const Bi = betaReplies[i - 1].samples[atom];
        const ri = Math.sqrt(Bi);
        const diag = Math.sqrt(1 - Bi);
        let zsq = 0;
        const z = new Float64Array(i);
        for (let k = 0; k < i; k++) {
          const v = normals[nIdx++];
          z[k] = v;
          zsq += v * v;
        }
        const invNorm = zsq > 0 ? 1 / Math.sqrt(zsq) : 0;
        for (let k = 0; k < i; k++) {
          data[base + i * n + k] = ri * z[k] * invNorm;
        }
        data[base + i * n + i] = diag;
      }
    }
    return measureFromValue(
      { shape: [N, n, n], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// =====================================================================
// matLKJ — atom-batched n×n correlation matrix (LKJCholesky · ^T)
// =====================================================================
function matLKJ(name: string, d: DerivationLKJ, ctx: any): Promise<EmpiricalMeasure> {
  const choleskyD: DerivationLKJCholesky = { kind: 'lkjcholesky', distIR: d.distIR };
  return matLKJCholesky(name, choleskyD, ctx).then((Lmeasure: any) => {
    const dims = Lmeasure.dims;
    if (!dims || dims.length !== 2 || dims[0] !== dims[1]) {
      return Promise.reject(new Error('LKJ: unexpected LKJCholesky output shape'));
    }
    const n = dims[0];
    const N = Lmeasure.samples.length / (n * n);
    const data = new Float64Array(N * n * n);
    const L = Lmeasure.samples;
    for (let atom = 0; atom < N; atom++) {
      const base = atom * n * n;
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          let s = 0;
          const maxK = Math.min(i, j);
          for (let k = 0; k <= maxK; k++) s += L[base + i * n + k] * L[base + j * n + k];
          data[base + i * n + j] = s;
        }
      }
    }
    return measureFromValue(
      { shape: [N, n, n], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

// =====================================================================
// matBinnedPoissonProcess — atom-batched K-vector of independent
// Poisson counts
// =====================================================================
function matBinnedPoissonProcess(name: string, d: DerivationBinnedPoissonProcess, ctx: any): Promise<EmpiricalMeasure> {
  const distIR = d.distIR as any;
  const ratesIR = (distIR.kwargs && distIR.kwargs.rates)
    || (distIR.args && distIR.args[0]);
  if (!ratesIR) {
    return Promise.reject(new Error(
      'BinnedPoissonProcess: requires rates argument (per-bin Poisson means)'));
  }
  const ratesVal = orchestrator.resolveIRToValue(ratesIR, ctx.bindings, ctx.fixedValues);
  if (ratesVal == null) {
    return Promise.reject(new Error('BinnedPoissonProcess: cannot resolve rates (per-atom rates deferred)'));
  }
  const ratesValue = valueLib.asValue(ratesVal);
  if (ratesValue.shape.length !== 1) {
    return Promise.reject(new Error(
      'BinnedPoissonProcess: rates must be a vector, got shape=' + JSON.stringify(ratesValue.shape)));
  }
  const K = ratesValue.shape[0];
  if (K === 0) {
    return Promise.reject(new Error('BinnedPoissonProcess: rates must be non-empty'));
  }
  for (let k = 0; k < K; k++) {
    if (!(ratesValue.data[k] >= 0)) {
      return Promise.reject(new Error(
        'BinnedPoissonProcess: rates[' + k + '] = ' + ratesValue.data[k] + ' must be non-negative'));
    }
  }
  const N = ctx.sampleCount;
  const promises: Promise<any>[] = [];
  for (let k = 0; k < K; k++) {
    const rk = ratesValue.data[k];
    if (rk === 0) {
      // Degenerate Poisson(0) — all atoms are 0 a.s. Short-circuit
      // because stdlib's Poisson constructor disallows rate == 0.
      promises.push(Promise.resolve({ samples: new Float64Array(N) }));
      continue;
    }
    const poissonIR = {
      kind: 'call', op: 'Poisson',
      kwargs: { rate: { kind: 'lit', value: rk } },
    };
    promises.push(ctx.sendWorker({
      type: 'sampleN', ir: poissonIR, count: N,
      refArrays: {},
      seed: nameSeed(name + '|bin|' + k, ctx.rootSeed),
    }));
  }
  return Promise.all(promises).then((replies: any[]) => {
    const data = new Float64Array(N * K);
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < K; k++) data[i * K + k] = replies[k].samples[i];
    }
    return measureFromValue(
      { shape: [N, K], data } as any,
      { logWeights: null, logTotalmass: 0, n_eff: N });
  });
}

module.exports = {
  matMvNormal,
  matDirichlet,
  matMultinomial,
  matWishart,
  matInverseWishart,
  matLKJCholesky,
  matLKJ,
  matBinnedPoissonProcess,
};
