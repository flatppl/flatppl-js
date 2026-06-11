'use strict';

// Sampling ≡ density agreement harness (no test() calls — imported by
// sampling-density-agreement.test.ts and usable from scratch probes).
// See flatppl-dev/measure-lowering-unification-plan.md (Phase 0).
//
// For a CLOSED scalar measure M: compares logdensityof(M, x) against the
// empirical log-density of the SAMPLE histogram of M, as a RATIO between probe
// points (cancels the normalisation constant AND the bin width — so it works
// for unnormalised measures, but is BLIND to a pure normalisation-constant bug
// like M3; those need a totalmass check, not this).

const ENG = '../';
const { processSource, orchestrator, materialiser } = require(ENG + 'index.ts');
const der = require(ENG + 'derivations.ts');
const { createWorkerHandler } = require(ENG + 'worker.ts');

function buildCtx(src: string, N: number, seed: number, mc?: number) {
  const built = orchestrator.buildDerivations(processSource(src).bindings);
  for (const [nm, b] of built.bindings) {
    if (!built.derivations[nm]) {
      try { const c = der.classifyDerivation(b, built.bindings, built.fixedValues); if (c) built.derivations[nm] = c; }
      catch (_) { /* leave unclassified */ }
    }
  }
  const w = createWorkerHandler();
  w.handle({ type: 'init', seed });
  const cache = new Map();
  const ctx: any = {
    derivations: built.derivations, bindings: built.bindings,
    fixedValues: built.fixedValues || new Map(), sampleCount: N, rootKey: seed,
    rootSeed: seed, marginalizationCount: mc || 200,
    getMeasure: (n: string) => {
      if (cache.has(n)) return cache.get(n);
      const p = materialiser.materialiseMeasure(n, ctx); cache.set(n, p); return p;
    },
    sendWorker: (m: any) => Promise.resolve(w.handle(m)),
  };
  return { built, ctx };
}

function scalarColumn(m: any): { x: Float64Array; lw: Float64Array | null } {
  let x = m.samples;
  if (!x && m.value && m.value.data) x = m.value.data;
  if (!x && m.fields) {
    const k = Object.keys(m.fields)[0];
    x = m.fields[k].samples || (m.fields[k].value && m.fields[k].value.data);
  }
  return { x, lw: m.logWeights || null };
}

function makeHist(x: Float64Array, lw: Float64Array | null, nbins: number) {
  const N = x.length;
  const w = new Float64Array(N);
  if (lw) { let mx = -Infinity; for (let i = 0; i < N; i++) if (lw[i] > mx) mx = lw[i];
    for (let i = 0; i < N; i++) w[i] = Math.exp(lw[i] - mx); }
  else w.fill(1);
  const idx = Array.from({ length: N }, (_, i) => i).sort((a, b) => x[a] - x[b]);
  let tot = 0; for (let i = 0; i < N; i++) tot += w[i];
  const pct = (p: number) => { let acc = 0; for (const i of idx) { acc += w[i]; if (acc >= p * tot) return x[i]; } return x[idx[N - 1]]; };
  const lo = pct(0.02), hi = pct(0.98);
  const bw = (hi - lo) / nbins || 1;
  const mass = new Float64Array(nbins);
  for (let i = 0; i < N; i++) { const b = Math.floor((x[i] - lo) / bw); if (b >= 0 && b < nbins) mass[b] += w[i]; }
  return {
    lo, hi, bw, mass, tot,
    logDensityAt(v: number): number {
      const b = Math.floor((v - lo) / bw);
      if (b < 0 || b >= nbins || mass[b] <= 0) return -Infinity;
      return Math.log(mass[b] / (tot * bw));
    },
    // The modal bin centre (the low-noise reference probe).
    modalCentre(): number {
      let bi = 0; for (let b = 1; b < nbins; b++) if (mass[b] > mass[bi]) bi = b;
      return lo + (bi + 0.5) * bw;
    },
    // Probe centres SPREAD across the support (not clustered at the mode), so a
    // variance/shape divergence — e.g. a density that uses the narrow
    // conditional Normal(·,1) where the marginal is the wider Normal(0,√2)
    // (H8) — is actually exercised. Keep only bins with adequate count
    // (≥ minFrac of the modal mass, so histogram noise stays bounded), then
    // sample them evenly by POSITION including the shoulders.
    probeCentres(k: number, minFrac = 0.12): number[] {
      let mx = 0; for (let b = 0; b < nbins; b++) if (mass[b] > mx) mx = mass[b];
      const ok: number[] = [];
      for (let b = 0; b < nbins; b++) if (mass[b] >= minFrac * mx) ok.push(b);
      if (ok.length <= k) return ok.map((b) => lo + (b + 0.5) * bw);
      const out: number[] = [];
      for (let j = 0; j < k; j++) out.push(ok[Math.round(j * (ok.length - 1) / (k - 1))]);
      return Array.from(new Set(out)).map((b) => lo + (b + 0.5) * bw);
    },
  };
}

async function densityAt(ctx: any, measureName: string, probe: number): Promise<number> {
  const key = '__ld$' + measureName + '$' + probe;
  ctx.derivations[key] = { kind: 'logdensityof', measureName, obsIR: { kind: 'lit', value: probe } };
  const m = await materialiser.materialiseMeasure(key, ctx);
  const s = m.samples || (m.value && m.value.data);
  return s ? s[0] : NaN;
}

async function agreement(src: string, measureName: string, opts: any = {}) {
  const N = opts.N || 40000;
  const tol = opts.tol || 0.35;
  const nbins = opts.nbins || 40;
  let sampleM: any;
  try {
    const { ctx } = buildCtx(src, N, opts.seed || 1234, opts.mc);
    sampleM = await ctx.getMeasure(measureName);
  } catch (e: any) { return { ok: false, crashed: 'sample', reason: 'SAMPLE threw: ' + e.message }; }

  const { x, lw } = scalarColumn(sampleM);
  if (!x || x.length === 0) return { ok: false, crashed: 'sample', reason: 'no scalar sample column' };
  const hist = makeHist(x, lw, nbins);
  const ref = hist.modalCentre();
  const centres = [ref].concat(hist.probeCentres(6).filter((c: number) => c !== ref));
  if (centres.length < 3) return { ok: false, crashed: 'sample', reason: 'sample too concentrated for a histogram' };

  let dCtx: any;
  try { dCtx = buildCtx(src, N, (opts.seed || 1234) + 7, opts.mc).ctx; }
  catch (e: any) { return { ok: false, crashed: 'density', reason: 'density-ctx build threw: ' + e.message }; }

  let dRef: number;
  try { dRef = await densityAt(dCtx, measureName, ref); }
  catch (e: any) { return { ok: false, crashed: 'density', reason: 'DENSITY threw: ' + e.message }; }
  if (!Number.isFinite(dRef)) return { ok: false, crashed: 'density', reason: `density at modal probe ${ref.toFixed(3)} = ${dRef}` };
  const hRef = hist.logDensityAt(ref);

  const probes: any[] = [];
  let maxErr = 0;
  for (let i = 1; i < centres.length; i++) {
    const p = centres[i];
    const hP = hist.logDensityAt(p);
    if (!Number.isFinite(hP)) continue;
    let dP: number;
    try { dP = await densityAt(dCtx, measureName, p); }
    catch (e: any) { return { ok: false, crashed: 'density', reason: 'DENSITY threw at probe: ' + e.message }; }
    if (!Number.isFinite(dP)) return { ok: false, crashed: 'density', reason: `density at probe ${p.toFixed(3)} = ${dP} (sample has mass there)` };
    const err = Math.abs((dP - dRef) - (hP - hRef));
    probes.push({ p: +p.toFixed(3), dRatio: +(dP - dRef).toFixed(3), hRatio: +(hP - hRef).toFixed(3), err: +err.toFixed(3) });
    if (err > maxErr) maxErr = err;
  }
  if (probes.length === 0) return { ok: false, crashed: 'density', reason: 'no comparable probe bins' };
  return { ok: maxErr < tol, maxErr, probes };
}

// Correlation of two record fields of a sampled measure (for H7 / reused-factor
// independence-vs-coupling — the scalar-density harness can't see correlation).
async function fieldCorrelation(src: string, measureName: string, fA: string, fB: string, opts: any = {}) {
  const N = opts.N || 40000;
  const { ctx } = buildCtx(src, N, opts.seed || 99);
  const m = await ctx.getMeasure(measureName);
  if (!m.fields || !m.fields[fA] || !m.fields[fB]) throw new Error('fields missing: ' + Object.keys(m.fields || {}));
  const a = m.fields[fA].samples, b = m.fields[fB].samples;
  const n = Math.min(a.length, b.length);
  let ma = 0, mb = 0; for (let i = 0; i < n; i++) { ma += a[i]; mb += b[i]; } ma /= n; mb /= n;
  let cov = 0, va = 0, vb = 0;
  for (let i = 0; i < n; i++) { cov += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; }
  return cov / Math.sqrt(va * vb);
}

module.exports = { buildCtx, scalarColumn, makeHist, densityAt, agreement, fieldCorrelation };
