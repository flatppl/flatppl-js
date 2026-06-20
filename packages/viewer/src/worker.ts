// @ts-check
// @flatppl/viewer — sampler-worker subsystem —
//
// Spawns the Web Worker from ctx.SAMPLER_WORKER_URL on first use,
// multiplexes requests by id through ctx.pendingRequests, and
// surfaces a Promise-based sendWorker API the engine-facade and
// renderers use. cancelAllSampling tears the worker down and
// rejects every in-flight request — used by the Stop button and
// when SAMPLE_COUNT / REJECTION_BUDGET change.

/**
 * Asynchronously spawn the sampler worker, caching the result.
 * Returns a Promise<Worker> so callers can await spawn completion.
 *
 * Why blob-URL? VS Code webviews are sandboxed iframes whose CSP and
 * cross-origin posture refuses 'new Worker(webview-uri)' in some
 * VS Code versions. The reliable workaround is to fetch the bundle
 * text via the webview URI (which IS allowed by connect-src) and
 * spawn the worker from a same-origin blob: URL. This pattern is
 * also documented in the official VS Code webview samples.
 */
import type { Ctx } from './types';

export function ensureSamplerWorker(ctx: Ctx) {
  if (ctx.samplerWorker) return Promise.resolve(ctx.samplerWorker);
  if (ctx.samplerWorkerPromise) return ctx.samplerWorkerPromise;

  ctx.samplerWorkerPromise = (async function() {
    // Try direct construction first — cheapest path on hosts where
    // it works. Fall back to blob: on any failure (security error,
    // cross-origin block, etc.). Async fetch (not sync XHR — sync
    // XHR is blocked in the VS Code webview's Chromium for
    // `vscode-webview:` URIs).
    let w: Worker | null = null;
    try {
      w = new Worker(ctx.SAMPLER_WORKER_URL);
    } catch (e) {
      console.warn('FlatPPL: direct worker spawn failed, retrying via blob URL:',
        e instanceof Error ? e.message : e);
    }
    if (!w) {
      const resp = await fetch(ctx.SAMPLER_WORKER_URL);
      if (!resp.ok) throw new Error('failed to fetch worker bundle: '
        + resp.status + ' ' + resp.statusText);
      const src = await resp.text();
      const blob = new Blob([src], { type: 'application/javascript' });
      const url = URL.createObjectURL(blob);
      w = new Worker(url);
      // The blob URL only needs to live until the Worker has parsed
      // its source — revoke after a short delay so the URL isn't
      // leaked (the worker keeps running independently).
      setTimeout(function() { try { URL.revokeObjectURL(url); } catch (_) {} }, 5000);
    }
    wireWorker(ctx, w);
    ctx.samplerWorker = w;
    // Initialize with a fixed seed for deterministic output. Future:
    // plumb a "Resample" button that re-seeds (e.g. from Date.now()).
    sendWorkerNow(ctx, w, { type: 'init', seed: 1 });
    return w;
  })();

  ctx.samplerWorkerPromise.catch(function(err: any) {
    ctx.samplerWorkerError = err;
    ctx.samplerWorkerPromise = null;
    console.error('FlatPPL: sampler worker unavailable:', err);
  });

  return ctx.samplerWorkerPromise;
}

export function wireWorker(ctx: Ctx, w: any) {
  w.addEventListener('message', function(ev: any) {
    const reply = ev.data;
    if (!reply || reply.id == null) return;
    const p = ctx.pendingRequests.get(reply.id);
    if (!p) return;
    // Progress messages stream mid-run: notify, but keep the request pending
    // (the final mcmcResult/error reply is what resolves it).
    if (reply.type === 'mcmcProgress') { if (p.onProgress) p.onProgress(reply); return; }
    ctx.pendingRequests.delete(reply.id);
    if (reply.type === 'error') p.reject(new Error(reply.message || 'worker error'));
    else p.resolve(reply);
  });
  w.addEventListener('error', function(e: any) {
    // A top-level worker error fails every outstanding request — there's
    // no way to know which request the error pertains to, and the worker
    // may be dead. Reject all and reset so a future request can retry
    // the spawn.
    console.error('FlatPPL sampler worker error:', e.message || e);
    for (const entry of ctx.pendingRequests.values()) entry.reject(new Error(e.message || 'worker crashed'));
    ctx.pendingRequests.clear();
    try { w.terminate(); } catch (_) {}
    if (ctx.samplerWorker === w) {
      ctx.samplerWorker = null;
      ctx.samplerWorkerPromise = null;
    }
  });
}

export function sendWorkerNow(ctx: Ctx, w: any, msg: any) {
  const id = ++ctx.samplerReqId;
  w.postMessage(Object.assign({ id: id }, msg));
}

export function sendWorker(ctx: Ctx, msg: any) {
  return ensureSamplerWorker(ctx).then(function(w: any) {
    const id = ++ctx.samplerReqId;
    const wrapped = Object.assign({ id: id }, msg);
    return new Promise(function(resolve, reject) {
      ctx.pendingRequests.set(id, { resolve: resolve, reject: reject });
      w.postMessage(wrapped);
    });
  });
}

/**
 * Cancel any in-flight sample requests by terminating the worker
 * and rejecting every pending promise. The main-thread sample
 * cache is preserved, so bindings that finished before the cancel
 * stay available — only the in-flight request is dropped.
 *
 * The next sendWorker() call will lazily re-spawn the worker via
 * ensureSamplerWorker(). Cheap enough that we don't bother
 * keeping a "warm" worker around.
 */
export function cancelAllSampling(ctx: Ctx) {
  if (ctx.samplerWorker) {
    try { ctx.samplerWorker.terminate(); } catch (_) {}
    ctx.samplerWorker = null;
    ctx.samplerWorkerPromise = null;
  }
  for (const w of (ctx.mcmcPool || [])) { try { w.terminate(); } catch (_) {} }
  ctx.mcmcPool = [];
  const entries = ctx.pendingRequests.values();
  ctx.pendingRequests = new Map();
  for (const entry of entries) {
    try { entry.reject(new Error('cancelled')); } catch (_) {}
  }
}

// Spawn ONE fresh worker (direct, blob fallback), wired to ctx.pendingRequests
// and initialised. Used to build the MCMC pool — each worker runs an
// independent share of chains/ensembles off the main thread.
async function spawnWorker(ctx: Ctx, seed: number): Promise<any> {
  let w: any = null;
  try { w = new Worker(ctx.SAMPLER_WORKER_URL); } catch (_) { /* blob fallback */ }
  if (!w) {
    const resp = await fetch(ctx.SAMPLER_WORKER_URL);
    if (!resp.ok) throw new Error('failed to fetch worker bundle: ' + resp.status);
    const url = URL.createObjectURL(new Blob([await resp.text()], { type: 'application/javascript' }));
    w = new Worker(url);
    setTimeout(function () { try { URL.revokeObjectURL(url); } catch (_) {} }, 5000);
  }
  wireWorker(ctx, w);
  sendWorkerNow(ctx, w, { type: 'init', seed });
  return w;
}

function sendTo(ctx: Ctx, w: any, msg: any, onProgress?: (m: any) => void): Promise<any> {
  const id = ++ctx.samplerReqId;
  return new Promise(function (resolve, reject) {
    ctx.pendingRequests.set(id, { resolve, reject, onProgress });
    w.postMessage(Object.assign({ id }, msg));
  });
}

// Concatenate per-worker posterior measures (each a record of scalar/array
// field measures) into one, and combine diagnostics: acceptRate averaged,
// per-parameter R̂ taken as the worst (conservative) across the independent
// runs, bulk-ESS summed (ESS is additive across independent chains/ensembles).
function poolMeasures(measures: any[]): any {
  if (measures.length === 1) return measures[0];
  const first = measures[0];
  const cat = (key: string, arrs: Float64Array[]) => {
    let n = 0; for (const a of arrs) n += a.length;
    const out = new Float64Array(n); let o = 0;
    for (const a of arrs) { out.set(a, o); o += a.length; }
    return out;
  };
  const fields: any = {};
  for (const fn of Object.keys(first.fields)) {
    const parts = measures.map((m) => m.fields[fn]);
    const samples = cat(fn, parts.map((p: any) => p.samples));
    if (parts[0].shape === 'array') {
      const dims = parts[0].dims;
      const stride = dims.reduce((a: number, b: number) => a * b, 1);
      fields[fn] = { shape: 'array', dims, samples, logWeights: null, logTotalmass: 0,
        n_eff: samples.length / stride, value: { shape: [samples.length / stride].concat(dims), data: samples } };
    } else {
      fields[fn] = { samples, logWeights: null, logTotalmass: 0, n_eff: samples.length,
        value: { shape: [samples.length], data: samples } };
    }
  }
  // Combine diagnostics.
  let accSum = 0; const perParam: any = {};
  for (const m of measures) {
    const dg = m.diagnostics || {};
    accSum += (dg.acceptRate || 0);
    for (const k of Object.keys(dg.perParam || {})) {
      const e = dg.perParam[k];
      if (!perParam[k]) perParam[k] = { rHat: 0, essBulk: 0 };
      perParam[k].rHat = Math.max(perParam[k].rHat, e.rHat || 0);
      perParam[k].essBulk += (e.essBulk || 0);
    }
  }
  const totalN = fields[Object.keys(fields)[0]].samples.length;
  // Elliptical slice: chains pooled like MH (R̂ worst, ESS summed via perParam
  // above), but keep the ess-slice tag + mode + mean shrinks for the readout.
  if (first.diagnostics && first.diagnostics.method === 'ess-slice') {
    let shr = 0, k = 0; for (const m of measures) { const d = m.diagnostics || {}; if (Number.isFinite(d.meanShrinks)) { shr += d.meanShrinks; k++; } }
    return { shape: 'record', fields, logWeights: null, logTotalmass: 0, n_eff: totalN,
      diagnostics: { method: 'ess-slice', mode: first.diagnostics.mode, meanShrinks: k > 0 ? shr / k : NaN, perParam } };
  }
  // AMIS is an importance sampler, not MCMC: report combined IS quality
  // (effective sample sizes add across independent runs) and the worst auto-K,
  // not an accept rate. essBulk per param already summed above.
  if (first.diagnostics && first.diagnostics.method === 'smc') {
    // Independent SMC runs each give an unbiased estimate Ẑ_r of the evidence;
    // the pooled unbiased estimate is their ARITHMETIC mean, reported as its log
    // (= log-sum-exp(logZ_r) − log k). Worst-case rung count, mean acceptance.
    let mx = -Infinity; for (const m of measures) { const z = (m.diagnostics || {}).logZ; if (Number.isFinite(z) && z > mx) mx = z; }
    let s = 0, k = 0, accS = 0, rMax = 0;
    for (const m of measures) { const d = m.diagnostics || {}; if (Number.isFinite(d.logZ)) { s += Math.exp(d.logZ - mx); k++; } accS += (d.acceptRate || 0); rMax = Math.max(rMax, d.rungs || 0); }
    const logZ = k > 0 ? mx + Math.log(s / k) : NaN;
    return { shape: 'record', fields, logWeights: null, logTotalmass: 0, n_eff: totalN,
      diagnostics: { method: 'smc', logZ, rungs: rMax, acceptRate: accS / measures.length, nSamples: totalN } };
  }
  if (first.diagnostics && first.diagnostics.method === 'amis') {
    let essSum = 0, kMax = 0, nSum = 0;
    for (const m of measures) {
      const dg = m.diagnostics || {};
      essSum += (dg.ess || 0); kMax = Math.max(kMax, dg.K || 0); nSum += (dg.nSamples || 0);
    }
    return { shape: 'record', fields, logWeights: null, logTotalmass: 0, n_eff: totalN,
      diagnostics: { method: 'amis', ess: essSum, essFrac: nSum > 0 ? essSum / nSum : 0, K: kMax, nSamples: nSum, perParam } };
  }
  return { shape: 'record', fields, logWeights: null, logTotalmass: 0, n_eff: totalN,
    diagnostics: { acceptRate: accSum / measures.length, perParam } };
}

// Run an MCMC posterior off the main thread, parallelised across a worker pool.
// `name` is the bayesupdate binding; `opts` is ctx.inferenceOpts. MH partitions
// its independent chains across workers; emcee runs one independent ensemble
// per worker (P× the draws in P× less wall-clock). Each worker gets a distinct
// seed and produces sampleCount/P atoms; the results are pooled.
export async function runMcmcPool(ctx: Ctx, name: string, opts: any): Promise<any> {
  const source = ctx.currentSource;
  const sampleCount = ctx.SAMPLE_COUNT;
  const baseSeed = (opts.seed != null ? opts.seed : (ctx.rootSeed || 1)) | 0;
  const hw = (typeof navigator !== 'undefined' && (navigator as any).hardwareConcurrency) || 4;
  const cap = Math.max(1, Math.min(opts.parallel || hw, 8));

  // Partition: shares = per-worker { inferenceOpts, sampleCount }.
  // emcee and AMIS run one independent instance per worker (a distinct seed);
  // their per-worker results are concatenated, which lowers the combined
  // estimator's variance — independent ensembles / IS runs add effective
  // sample size. MH partitions its independent chains across workers instead.
  // NOTE: each worker runs a FULL independent instance at the user's knob
  // settings (emcee walkers / amisSamples,amisIters / smcParticles) — these are
  // per-worker, deliberately NOT divided by P. The runs are pooled for variance
  // reduction (more independent samples), so total compute is ≈ P× the knob but
  // wall-clock stays parallel and each run is full quality. Only `sampleCount`
  // (the final atom budget) is split, since the pool concatenates to it.
  const shares: any[] = [];
  if (opts.backend === 'emcee' || opts.backend === 'amis' || opts.backend === 'smc') {
    const P = cap;
    for (let i = 0; i < P; i++) {
      shares.push({ inferenceOpts: Object.assign({}, opts, { seed: baseSeed + i * 7919 }),
        sampleCount: Math.ceil(sampleCount / P) });
    }
  } else {
    const chains = opts.chains || 4;
    const P = Math.max(1, Math.min(cap, chains));
    const per = Math.ceil(chains / P);
    for (let i = 0; i < P; i++) {
      shares.push({ inferenceOpts: Object.assign({}, opts, { chains: per, seed: baseSeed + i * 7919 }),
        sampleCount: Math.ceil(sampleCount / P) });
    }
  }

  const workers = await Promise.all(shares.map((_, i) => spawnWorker(ctx, baseSeed + i * 7919)));
  ctx.mcmcPool = workers;
  // Per-worker progress fractions, averaged into one bar. Each worker carries an
  // equal share of the work, so the mean fraction tracks overall completion.
  const fracs = new Array(shares.length).fill(0);
  let lastPhase = 'warmup';
  const report = (i: number, m: any) => {
    fracs[i] = m.frac || 0; lastPhase = m.phase || lastPhase;
    if (ctx.onSamplingProgress) {
      let s = 0; for (const f of fracs) s += f;
      ctx.onSamplingProgress(s / fracs.length, lastPhase);
    }
  };
  try {
    const replies = await Promise.all(shares.map((s, i) =>
      sendTo(ctx, workers[i], {
        type: 'mcmcRun', source, name,
        inferenceOpts: s.inferenceOpts, sampleCount: s.sampleCount, seed: s.inferenceOpts.seed,
      }, (m: any) => report(i, m))));
    return poolMeasures(replies.map((r: any) => r.measure));
  } finally {
    for (const w of workers) { try { w.terminate(); } catch (_) {} }
    // Only clear the shared pool if it still points at OUR workers — a run that
    // was cancelled while a later run already started must not wipe the live
    // pool reference (which would leak the new run's workers).
    if (ctx.mcmcPool === workers) ctx.mcmcPool = [];
  }
}
