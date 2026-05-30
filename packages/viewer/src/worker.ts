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

/**
 * Spawn a sampler Worker from a URL, with a blob-URL fallback for
 * CSP-restricted hosts (notably the VS Code webview, where some
 * versions reject `new Worker(webview-uri)` outright). The fallback
 * fetches the bundle text via the same URI (which connect-src does
 * allow), wraps it in a same-origin blob, and constructs the Worker
 * from that.
 *
 * Exported so per-host bootstraps (web's main app entry,
 * vscode-extension's webview script) can pass the same spawn
 * function into the engine Backend's `spawnWorker` slot — the
 * pool then uses one consistent spawning path for every Worker it
 * creates, instead of each host re-implementing the CSP dance.
 *
 * Synchronous-on-the-happy-path: `new Worker(url)` returns
 * immediately on hosts where it works, so the parallel-pool's
 * spawnSlots() loop doesn't have to be async. The fallback path
 * uses synchronous XHR (the only browser primitive that fetches a
 * URL string -> bytes without going through a Promise) so the same
 * sync contract holds for CSP-restricted hosts too. The 5-second
 * blob-URL revoke timer protects against URL leaks once the worker
 * has finished parsing its source; the worker keeps running.
 */
export function spawnSamplerWorker(url: string): Worker {
  try {
    return new Worker(url);
  } catch (e) {
    console.warn('FlatPPL: direct worker spawn failed, retrying via blob URL:',
      e instanceof Error ? e.message : e);
    // CSP-restricted host (VS Code webview, some Firefox configs).
    // Synchronously fetch + blob the bundle so this stays a plain
    // (non-Promise) spawn — matches the engine Backend.spawnWorker
    // contract used by the parallel worker pool.
    const req = new XMLHttpRequest();
    req.open('GET', url, /* async */ false);
    req.send();
    if (req.status !== 0 && (req.status < 200 || req.status >= 300)) {
      throw new Error('failed to fetch worker bundle: ' + req.status + ' ' + req.statusText);
    }
    const blob = new Blob([req.responseText], { type: 'application/javascript' });
    const blobUrl = URL.createObjectURL(blob);
    const w = new Worker(blobUrl);
    setTimeout(function() { try { URL.revokeObjectURL(blobUrl); } catch (_) {} }, 5000);
    return w;
  }
}

export function ensureSamplerWorker(ctx: Ctx) {
  if (ctx.samplerWorker) return Promise.resolve(ctx.samplerWorker);
  if (ctx.samplerWorkerPromise) return ctx.samplerWorkerPromise;

  ctx.samplerWorkerPromise = (async function() {
    // Route through the shared helper so the CSP / blob-URL fallback
    // logic lives in exactly one place — see spawnSamplerWorker above.
    const w: Worker = spawnSamplerWorker(ctx.SAMPLER_WORKER_URL);
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
  const entries = ctx.pendingRequests.values();
  ctx.pendingRequests = new Map();
  for (const entry of entries) {
    try { entry.reject(new Error('cancelled')); } catch (_) {}
  }
  // Tear down the parallel worker pool too — Stop button / page-
  // unload should free every pool worker, not just the single-thread
  // sampler held in ctx. disposePool is a no-op when the pool was
  // never spawned, so this is safe to call unconditionally. Reached
  // via the FlatPPLEngine global so the viewer bundle doesn't pull
  // worker-pool into its own dependency graph (the engine bundle
  // owns it).
  const FE: any = (typeof window !== 'undefined') ? (window as any).FlatPPLEngine : null;
  if (FE && FE.workerPool && typeof FE.workerPool.disposePool === 'function') {
    try { FE.workerPool.disposePool(); } catch (_) {}
  }
}
