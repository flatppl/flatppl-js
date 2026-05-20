// @ts-check
// @flatppl/viewer — sampler-worker subsystem (Phase 4c).
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
export function ensureSamplerWorker(ctx) {
  if (ctx.samplerWorker) return Promise.resolve(ctx.samplerWorker);
  if (ctx.samplerWorkerPromise) return ctx.samplerWorkerPromise;

  ctx.samplerWorkerPromise = (async function() {
    // Try direct construction first — cheapest path on hosts where
    // it works. Fall back to blob: on any failure (security error,
    // cross-origin block, etc.).
    let w: Worker | null = null;
    try {
      w = new Worker(ctx.SAMPLER_WORKER_URL);
    } catch (e) {
      // continue to blob fallback
      console.warn('FlatPPL: direct worker spawn failed, retrying via blob URL:',
        e instanceof Error ? e.message : e);
    }
    if (!w) {
      var resp = await fetch(ctx.SAMPLER_WORKER_URL);
      if (!resp.ok) throw new Error('failed to fetch worker bundle: ' + resp.status + ' ' + resp.statusText);
      var src = await resp.text();
      var blob = new Blob([src], { type: 'application/javascript' });
      var url = URL.createObjectURL(blob);
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

  ctx.samplerWorkerPromise.catch(function(err) {
    ctx.samplerWorkerError = err;
    ctx.samplerWorkerPromise = null;
    console.error('FlatPPL: sampler worker unavailable:', err);
  });

  return ctx.samplerWorkerPromise;
}

export function wireWorker(ctx, w) {
  w.addEventListener('message', function(ev) {
    var reply = ev.data;
    if (!reply || reply.id == null) return;
    var p = ctx.pendingRequests.get(reply.id);
    if (!p) return;
    ctx.pendingRequests.delete(reply.id);
    if (reply.type === 'error') p.reject(new Error(reply.message || 'worker error'));
    else p.resolve(reply);
  });
  w.addEventListener('error', function(e) {
    // A top-level worker error fails every outstanding request — there's
    // no way to know which request the error pertains to, and the worker
    // may be dead. Reject all and reset so a future request can retry
    // the spawn.
    console.error('FlatPPL sampler worker error:', e.message || e);
    for (var entry of ctx.pendingRequests.values()) entry.reject(new Error(e.message || 'worker crashed'));
    ctx.pendingRequests.clear();
    try { w.terminate(); } catch (_) {}
    if (ctx.samplerWorker === w) {
      ctx.samplerWorker = null;
      ctx.samplerWorkerPromise = null;
    }
  });
}

export function sendWorkerNow(ctx, w, msg) {
  var id = ++ctx.samplerReqId;
  w.postMessage(Object.assign({ id: id }, msg));
}

export function sendWorker(ctx, msg) {
  return ensureSamplerWorker(ctx).then(function(w) {
    var id = ++ctx.samplerReqId;
    var wrapped = Object.assign({ id: id }, msg);
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
export function cancelAllSampling(ctx) {
  if (ctx.samplerWorker) {
    try { ctx.samplerWorker.terminate(); } catch (_) {}
    ctx.samplerWorker = null;
    ctx.samplerWorkerPromise = null;
  }
  var entries = ctx.pendingRequests.values();
  ctx.pendingRequests = new Map();
  for (var entry of entries) {
    try { entry.reject(new Error('cancelled')); } catch (_) {}
  }
}
