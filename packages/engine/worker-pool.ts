'use strict';

// FlatPPL persistent per-session worker pool.
//
// The pool fans a single conceptual `runChunked` call across N workers,
// each running the message handler defined in worker.ts (sampleN /
// truncateSampleN / evaluateN / profileN / logDensityN). It is the
// only place in the engine that talks to multiple workers — the rest
// of the engine sees one `runChunked()` Promise and the orchestrator
// stitches the per-chunk results.
//
// Lifecycle:
//
//   * Lazy first-call: `getPool()` spawns up to `backend.maxWorkers`
//     workers via `backend.spawnWorker()`. If the host can't spawn
//     (default JS backend returns null), `getPool()` returns null and
//     callers fall back to in-process execution.
//   * Per-session lifetime: workers stay warm across multiple
//     `runChunked` calls. `disposePool()` is the only kill switch —
//     the host fires it on `cancelAllSampling` and `beforeunload`.
//   * No idle reaper: bounded by session lifetime, so we don't pay
//     timer or accounting overhead. Per-session workers are cheap
//     compared to per-call respawn.
//
// Worker transport — feature-detected to support both shapes:
//
//   * Browser Worker — `worker.onmessage = handler`, `worker.onerror`,
//     `worker.terminate()`. This is what the VS Code webview uses.
//   * Node worker_threads.Worker — `.on('message', handler)`,
//     `.on('error', handler)`, `.terminate()`. Used by tests + any
//     server-side sampling path.
//
//   `backend.spawnWorker()` returns the platform-appropriate object;
//   we sniff which API it exposes by checking whether `onmessage` is
//   a settable property (browser shape) or only `.on()` exists (Node
//   shape). Tests can substitute either — pure feature detection,
//   no host imports.
//
// Transferables — Float64Array.buffer is moved zero-copy across the
// worker boundary in both directions. We scan request payloads for
// Float64Array fields (top-level + refArrays values) and pass their
// buffers in the transferList; the worker's reply uses the same
// convention (samples.buffer is moved into the parent's heap).
//
// Scheduling — FIFO queue. When `runChunked(chunks)` is called with
// more chunks than idle workers, the overflow chunks are queued and
// pulled off on each reply. No work-stealing — the chunks are
// pre-sized by the caller so there's no per-worker imbalance to
// reclaim.

import { getBackend } from './backend.ts';

// ---------------------------------------------------------------------------
// Public surface — see header for protocol notes.
// ---------------------------------------------------------------------------

export interface PoolChunkRequest {
  ir: any;
  count: number;                 // chunk size, not total
  refArrays?: any;               // pre-sliced per chunk
  seed: [number, number];        // PhiloxKey for this chunk (split key)
  env?: any;
  // Pass-through for truncate / profile modes:
  setDescr?: any;
  mode?: 'cdf' | 'rejection';
  budget?: number;
  observed?: any;                // for logDensityN if ever fanned-out
}

export interface PoolChunkReply {
  samples?: any;                 // Float64Array for sampleN/truncateSampleN
  result?: number;               // for logDensityN if ever fanned-out
  n_eff?: number;                // truncate only
  totalDraws?: number;           // truncate only
}

export type PoolMsgType =
  | 'sampleN'
  | 'truncateSampleN'
  | 'evaluateN'
  | 'profileN'
  | 'logDensityN';

export interface WorkerPool {
  runChunked(
    msgType: PoolMsgType,
    chunks: PoolChunkRequest[],
  ): Promise<PoolChunkReply[]>;
  terminate(): void;
  readonly size: number;         // worker count
}

// ---------------------------------------------------------------------------
// Internal worker slot bookkeeping.
// ---------------------------------------------------------------------------

interface PendingEntry {
  resolve: (reply: PoolChunkReply) => void;
  reject: (err: any) => void;
  chunkIndex: number;
}

interface WorkerSlot {
  worker: any;
  // Pending message id → resolver pair. In steady state at most one
  // entry per slot (FIFO scheduling, one chunk per worker at a time),
  // but the map shape leaves room for pipelining without changing the
  // reply-routing logic.
  pending: Map<number, PendingEntry>;
  busy: boolean;
  // Per-slot bound listeners — kept so replaceWorker() can detach
  // them and freshly attach the new worker. Browser path uses
  // .onmessage/.onerror property assignment (no detach needed since
  // we drop the worker), Node path uses .on() (also dropped with the
  // worker). The references aren't strictly needed for cleanup but
  // are useful for in-place worker replacement on error.
  onMsg: (msg: any) => void;
  onErr: (err: any) => void;
  // Detected once per slot — true if onmessage is a settable property
  // (browser Worker shape). Cached so we don't re-sniff per send.
  isBrowserShape: boolean;
}

// Module-level pool — single instance per process/session. `null` until
// `getPool()` is called for the first time AND the backend can spawn.
let _pool: WorkerPoolImpl | null = null;
// Sticky "we tried and couldn't" flag. Once a host has returned null
// from spawnWorker the first time, don't keep re-trying on every call.
let _spawnFailed = false;

// Monotonic message id, scoped to the pool instance. Reset on dispose.
let _nextId = 1;

// ---------------------------------------------------------------------------
// Feature detection — browser Worker vs node:worker_threads.
// ---------------------------------------------------------------------------

// Returns true when `w` exposes `.onmessage` as a settable property
// (Worker / MessagePort shape). Node's Worker constructor in
// `worker_threads` does NOT expose .onmessage; you must use .on('message').
// Detection is done by attempting to assign a no-op and checking
// whether the slot took. A simple `'onmessage' in w` is unreliable
// (Node's Worker has no such own property, but a sub-prototype could
// add one); the assignment probe is the most robust.
function detectBrowserShape(w: any): boolean {
  if (!w) return false;
  // Browser-side Worker is `instanceof Worker` but we avoid that name
  // since Node may not define the global. Just look for a postMessage
  // method (both have it) and a settable onmessage slot.
  if (typeof w.postMessage !== 'function') return false;
  if (typeof w.onmessage === 'function' || w.onmessage === null) return true;
  // Some test doubles set onmessage to undefined explicitly; treat as
  // browser-shape if there's no `.on` method to use instead.
  if (typeof w.on !== 'function') return true;
  return false;
}

// ---------------------------------------------------------------------------
// Transferables — collect Float64Array buffers from a request payload.
// ---------------------------------------------------------------------------

// Scan the request payload for Float64Array (and other typed-array)
// buffers and return their .buffer references for the transferList.
// We look at top-level fields plus refArrays values — that covers
// every typed buffer the current worker.ts handlers consume. If
// future message shapes add more typed-array payloads, extend this
// scanner (callers don't choose what to transfer).
function collectTransferables(payload: any): ArrayBuffer[] {
  const out: ArrayBuffer[] = [];
  if (!payload || typeof payload !== 'object') return out;
  for (const k of Object.keys(payload)) {
    const v = payload[k];
    if (v && v.BYTES_PER_ELEMENT !== undefined && v.buffer instanceof ArrayBuffer) {
      out.push(v.buffer);
    } else if (k === 'refArrays' && v && typeof v === 'object') {
      for (const rk of Object.keys(v)) {
        const ra = v[rk];
        if (ra && ra.BYTES_PER_ELEMENT !== undefined && ra.buffer instanceof ArrayBuffer) {
          out.push(ra.buffer);
        }
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Pool implementation.
// ---------------------------------------------------------------------------

class WorkerPoolImpl implements WorkerPool {
  slots: WorkerSlot[] = [];
  // FIFO queue of (msgType, chunk) waiting for an idle worker. Filled
  // when runChunked is called with more chunks than free slots.
  queue: Array<{
    msgType: PoolMsgType;
    chunk: PoolChunkRequest;
    entry: PendingEntry;
  }> = [];

  get size(): number { return this.slots.length; }

  // Spawn up to `n` workers via backend.spawnWorker(). Stops early if
  // spawnWorker returns null (host capability), with a non-empty pool
  // counted as success.
  spawnSlots(n: number): boolean {
    const backend = getBackend();
    for (let i = 0; i < n; i++) {
      const w = backend.spawnWorker();
      if (!w) break;
      this.slots.push(this.makeSlot(w));
    }
    return this.slots.length > 0;
  }

  makeSlot(worker: any): WorkerSlot {
    const slot: WorkerSlot = {
      worker,
      pending: new Map(),
      busy: false,
      onMsg: () => {},
      onErr: () => {},
      isBrowserShape: detectBrowserShape(worker),
    };
    slot.onMsg = (msg: any) => this.handleReply(slot, msg);
    slot.onErr = (err: any) => this.handleWorkerError(slot, err);
    this.attachListeners(slot);
    return slot;
  }

  // Wire up message / error listeners per platform. Browser Worker
  // uses .onmessage property assignment (idempotent — last assignment
  // wins); Node worker_threads uses .on() (additive, but we only
  // attach once per slot, and `replaceWorker` drops the old worker
  // entirely so the listeners go with it).
  attachListeners(slot: WorkerSlot): void {
    if (slot.isBrowserShape) {
      slot.worker.onmessage = (ev: any) => slot.onMsg(ev && 'data' in ev ? ev.data : ev);
      slot.worker.onerror   = (ev: any) => slot.onErr(ev && 'message' in ev ? ev : (ev && ev.error) || ev);
    } else {
      slot.worker.on('message', slot.onMsg);
      slot.worker.on('error',   slot.onErr);
    }
  }

  // ---- send / route -------------------------------------------------------

  // Send `chunk` to `slot` and store the resolver under the message
  // id. Caller is responsible for ensuring slot.busy was false on entry
  // (FIFO scheduler guarantees this).
  send(slot: WorkerSlot, msgType: PoolMsgType, chunk: PoolChunkRequest, entry: PendingEntry): void {
    const id = _nextId++;
    slot.pending.set(id, entry);
    slot.busy = true;
    const msg = { id, type: msgType, ...chunk };
    const transfer = collectTransferables(chunk);
    // postMessage signature is identical in both shapes (msg, transferList).
    // Node's worker_threads accepts only ArrayBuffers (not typed arrays)
    // in the transferList; collectTransferables already returns buffers.
    if (transfer.length > 0) {
      slot.worker.postMessage(msg, transfer);
    } else {
      slot.worker.postMessage(msg);
    }
  }

  handleReply(slot: WorkerSlot, raw: any): void {
    if (!raw || typeof raw !== 'object') return;
    const id = raw.id;
    const entry = slot.pending.get(id);
    if (!entry) return;  // stale or unknown id — drop
    slot.pending.delete(id);
    slot.busy = false;
    if (raw.type === 'error') {
      const err: any = new Error(raw.message || 'worker error');
      if (raw.stack) err.stack = raw.stack;
      entry.reject(err);
    } else {
      // Translate worker reply shape → PoolChunkReply shape. The
      // worker's 'samples' replies carry `samples` (Float64Array) +
      // optional `n_eff` / `totalDraws` (truncate) / `result`
      // (singular value reply, defensive — current workers don't emit
      // this shape but logDensityN-aggregate variants might).
      const reply: PoolChunkReply = {};
      if (raw.samples    !== undefined) reply.samples    = raw.samples;
      if (raw.result     !== undefined) reply.result     = raw.result;
      if (raw.n_eff      !== undefined) reply.n_eff      = raw.n_eff;
      if (raw.totalDraws !== undefined) reply.totalDraws = raw.totalDraws;
      entry.resolve(reply);
    }
    // Pull next queued chunk if any.
    this.pump(slot);
  }

  // Worker top-level error — node 'error' event or browser 'error' event.
  // Rejects every pending entry on this slot and rebuilds the slot with
  // a fresh worker, so the pool stays at full size for subsequent calls.
  handleWorkerError(slot: WorkerSlot, err: any): void {
    const wrapped = err instanceof Error ? err : new Error(String(err && err.message || err));
    for (const entry of slot.pending.values()) entry.reject(wrapped);
    slot.pending.clear();
    slot.busy = false;
    this.replaceWorker(slot);
    // After replacement, try to drain the queue onto the now-idle slot.
    this.pump(slot);
  }

  replaceWorker(slot: WorkerSlot): void {
    // Drop the broken worker and spawn a fresh one. If the backend
    // can't spawn (rare — implies the host lost capability mid-session)
    // the slot stays dead-but-empty and pump() will just skip it.
    try { slot.worker.terminate(); } catch (_e) { /* ignore */ }
    const fresh = getBackend().spawnWorker();
    if (!fresh) {
      // No replacement available. Leave the slot in a quiescent state
      // (no listeners attached, busy=false). pump() never picks it
      // because we'll filter dead slots; mark with null worker.
      slot.worker = null;
      return;
    }
    slot.worker = fresh;
    slot.isBrowserShape = detectBrowserShape(fresh);
    this.attachListeners(slot);
  }

  // After a slot becomes idle, hand it the next queued chunk (if any).
  // Idempotent — safe to call when the queue is empty.
  pump(slot: WorkerSlot): void {
    if (!slot.worker) return;
    if (slot.busy) return;
    const next = this.queue.shift();
    if (!next) return;
    this.send(slot, next.msgType, next.chunk, next.entry);
  }

  // ---- public API ---------------------------------------------------------

  // Top up live slots to at least `want`, capped by backend.maxWorkers.
  // Lazy on-demand growth — keeps small-N callers from spawning the
  // full pool's worth of workers up front. Reuses spawnSlots
  // semantics: stops on the first spawnWorker returning null.
  ensureSlots(want: number): void {
    const backend = getBackend();
    const cap = Math.max(1, backend.maxWorkers | 0);
    const target = Math.min(want, cap);
    const liveCount = this.slots.reduce(
      (n, s) => n + (s.worker ? 1 : 0), 0);
    if (liveCount >= target) return;
    this.spawnSlots(target - liveCount);
  }

  runChunked(msgType: PoolMsgType, chunks: PoolChunkRequest[]): Promise<PoolChunkReply[]> {
    // Lazy grow: ensure the pool is sized for at least this batch's
    // chunk count (capped at maxWorkers). On a small first call we
    // spawn only what we need; later larger calls top up.
    this.ensureSlots(chunks.length);

    const replies: PoolChunkReply[] = new Array(chunks.length);
    const promises: Array<Promise<void>> = new Array(chunks.length);

    for (let i = 0; i < chunks.length; i++) {
      const chunkIdx = i;
      promises[i] = new Promise<void>((resolve, reject) => {
        const entry: PendingEntry = {
          chunkIndex: chunkIdx,
          resolve: (r: PoolChunkReply) => { replies[chunkIdx] = r; resolve(); },
          reject,
        };
        // Find an idle, live slot; otherwise queue.
        const slot = this.slots.find((s) => s.worker && !s.busy);
        if (slot) {
          this.send(slot, msgType, chunks[chunkIdx], entry);
        } else {
          this.queue.push({ msgType, chunk: chunks[chunkIdx], entry });
        }
      });
    }
    return Promise.all(promises).then(() => replies);
  }

  terminate(): void {
    // Reject any in-flight or queued work so callers don't hang on
    // a dispose that crosses a runChunked() boundary.
    const disposed = new Error('worker pool disposed');
    for (const slot of this.slots) {
      for (const entry of slot.pending.values()) entry.reject(disposed);
      slot.pending.clear();
      if (slot.worker) {
        try { slot.worker.terminate(); } catch (_e) { /* ignore */ }
      }
      slot.worker = null;
    }
    for (const q of this.queue) q.entry.reject(disposed);
    this.queue.length = 0;
    this.slots.length = 0;
  }
}

// ---------------------------------------------------------------------------
// Module-level accessors.
// ---------------------------------------------------------------------------

export function getPool(): WorkerPool | null {
  if (_pool) return _pool;
  if (_spawnFailed) return null;
  // Spawn-capability probe: try ONE worker. If null, the host can't
  // spawn — sticky-flag and return null. The rest grow lazily via
  // `runChunked → ensureSlots`, sized to actual chunk count (capped
  // at backend.maxWorkers). Avoids eagerly spawning all maxWorkers
  // workers on a small first call.
  const pool = new WorkerPoolImpl();
  if (!pool.spawnSlots(1)) {
    _spawnFailed = true;
    return null;
  }
  _pool = pool;
  return _pool;
}

export function disposePool(): void {
  if (_pool) {
    _pool.terminate();
    _pool = null;
  }
  // Reset the sticky-failure flag so a backend swap (e.g. host
  // reinstall + setBackend) can re-attempt spawning on the next
  // getPool() call.
  _spawnFailed = false;
  // Reset the monotonic message-id counter to match the doc-comment
  // contract at the declaration site.
  _nextId = 1;
}
