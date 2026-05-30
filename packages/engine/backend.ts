// backend.ts — backend abstraction (minimal seed).
//
// The Backend owns environment-specific traits the engine consults to
// decide HOW to execute (parallelism threshold + spawn capability for
// commit 4; broadcast helpers + storage type later when accelerated
// backends land). Today there is exactly one impl: the default JS
// backend with no Worker spawn (single-threaded). Hosts (web /
// vscode-extension) replace it at bootstrap.
//
// Conservative shape — only fields the engine actually consults today.

export interface Backend {
  readonly name: string;
  // Below this N, sampleN/truncateSampleN stay on the calling thread.
  // Above it, the orchestrator may fan out across workers (if
  // FLATPPL_PARALLEL_SAMPLE flag is set + maxWorkers > 1 + spawnWorker
  // returns non-null). Tuneable per-backend: CPU IPC threshold ~4K
  // atoms; future GPU/TF.js kernel-launch threshold higher.
  readonly parallelismThreshold: number;
  // Worker cap, already clamped by the host's environment.
  readonly maxWorkers: number;
  // Returns a fresh Worker, or null if this host can't spawn. The pool
  // calls this lazily on first parallel-eligible sampleN. Typed as
  // 'any' because Node's worker_threads.Worker and the browser Worker
  // have different shapes — keeping the seam platform-agnostic avoids
  // cross-platform type imports here.
  spawnWorker(): any | null;
}

// Default JS backend: single-threaded, no spawn. Hosts that can do
// better (web with Worker, vscode-extension with worker_threads) call
// setBackend() at bootstrap with a richer impl.
export const jsBackend: Backend = {
  name: 'js',
  parallelismThreshold: 4096,
  maxWorkers: 1,
  spawnWorker: () => null,
};

let _active: Backend = jsBackend;
export function getBackend(): Backend { return _active; }
export function setBackend(b: Backend): void { _active = b; }

// FLATPPL_PARALLEL_SAMPLE flag — when truthy, sampleN/truncateSampleN
// fan out across pool workers above parallelismThreshold. Until commit 5
// removes the flag, default off. Read from globalThis at call time so
// hosts can flip it dynamically (e.g. devtools console for testing).
export function parallelSampleEnabled(): boolean {
  const g: any = (typeof globalThis !== 'undefined') ? globalThis : {};
  if (g.FLATPPL_PARALLEL_SAMPLE) return true;
  if (g.process && g.process.env && g.process.env.FLATPPL_PARALLEL_SAMPLE) return true;
  return false;
}
