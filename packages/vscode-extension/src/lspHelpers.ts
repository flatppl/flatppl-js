// Pure helpers for the FlatPPL language client: host→triple resolution,
// server-binary resolution, and catalogue reading. This module imports only
// Node built-ins — NO `vscode`, NO `vscode-languageclient` — so it is
// type-stripped (not bundled) and can be unit-tested by a plain `require`
// without stubbing the VS Code host. The VS Code-coupled lifecycle lives in
// `lspClient.ts`, which imports these.
import * as fs from 'node:fs';
import * as path from 'node:path';

/** Map a Node host (process.platform + process.arch) to the release target
 *  triple and executable suffix. Returns null for unsupported hosts.
 *
 *  INTENTIONAL DUPLICATE: `hostTripleForNode` in `../build-lsp-source.cjs`
 *  mirrors this table. The build script needs the mapping in plain JS (no TS
 *  transpile step), so the two are kept separate by design — change both
 *  together if the triple set ever changes. */
export function hostTriple(
  platform: NodeJS.Platform,
  arch: string,
): { triple: string; exe: string } | null {
  const exe = platform === 'win32' ? '.exe' : '';
  if (platform === 'darwin' && arch === 'arm64')
    return { triple: 'aarch64-apple-darwin', exe };
  if (platform === 'darwin' && arch === 'x64')
    return { triple: 'x86_64-apple-darwin', exe };
  if (platform === 'linux' && arch === 'x64')
    return { triple: 'x86_64-unknown-linux-musl', exe };
  if (platform === 'linux' && arch === 'arm64')
    return { triple: 'aarch64-unknown-linux-musl', exe };
  if (platform === 'win32' && arch === 'x64')
    return { triple: 'x86_64-pc-windows-msvc', exe };
  return null;
}

/** The bundled binary filename for a host, e.g. `flatppl-lsp-aarch64-apple-darwin`. */
export function bundledBinaryName(
  platform: NodeJS.Platform,
  arch: string,
): string | null {
  const h = hostTriple(platform, arch);
  return h ? `flatppl-lsp-${h.triple}${h.exe}` : null;
}

/** Resolve the server binary path: explicit `flatppl.server.path` override
 *  wins; otherwise the bundled per-platform binary under `<extRoot>/bin`.
 *  Returns { path, source }. Throws if neither exists. */
export function resolveServerBinary(
  extRoot: string,
  overridePath: string,
  platform: NodeJS.Platform,
  arch: string,
  existsSync: (p: string) => boolean = fs.existsSync,
): { path: string; source: 'override' | 'bundled' } {
  const override = overridePath.trim();
  if (override) {
    if (!existsSync(override))
      throw new Error(
        `flatppl.server.path is set to "${override}" but no file exists there.`,
      );
    return { path: override, source: 'override' };
  }
  const name = bundledBinaryName(platform, arch);
  if (!name)
    throw new Error(
      `No bundled flatppl-lsp binary for ${platform}/${arch}. ` +
        `Set "flatppl.server.path" to a locally built binary.`,
    );
  const bundled = path.join(extRoot, 'bin', name);
  if (!existsSync(bundled))
    throw new Error(
      `Bundled flatppl-lsp binary not found at ${bundled}. ` +
        `Set "flatppl.server.path" to a locally built binary.`,
    );
  return { path: bundled, source: 'bundled' };
}

/** Read every `*.ron` in `dir` and return their source strings. Unreadable
 *  files are skipped (reported via `onWarn`); a missing/empty dir yields []. */
export function readCatalogues(
  dir: string,
  onWarn: (msg: string) => void = () => {},
  deps: {
    existsSync?: (p: string) => boolean;
    readdirSync?: (p: string) => string[];
    readFileSync?: (p: string) => string;
  } = {},
): string[] {
  const existsSync = deps.existsSync ?? fs.existsSync;
  const readdirSync = deps.readdirSync ?? ((p: string) => fs.readdirSync(p));
  const readFileSync =
    deps.readFileSync ?? ((p: string) => fs.readFileSync(p, 'utf8'));
  const d = dir.trim();
  if (!d || !existsSync(d)) return [];
  const sources: string[] = [];
  for (const name of readdirSync(d).sort()) {
    if (!name.endsWith('.ron')) continue;
    try {
      sources.push(readFileSync(path.join(d, name)));
    } catch (e) {
      onWarn(`Could not read catalogue ${name}: ${(e as Error).message}`);
    }
  }
  return sources;
}

/** A serial task queue: each enqueued async op runs only after all previously
 *  enqueued ops have settled, so they never overlap. Returns a promise that
 *  resolves when this op finishes. A failing op does not break the chain. */
export function makeSerialQueue(): (op: () => Promise<void>) => Promise<void> {
  let tail: Promise<void> = Promise.resolve();
  return (op) => {
    const run = tail.then(op, op); // run `op` whether or not the prior op rejected
    // keep the chain alive even if `op` rejects
    tail = run.catch(() => {});
    return run;
  };
}

/** Wrap `fn` so that rapid calls within `ms` collapse into a single trailing
 *  invocation. Returns the debounced function plus a `cancel` to clear a
 *  pending call (e.g. on dispose). */
export function makeDebounced(
  fn: () => void,
  ms: number,
): { call: () => void; cancel: () => void } {
  let timer: ReturnType<typeof setTimeout> | undefined;
  return {
    call() {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => {
        timer = undefined;
        fn();
      }, ms);
    },
    cancel() {
      if (timer) clearTimeout(timer);
      timer = undefined;
    },
  };
}
