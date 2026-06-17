import * as fs from 'node:fs';
import * as path from 'node:path';
import * as vscode from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from 'vscode-languageclient/node';

/** Map a Node host (process.platform + process.arch) to the release target
 *  triple and executable suffix. Returns null for unsupported hosts. */
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

const OUTPUT_CHANNEL = 'FlatPPL Language Server';

/** Build (but do not start) the LanguageClient for the resolved binary and
 *  the catalogues read from the configured directory. */
export function createClient(
  serverPath: string,
  catalogues: string[],
): LanguageClient {
  const serverOptions: ServerOptions = {
    command: serverPath,
    transport: TransportKind.stdio,
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'flatppl' }],
    initializationOptions: { catalogues },
    outputChannelName: OUTPUT_CHANNEL,
  };
  return new LanguageClient(
    'flatppl-lsp',
    OUTPUT_CHANNEL,
    serverOptions,
    clientOptions,
  );
}

/** Owns the running client and the lifecycle: read config → resolve binary →
 *  read catalogues → start. `restart` tears down and rebuilds (used on
 *  catalogue/config change). Errors surface as a VS Code warning, not a throw. */
export class LspClientManager {
  private client: LanguageClient | undefined;

  constructor(private readonly extRoot: string) {}

  async start(): Promise<void> {
    const cfg = vscode.workspace.getConfiguration('flatppl');
    const serverPathSetting = cfg.get<string>('server.path', '');
    const cataloguesDir = cfg.get<string>('catalogues.path', '');
    let bin: { path: string; source: string };
    try {
      bin = resolveServerBinary(
        this.extRoot,
        serverPathSetting,
        process.platform,
        process.arch,
      );
    } catch (e) {
      vscode.window.showWarningMessage(
        `FlatPPL language server not started: ${(e as Error).message}`,
      );
      return;
    }
    const catalogues = readCatalogues(cataloguesDir, (m) =>
      vscode.window.showWarningMessage(`FlatPPL: ${m}`),
    );
    this.client = createClient(bin.path, catalogues);
    await this.client.start();
  }

  async stop(): Promise<void> {
    const c = this.client;
    this.client = undefined;
    if (c) await c.stop();
  }

  async restart(): Promise<void> {
    await this.stop();
    await this.start();
  }
}
