// The VS Code-coupled language client. This module imports `vscode` and
// `vscode-languageclient`, so build-vendor BUNDLES it into a self-contained
// CommonJS file (the packaged .vsix ships no node_modules). The pure helpers
// it uses live in `lspHelpers.ts` (vscode-free, unit-tested directly).
import * as vscode from 'vscode';
import {
  LanguageClient,
  LanguageClientOptions,
  ServerOptions,
  TransportKind,
} from 'vscode-languageclient/node';
import {
  resolveServerBinary, readCatalogues, makeSerialQueue, urlSourcesFeedSig,
} from './lspHelpers';

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

export interface LspManager {
  start(): Promise<void>;
  stop(): Promise<void>;
  restart(): Promise<void>;
  /** Push client-fetched URL `load_module` sources into the server's FileSet
   *  (`flatppl/urlSources`). The server resolves URL deps but never fetches —
   *  the extension is the sole fetcher+truster (spec §04 #sec:url-cache). No-op
   *  when no server is running or the payload is empty/unchanged. */
  feedUrlSources(sources: Array<{ uri: string; text: string }>): Promise<void>;
}

/** Owns the running client lifecycle: read config → resolve binary → read
 *  catalogues → start. Errors surface as a VS Code warning, not a throw.
 *  `onStart` (if given) fires after each successful (re)start — the host uses
 *  it to (re)feed URL sources, since a fresh server starts with an empty
 *  FileSet. */
export function createLspManager(
  extRoot: string,
  onStart?: () => void,
): LspManager {
  let client: LanguageClient | undefined;
  let lastUrlSig: string | undefined;   // dedup identical re-feeds (per client)
  const enqueue = makeSerialQueue();

  async function start(): Promise<void> {
    const cfg = vscode.workspace.getConfiguration('flatppl');
    const serverPathSetting = cfg.get<string>('server.path', '');
    const cataloguesDir = cfg.get<string>('catalogues.path', '');
    let bin: { path: string; source: string };
    try {
      bin = resolveServerBinary(extRoot, serverPathSetting, process.platform, process.arch);
    } catch (e) {
      vscode.window.showWarningMessage(
        `FlatPPL language server not started: ${(e as Error).message}`,
      );
      return;
    }
    const catalogues = readCatalogues(cataloguesDir, (m) =>
      vscode.window.showWarningMessage(`FlatPPL: ${m}`),
    );
    const c = createClient(bin.path, catalogues);
    try {
      await c.start();
      client = c;
      onStart?.();   // fresh FileSet ⇒ host re-feeds URL sources
    } catch (e) {
      vscode.window.showWarningMessage(
        `FlatPPL language server failed to start: ${(e as Error).message}`,
      );
    }
  }

  async function stop(): Promise<void> {
    const c = client;
    client = undefined;
    lastUrlSig = undefined;   // next client's FileSet is empty ⇒ force a re-feed
    if (c) await c.stop();
  }

  async function feedUrlSources(
    sources: Array<{ uri: string; text: string }>,
  ): Promise<void> {
    const c = client;
    if (!c) return;                       // server not running ⇒ nothing to feed
    const sig = urlSourcesFeedSig(sources, lastUrlSig);
    if (sig === null) return;             // empty, or unchanged since last feed
    lastUrlSig = sig;
    try {
      await c.sendNotification('flatppl/urlSources', { sources });
    } catch (_e) {
      lastUrlSig = undefined;             // delivery failed ⇒ allow a retry
    }
  }

  function restart(): Promise<void> {
    return enqueue(async () => {
      await stop();
      await start();
    });
  }

  return { start, stop, restart, feedUrlSources };
}
