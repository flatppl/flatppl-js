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
import { resolveServerBinary, readCatalogues } from './lspHelpers';

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
