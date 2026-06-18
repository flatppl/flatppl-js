// FlatPPL inference annotation. Renders each binding's inferred type — sourced
// from the language server's inlay hints — as a label-only CodeLens ABOVE the
// binding (`▷ <type>`), instead of VSCode's inline inlay-hint rendering (which
// is turned off for `flatppl` via configurationDefaults in package.json).
//
// Off by default: gated on the `flatppl.inference.show` setting. The toggle
// command flips that setting; the provider re-fires when it changes. This is
// vscode-coupled, so build-vendor BUNDLES it (the .vsix ships no node_modules).
import * as vscode from 'vscode';
import { inferenceLensTitle, makeDebounced } from './lspHelpers';

const SETTING = 'inference.show';

export function registerInferenceLens(context: vscode.ExtensionContext): void {
  const emitter = new vscode.EventEmitter<void>();

  const provider: vscode.CodeLensProvider = {
    onDidChangeCodeLenses: emitter.event,
    async provideCodeLenses(document) {
      const show = vscode.workspace
        .getConfiguration('flatppl')
        .get<boolean>(SETTING, false);
      if (!show) return [];

      // Whole-document range; VSCode clamps the end position.
      const fullRange = new vscode.Range(0, 0, document.lineCount, 0);
      // Re-source the server's inlay hints through VSCode's provider registry.
      // This works even though inline inlay-hint *rendering* is off for flatppl
      // (the setting controls display, not provider invocation).
      const hints = await vscode.commands.executeCommand<vscode.InlayHint[]>(
        'vscode.executeInlayHintProvider', document.uri, fullRange,
      );
      if (!hints) return [];

      const lenses: vscode.CodeLens[] = [];
      for (const h of hints) {
        const title = inferenceLensTitle(h.label as string | Array<{ value: string }>);
        if (!title) continue;
        const line = h.position.line;
        // Empty command string => non-clickable, label-only lens. Already
        // resolved (has a command), so no resolveCodeLens pass is needed.
        lenses.push(new vscode.CodeLens(
          new vscode.Range(line, 0, line, 0),
          { title, command: '' },
        ));
      }
      return lenses;
    },
  };

  const providerReg = vscode.languages.registerCodeLensProvider(
    { language: 'flatppl' }, provider,
  );

  const toggleCmd = vscode.commands.registerCommand(
    'flatppl.toggleInference', async () => {
      const cfg = vscode.workspace.getConfiguration('flatppl');
      const current = cfg.get<boolean>(SETTING, false);
      await cfg.update(SETTING, !current, vscode.ConfigurationTarget.Global);
    },
  );

  // Re-render when the toggle setting changes.
  const cfgListener = vscode.workspace.onDidChangeConfiguration((e) => {
    if (e.affectsConfiguration(`flatppl.${SETTING}`)) emitter.fire();
  });

  // VSCode re-requests lenses on edit, but the server's hints may lag a beat
  // behind analysis; a debounced re-fire pulls fresh hints once it catches up.
  const refresh = makeDebounced(() => emitter.fire(), 400);
  const docListener = vscode.workspace.onDidChangeTextDocument((e) => {
    if (e.document.languageId === 'flatppl') refresh.call();
  });

  context.subscriptions.push(
    providerReg, toggleCmd, cfgListener, docListener, emitter,
    { dispose: () => refresh.cancel() },
  );
}
