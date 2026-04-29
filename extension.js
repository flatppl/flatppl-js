'use strict';
const vscode = require('vscode');
const { processSource, computeSubDAG, findBindingAtLine } = require('./engine');
const { DAGPanel } = require('./src/dagView');

function activate(context) {
  // Cache parsed results to avoid re-parsing on every cursor move
  let cachedUri = '';
  let cachedVersion = -1;
  let cachedResult = null;

  // Diagnostic collection for FlatPPL errors/warnings
  const diagCollection = vscode.languages.createDiagnosticCollection('flatppl');
  context.subscriptions.push(diagCollection);

  function getParsed(document) {
    const uri = document.uri.toString();
    if (uri === cachedUri && document.version === cachedVersion) return cachedResult;

    const source = document.getText();
    const { ast, bindings, symbols, diagnostics } = processSource(source);
    cachedResult = { ast, bindings, symbols, diagnostics };
    cachedVersion = document.version;
    cachedUri = uri;

    // Update diagnostics in VS Code
    const vsDiags = diagnostics.map(d => {
      const range = new vscode.Range(
        d.loc.start.line, d.loc.start.col,
        d.loc.end.line, d.loc.end.col
      );
      const severity = d.severity === 'error'
        ? vscode.DiagnosticSeverity.Error
        : d.severity === 'warning'
        ? vscode.DiagnosticSeverity.Warning
        : vscode.DiagnosticSeverity.Information;
      return new vscode.Diagnostic(range, d.message, severity);
    });
    diagCollection.set(document.uri, vsDiags);

    return cachedResult;
  }

  function setupZoomHandler() {
    if (!DAGPanel.currentPanel) return;
    DAGPanel.currentPanel.onZoomInto = (nodeId) => {
      if (!cachedResult) return;
      const dagData = computeSubDAG(cachedResult.bindings, nodeId);
      if (dagData.nodes.length > 0) {
        DAGPanel.currentPanel.update(dagData, nodeId, null, true);
      }
    };
  }

  function showDAGForCursor(editor) {
    if (!editor || editor.document.languageId !== 'flatppl') return false;

    const { bindings } = getParsed(editor.document);
    const pos = editor.selection.active;
    const binding = findBindingAtLine(bindings, pos.line, pos.character);
    if (!binding) return false;

    const name = binding.name;
    const dagData = computeSubDAG(bindings, name);

    DAGPanel.createOrShow(context);
    setupZoomHandler();
    DAGPanel.currentPanel.update(dagData, name, editor.document.uri, false);
    return true;
  }

  // --- Commands ---

  const showDagCmd = vscode.commands.registerCommand('flatppl.showDAG', () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'flatppl') {
      vscode.window.showErrorMessage('Place cursor in a FlatPPL file');
      return;
    }
    if (!showDAGForCursor(editor)) {
      vscode.window.showInformationMessage('Place cursor on a variable definition line');
    }
  });

  // --- Live DAG update on cursor move ---

  let updateTimeout;
  let lastShownName = '';

  const selectionListener = vscode.window.onDidChangeTextEditorSelection(e => {
    if (!DAGPanel.currentPanel) return;
    if (e.textEditor.document.languageId !== 'flatppl') return;

    clearTimeout(updateTimeout);
    updateTimeout = setTimeout(() => {
      const { bindings } = getParsed(e.textEditor.document);
      const pos = e.selections[0].active;
      const binding = findBindingAtLine(bindings, pos.line, pos.character);
      if (binding && binding.name !== lastShownName) {
        lastShownName = binding.name;
        showDAGForCursor(e.textEditor);
      }
    }, 150);
  });

  // --- Re-parse on document change (for diagnostics) ---

  const changeListener = vscode.workspace.onDidChangeTextDocument(e => {
    if (e.document.languageId === 'flatppl') {
      getParsed(e.document);
    }
  });

  // --- Parse on open ---

  const openListener = vscode.workspace.onDidOpenTextDocument(doc => {
    if (doc.languageId === 'flatppl') {
      getParsed(doc);
    }
  });

  // --- Go-to-definition ---

  const defProvider = vscode.languages.registerDefinitionProvider('flatppl', {
    provideDefinition(document, position) {
      const { bindings } = getParsed(document);
      const wordRange = document.getWordRangeAtPosition(position);
      if (!wordRange) return null;
      const word = document.getText(wordRange);
      const binding = bindings.get(word);
      if (!binding) return null;
      return new vscode.Location(document.uri,
        new vscode.Position(binding.nameLoc.start.line, binding.nameLoc.start.col));
    }
  });

  // --- Hover ---

  const hoverProvider = vscode.languages.registerHoverProvider('flatppl', {
    provideHover(document, position) {
      const { bindings } = getParsed(document);
      const wordRange = document.getWordRangeAtPosition(position);
      if (!wordRange) return null;
      const word = document.getText(wordRange);
      const binding = bindings.get(word);
      if (!binding) return null;

      const lines = [
        `**${binding.name}** \u2014 *${binding.type}*`,
        '```flatppl',
        `${binding.name} = ${binding.rhs}`,
        '```',
      ];
      if (binding.deps.length > 0) {
        lines.push(`Dependencies: ${binding.deps.join(', ')}`);
      }
      const md = new vscode.MarkdownString(lines.join('\n'));
      md.isTrusted = true;
      return new vscode.Hover(md, wordRange);
    }
  });

  // --- Document symbols (outline) ---

  const symbolProvider = vscode.languages.registerDocumentSymbolProvider('flatppl', {
    provideDocumentSymbols(document) {
      const { symbols } = getParsed(document);
      const kindMap = {
        Variable: vscode.SymbolKind.Variable,
        Function: vscode.SymbolKind.Function,
        Constant: vscode.SymbolKind.Constant,
        Module: vscode.SymbolKind.Module,
      };
      return symbols.map(s => new vscode.DocumentSymbol(
        s.name,
        s.type,
        kindMap[s.kind] || vscode.SymbolKind.Variable,
        new vscode.Range(s.loc.start.line, s.loc.start.col, s.loc.end.line, s.loc.end.col),
        new vscode.Range(s.nameLoc.start.line, s.nameLoc.start.col, s.nameLoc.end.line, s.nameLoc.end.col),
      ));
    }
  });

  // --- Clean up diagnostics on close ---

  const closeListener = vscode.workspace.onDidCloseTextDocument(doc => {
    diagCollection.delete(doc.uri);
  });

  // Parse any already-open FlatPPL documents
  for (const doc of vscode.workspace.textDocuments) {
    if (doc.languageId === 'flatppl') getParsed(doc);
  }

  context.subscriptions.push(
    showDagCmd, selectionListener, changeListener, openListener, closeListener,
    defProvider, hoverProvider, symbolProvider,
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
