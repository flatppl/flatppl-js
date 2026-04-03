'use strict';
const vscode = require('vscode');
const { parseFlatPPL, createBindingMap, computeSubDAG, findBindingAtLine } = require('./src/parser');
const { DAGPanel } = require('./src/dagView');

function activate(context) {
  // Cache parsed results to avoid re-parsing on every cursor move
  let cachedUri = '';
  let cachedVersion = -1;
  let cachedResult = null;

  function getParsed(document) {
    const uri = document.uri.toString();
    if (uri === cachedUri && document.version === cachedVersion) return cachedResult;
    const bindings = parseFlatPPL(document.getText());
    const bindingMap = createBindingMap(bindings);
    cachedResult = { bindings, bindingMap };
    cachedVersion = document.version;
    cachedUri = uri;
    return cachedResult;
  }

  function showDAGForCursor(editor) {
    if (!editor || editor.document.languageId !== 'flatppl') return false;

    const { bindings, bindingMap } = getParsed(editor.document);
    const line = editor.selection.active.line;
    const binding = findBindingAtLine(bindings, line);
    if (!binding) return false;

    const name = binding.names[0];
    const dagData = computeSubDAG(bindingMap, name);

    DAGPanel.createOrShow(context);
    DAGPanel.currentPanel.update(dagData, name);
    return true;
  }

  // Register the "FlatPPL: Show DAG" command
  const command = vscode.commands.registerCommand('flatppl.showDAG', () => {
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document.languageId !== 'flatppl') {
      vscode.window.showErrorMessage('Place cursor in a FlatPPL file');
      return;
    }
    if (!showDAGForCursor(editor)) {
      vscode.window.showInformationMessage('Place cursor on a variable definition line');
    }
  });

  // Live-update the DAG panel when cursor moves to a different definition
  let updateTimeout;
  let lastShownLine = -1;

  const selectionListener = vscode.window.onDidChangeTextEditorSelection(e => {
    if (!DAGPanel.currentPanel) return;
    if (e.textEditor.document.languageId !== 'flatppl') return;

    const line = e.selections[0].active.line;
    if (line === lastShownLine) return;

    clearTimeout(updateTimeout);
    updateTimeout = setTimeout(() => {
      const { bindings } = getParsed(e.textEditor.document);
      const binding = findBindingAtLine(bindings, line);
      if (binding) {
        lastShownLine = line;
        showDAGForCursor(e.textEditor);
      }
    }, 150);
  });

  context.subscriptions.push(command, selectionListener);
}

function deactivate() {}

module.exports = { activate, deactivate };
