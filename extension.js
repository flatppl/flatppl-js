'use strict';
const vscode = require('vscode');
const { processSource, computeSubDAG, findBindingAtLine, builtins } = require('./engine');
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

  // --- Completion provider ---

  // Snippets for built-in distributions: typing the name and accepting the
  // suggestion drops in a placeholder list of standard parameters.
  const DIST_SNIPPETS = {
    Normal: 'Normal(mu = ${1:0.0}, sigma = ${2:1.0})',
    Exponential: 'Exponential(rate = ${1:1.0})',
    Uniform: 'Uniform(support = ${1:reals})',
    LogNormal: 'LogNormal(mu = ${1:0.0}, sigma = ${2:1.0})',
    Cauchy: 'Cauchy(location = ${1:0.0}, scale = ${2:1.0})',
    StudentT: 'StudentT(nu = ${1:1.0})',
    Logistic: 'Logistic(mu = ${1:0.0}, s = ${2:1.0})',
    Gamma: 'Gamma(shape = ${1:1.0}, rate = ${2:1.0})',
    InverseGamma: 'InverseGamma(shape = ${1:1.0}, scale = ${2:1.0})',
    Beta: 'Beta(alpha = ${1:1.0}, beta = ${2:1.0})',
    Weibull: 'Weibull(shape = ${1:1.0}, scale = ${2:1.0})',
    Bernoulli: 'Bernoulli(p = ${1:0.5})',
    Categorical: 'Categorical(p = ${1:[0.5, 0.5]})',
    Binomial: 'Binomial(n = ${1:1}, p = ${2:0.5})',
    Poisson: 'Poisson(rate = ${1:1.0})',
    GeneralizedNormal: 'GeneralizedNormal(mean = ${1:0.0}, alpha = ${2:1.0}, beta = ${3:2.0})',
    MvNormal: 'MvNormal(mu = ${1}, cov = ${2})',
    Dirichlet: 'Dirichlet(alpha = ${1})',
    Multinomial: 'Multinomial(n = ${1}, p = ${2})',
    Wishart: 'Wishart(nu = ${1}, scale = ${2})',
    InverseWishart: 'InverseWishart(nu = ${1}, scale = ${2})',
    LKJ: 'LKJ(n = ${1}, eta = ${2:1.0})',
    LKJCholesky: 'LKJCholesky(n = ${1}, eta = ${2:1.0})',
    PoissonProcess: 'PoissonProcess(intensity = ${1})',
    BinnedPoissonProcess: 'BinnedPoissonProcess(bins = ${1}, intensity = ${2})',
    Lebesgue: 'Lebesgue(support = ${1:reals})',
    Counting: 'Counting(support = ${1:integers})',
    Dirac: 'Dirac(value = ${1})',
  };

  // Snippets for special forms — drop in a body placeholder.
  const SPECIAL_SNIPPETS = {
    elementof: 'elementof(${1:reals})',
    external: 'external(${1:reals})',
    draw: 'draw(${1})',
    lawof: 'lawof(${1})',
    functionof: 'functionof(${1})',
    fn: 'fn(${1})',
    record: 'record(${1})',
    likelihoodof: 'likelihoodof(${1:kernel}, ${2:data})',
    bayesupdate: 'bayesupdate(${1:L}, ${2:prior})',
    disintegrate: 'disintegrate(${1:["field"]}, ${2:joint_model})',
    load_module: 'load_module("${1:path.flatppl}")',
    standard_module: 'standard_module("${1:name}", "${2:0.1}")',
    load_data: 'load_data(source = "${1:path}", valueset = ${2:reals})',
  };

  function makeBuiltinCompletions() {
    const items = [];

    // Special forms
    for (const name of builtins.SPECIAL_FORMS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Keyword);
      const snippet = SPECIAL_SNIPPETS[name];
      if (snippet) {
        item.insertText = new vscode.SnippetString(snippet);
        item.detail = 'special form';
      } else {
        item.detail = 'special form';
      }
      items.push(item);
    }

    // Distributions
    for (const name of builtins.DISTRIBUTIONS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Class);
      const snippet = DIST_SNIPPETS[name];
      if (snippet) item.insertText = new vscode.SnippetString(snippet);
      item.detail = 'distribution';
      items.push(item);
    }

    // Measure operations
    for (const name of builtins.MEASURE_OPS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Function);
      item.detail = 'measure operation';
      items.push(item);
    }

    // Built-in functions
    for (const name of builtins.BUILTIN_FUNCTIONS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Function);
      item.detail = 'built-in function';
      items.push(item);
    }

    // Set constructors
    for (const name of builtins.SET_CONSTRUCTORS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Function);
      item.detail = 'set constructor';
      items.push(item);
    }

    // Predefined sets
    for (const name of builtins.SETS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Constant);
      item.detail = 'set';
      items.push(item);
    }

    // Constants
    for (const name of builtins.CONSTANTS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Constant);
      item.detail = 'constant';
      items.push(item);
    }

    // Reserved names
    for (const name of builtins.RESERVED_NAMES) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Module);
      item.detail = 'reserved module';
      items.push(item);
    }

    return items;
  }

  const builtinCompletions = makeBuiltinCompletions();

  const completionProvider = vscode.languages.registerCompletionItemProvider('flatppl', {
    provideCompletionItems(document, position) {
      const { bindings } = getParsed(document);
      const items = [...builtinCompletions];

      // Add user-defined names from this document
      for (const b of bindings.values()) {
        const item = new vscode.CompletionItem(b.name, vscode.CompletionItemKind.Variable);
        item.detail = b.type;
        if (b.rhs) {
          item.documentation = new vscode.MarkdownString('```flatppl\n' + b.name + ' = ' + b.rhs + '\n```');
        }
        items.push(item);
      }

      return items;
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
    defProvider, hoverProvider, symbolProvider, completionProvider,
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
