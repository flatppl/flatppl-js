'use strict';
const vscode = require('vscode');
// The extension host loads the engine module via Node `require` for IDE
// features (diagnostics, hover, definition, rename). The visualizer panel
// runs its own copy of the same engine bundled into a browser-loadable
// IIFE (`lib/engine.min.js`); the host posts source text to the webview
// and the webview parses, builds bindings, and computes sub-DAGs locally.
// computeSubDAG is therefore no longer used here — kept the import out
// to make the change explicit if anyone ever adds it back.
//
// Load the engine from the vendored bundle (build-vendor.mjs produces it
// from packages/engine/index.js). The installed VSIX ships `lib/` but
// NOT `node_modules/` — `.vscodeignore` excludes hoisted workspace deps
// — so `require('@flatppl/engine')` would fail there with "Cannot find
// module '@flatppl/engine'". The IIFE bundle's footer wires
// `module.exports = FlatPPLEngine`, so this require returns the same
// shape as the source module's exports. Dev workflow: `npm run
// build:vendor` once after install (and after engine changes); the
// watch task keeps it fresh.
const { processSource, findBindingAtLine, builtins,
  planRename, isValidBindingName, isValidPlaceholderText,
  findEnclosingRanges, variants } = require('./lib/engine.min.js');
const { FlatPPLPanel } = require('./src/visualPanel');

// Surface-syntax variants this extension handles. Per-document
// Activation uses the document's languageId (set by VS Code from the
// `.flatppl` extension via package.json's `languages` contribution).
// There is a single canonical FlatPPL surface syntax (flatppl-design
// cc81e4b removed FlatPPY/FlatPPJ); embedded FlatPPL inside Python/
// Julia is handled by injection grammars, not separate language IDs.
const FLATPPL_LANGS = new Set(['flatppl']);
function isFlatPPLDoc(document) {
  return document != null && FLATPPL_LANGS.has(document.languageId);
}
// One canonical FlatPPL surface syntax (flatppl-design cc81e4b).
function variantIdForDoc(_document) {
  return 'flatppl';
}

// Extract embedded FlatPPL from a host (Python/Julia) document — spec
// §05 "Host-language embedding": flatppl(r"""…""") / flatppl(r'''…''')
// in Python, flatppl"""…""" / flatppl"…" (string macro) in Julia. The
// FlatPPL inside is canonical and verbatim (host interpolation is
// disallowed, so f-string prefixes are intentionally NOT matched —
// such a block isn't a valid embedding and is skipped). Returns the
// block whose content contains `cursorOffset`; else the sole block;
// else the last block starting before the cursor; else null.
//
// Nothing here runs until the user explicitly invokes a FlatPPL
// command (onCommand activation); Python/Julia users uninterested in
// FlatPPL pay nothing.
//
// `findEmbeddedBlocks` is the single pure scan — every consumer
// (visualize at cursor, the cursor follower, embedded diagnostics)
// goes through it, so the embedding-recognition rule lives in exactly
// one place and stays in lockstep with the injection grammars.
// Returns ALL blocks: { start, end, source } with absolute offsets.
function findEmbeddedBlocks(text) {
  // Group 1: Python triple/triple-single after `flatppl( [raw]?`.
  // Group 2: Julia `"""` or `"` immediately after `flatppl`.
  const re = /\bflatppl\s*(?:\(\s*(?:[rRbB]{1,2})?\s*("""|''')|("""|"))/g;
  const blocks = [];
  let m;
  while ((m = re.exec(text)) !== null) {
    const delim = m[1] || m[2];
    const start = m.index + m[0].length;
    const end = text.indexOf(delim, start);
    if (end === -1) continue;                 // unterminated — skip
    blocks.push({ start, end, source: text.slice(start, end) });
    re.lastIndex = end + delim.length;
  }
  return blocks;
}

// The block at `cursorOffset`; else the sole block; else the last
// block starting before the cursor; else null. Pure cursor-pick over
// findEmbeddedBlocks (scan and pick kept orthogonal).
function extractEmbeddedFlatPPL(text, cursorOffset) {
  const blocks = findEmbeddedBlocks(text);
  if (blocks.length === 0) return null;
  const hit = blocks.find(b => cursorOffset >= b.start && cursorOffset <= b.end);
  if (hit) return hit;
  if (blocks.length === 1) return blocks[0];
  const before = blocks.filter(b => b.start <= cursorOffset);
  return before.length ? before[before.length - 1] : blocks[0];
}

// Engine diagnostics → vscode.Diagnostic[], shifting every line by
// `lineOffset` (0 for a native .flatppl document; the host line where
// an embedded block's content starts, for embedded). Single converter
// shared by the native parse path and the embedded-diagnostics path
// so severity/range mapping isn't duplicated. Column is unshifted:
// the canonical embedding opens `"""` then a newline, so content
// lines sit at their natural columns; a same-line-as-delim opening
// would be off only in column on the first line (documented edge).
function engineToVsDiagnostics(vscode, diagnostics, lineOffset) {
  const off = lineOffset || 0;
  return diagnostics.map(d => {
    const range = new vscode.Range(
      d.loc.start.line + off, d.loc.start.col,
      d.loc.end.line + off, d.loc.end.col
    );
    const severity = d.severity === 'error'
      ? vscode.DiagnosticSeverity.Error
      : d.severity === 'warning'
      ? vscode.DiagnosticSeverity.Warning
      : vscode.DiagnosticSeverity.Information;
    return new vscode.Diagnostic(range, d.message, severity);
  });
}

// Binding the cursor is on; else the next binding at/below the cursor
// line; else the last binding; else null (only when there are none).
// One selection rule shared by the native cursor view and the
// embedded cursor follower so they can never drift apart.
function pickBindingForCursor(bindings, line, char) {
  const at = findBindingAtLine(bindings, line, char);
  if (at) return at;
  const all = [...bindings.values()];
  if (all.length === 0) return null;
  let next = null;
  for (const b of all) {
    if (b.line >= line && (!next || b.line < next.line)) next = b;
  }
  return next || all[all.length - 1];
}

// ---------------------------------------------------------------------
// Embedded LSP plumbing — make the native FlatPPL providers serve
// flatppl(…) blocks inside Python/Julia, unchanged.
//
// A native provider takes a vscode.TextDocument + Position and returns
// vscode results whose ranges are FlatPPL-source coordinates. For an
// embedded block we hand the *same* provider a minimal TextDocument
// shim backed by the extracted block text (so `parsedFor` parses just
// the block, no diagnostics/cache), translate the incoming position
// block-relative, and shift result ranges back by the block's host
// base line. One shim + one wrapper per result shape ⇒ zero provider-
// logic duplication; native and embedded can't drift.
// ---------------------------------------------------------------------

const EMBEDDING_HOST_SELECTOR = [{ language: 'python' }, { language: 'julia' }];
const _WORD_RE = /[A-Za-z_][A-Za-z0-9_]*/g;

// Minimal vscode.TextDocument over one block's source. Implements only
// what the FlatPPL providers touch: getText([range]), positionAt,
// offsetAt, lineAt, getWordRangeAtPosition, uri/version/languageId,
// plus __embeddedSource so parsedFor parses the block (not the host).
function makeBlockDoc(vscode, hostDoc, block) {
  const src = block.source;
  const lineStart = [0];
  for (let i = 0; i < src.length; i++) if (src[i] === '\n') lineStart.push(i + 1);
  const offsetAt = (pos) => {
    const ls = lineStart[Math.max(0, Math.min(pos.line, lineStart.length - 1))] || 0;
    return Math.min(ls + pos.character, src.length);
  };
  const positionAt = (off) => {
    off = Math.max(0, Math.min(off, src.length));
    let lo = 0, hi = lineStart.length - 1;
    while (lo < hi) { const m = (lo + hi + 1) >> 1; if (lineStart[m] <= off) lo = m; else hi = m - 1; }
    return new vscode.Position(lo, off - lineStart[lo]);
  };
  return {
    uri: hostDoc.uri,
    version: hostDoc.version,
    languageId: 'flatppl',
    __embeddedSource: src,
    lineCount: lineStart.length,
    offsetAt,
    positionAt,
    getText(range) {
      if (!range) return src;
      return src.slice(offsetAt(range.start), offsetAt(range.end));
    },
    lineAt(lineOrPos) {
      const line = typeof lineOrPos === 'number' ? lineOrPos : lineOrPos.line;
      const a = lineStart[line] || 0;
      const b = line + 1 < lineStart.length ? lineStart[line + 1] - 1 : src.length;
      const text = src.slice(a, b);
      return { lineNumber: line, text,
        range: new vscode.Range(line, 0, line, text.length) };
    },
    getWordRangeAtPosition(pos) {
      const lineText = this.lineAt(pos.line).text;
      _WORD_RE.lastIndex = 0;
      let m;
      while ((m = _WORD_RE.exec(lineText)) !== null) {
        if (pos.character >= m.index && pos.character <= m.index + m[0].length) {
          return new vscode.Range(pos.line, m.index, pos.line, m.index + m[0].length);
        }
      }
      return undefined;
    },
  };
}

function _shiftRange(vscode, r, dl) {
  return new vscode.Range(
    r.start.line + dl, r.start.character, r.end.line + dl, r.end.character);
}

// Block whose content contains `position` (host coords) + the host
// line where it starts. Null when the cursor isn't in any block.
function blockAt(hostDoc, position) {
  const blocks = findEmbeddedBlocks(hostDoc.getText());
  if (blocks.length === 0) return null;
  const off = hostDoc.offsetAt(position);
  const b = blocks.find(x => off >= x.start && off <= x.end);
  if (!b) return null;
  return { block: b, baseLine: hostDoc.positionAt(b.start).line };
}

// Embedded wrapper for a position-based provider. `remap(vscode,
// result, baseLine)` shifts result ranges to host coords (identity for
// range-less results like completion items). Returns null/[] outside a
// block so the host language's own provider is unaffected.
function embedPositional(vscode, impl, method, remap, emptyValue) {
  return {
    [method](document, position, ...rest) {
      const at = blockAt(document, position);
      if (!at) return emptyValue !== undefined ? emptyValue : null;
      const vdoc = makeBlockDoc(vscode, document, at.block);
      const vpos = new vscode.Position(
        position.line - at.baseLine, position.character);
      const res = impl[method](vdoc, vpos, ...rest);
      return res == null ? res : remap(vscode, res, at.baseLine);
    },
  };
}

const remapHover = (vscode, h, dl) => new vscode.Hover(
  h.contents, h.range ? _shiftRange(vscode, h.range, dl) : undefined);

function _remapSymbol(vscode, s, dl) {
  const out = new vscode.DocumentSymbol(
    s.name, s.detail, s.kind,
    _shiftRange(vscode, s.range, dl),
    _shiftRange(vscode, s.selectionRange, dl));
  if (s.children && s.children.length) {
    out.children = s.children.map(c => _remapSymbol(vscode, c, dl));
  }
  return out;
}

function _remapSelectionRange(vscode, sr, dl) {
  return new vscode.SelectionRange(
    _shiftRange(vscode, sr.range, dl),
    sr.parent ? _remapSelectionRange(vscode, sr.parent, dl) : undefined);
}

// Location.uri is already the host uri (the shim's uri === hostDoc's);
// only the range needs shifting back into host coordinates.
const _remapLocation = (vscode, loc, dl) =>
  new vscode.Location(loc.uri, _shiftRange(vscode, loc.range, dl));

// definition → Location | Location[] (our impl: single or null).
const remapDefinition = (vscode, res, dl) => Array.isArray(res)
  ? res.map(l => _remapLocation(vscode, l, dl))
  : _remapLocation(vscode, res, dl);

const remapReferences = (vscode, res, dl) =>
  res.map(l => _remapLocation(vscode, l, dl));

const remapHighlights = (vscode, res, dl) => res.map(h =>
  new vscode.DocumentHighlight(_shiftRange(vscode, h.range, dl), h.kind));

// prepareRename → { range, placeholder }: shift the range, keep the
// placeholder text.
const remapPrepareRename = (vscode, res, dl) => ({
  range: _shiftRange(vscode, res.range, dl),
  placeholder: res.placeholder,
});

// Rebuild a WorkspaceEdit shifting every TextEdit's range to host
// coordinates. The uri is already the host doc's (the shim's uri ===
// hostDoc's), so only the line offset changes — the rename writes
// back into the .py/.jl at the correct lines within the block.
function remapWorkspaceEdit(vscode, edit, dl) {
  const out = new vscode.WorkspaceEdit();
  for (const [uri, edits] of edit.entries()) {
    for (const te of edits) {
      out.replace(uri, _shiftRange(vscode, te.range, dl), te.newText);
    }
  }
  return out;
}

function activate(context) {
  // Cache parsed results to avoid re-parsing on every cursor move
  let cachedUri = '';
  let cachedVersion = -1;
  let cachedResult = null;

  // Diagnostic collection for FlatPPL errors/warnings
  const diagCollection = vscode.languages.createDiagnosticCollection('flatppl');
  context.subscriptions.push(diagCollection);

  // Embedded-support menus/buttons gate on this; explicitly false so
  // the Activate command's `!flatppl.embeddedActive` palette gate is
  // correct before anything is armed.
  vscode.commands.executeCommand('setContext', 'flatppl.embeddedActive', false);

  function getParsed(document) {
    const uri = document.uri.toString();
    if (uri === cachedUri && document.version === cachedVersion) return cachedResult;

    const source = document.getText();
    const { ast, bindings, symbols, diagnostics } = processSource(source,
      { variant: variantIdForDoc(document) });
    cachedResult = { ast, bindings, symbols, diagnostics };
    cachedVersion = document.version;
    cachedUri = uri;

    // Update diagnostics in VS Code (native: no line shift).
    diagCollection.set(document.uri,
      engineToVsDiagnostics(vscode, diagnostics, 0));

    return cachedResult;
  }

  // Parse source for a language provider. A real .flatppl document
  // goes through getParsed (uri/version cache + diagnostic publish).
  // A virtual embedded-block document (carries __embeddedSource — the
  // extracted FlatPPL of one flatppl(…) block) is parsed directly with
  // no cache write and no diagnostic publish (embedded diagnostics are
  // owned by publishEmbeddedDiagnostics). This single indirection lets
  // every native provider serve embedded blocks unchanged — the only
  // edit to a provider body is getParsed → parsedFor.
  function parsedFor(document) {
    if (document && typeof document.__embeddedSource === 'string') {
      return processSource(document.__embeddedSource, { variant: 'flatppl' });
    }
    return getParsed(document);
  }

  // Show the visualizer for whichever binding contains the cursor (or
  // fall back to the last binding if the cursor is on a comment / blank
  // line). The host's only jobs now are:
  //   - finding the cursor's binding name (uses analyzer, host-side)
  //   - posting source + targetName to the webview, which parses + renders
  //
  // No more `computeSubDAG` here — the webview owns that.
  // pushHistory is true for navigation events (cursor moves driven by
  // editor selection, ctrl-click in the DAG that hops the cursor),
  // false for the initial open or other "fresh start" cases. The
  // webview's focusNode only pushes when the target actually changes,
  // so passing true here is safe even when the cursor lands on the
  // same binding.
  // One panel-drive used by every surface (native binding/module,
  // embedded). createOrShow + first-time config + the right panel
  // call; the readOnly/navOrigin opts flow straight through.
  // opts: { source, mode:'binding'|'module', targetName?, sourceUri?,
  //         readOnly?, navOrigin?, pushHistory? }
  function renderToPanel(opts) {
    const wasNew = !FlatPPLPanel.currentPanel;
    FlatPPLPanel.createOrShow(context);
    if (wasNew) {
      // Push initial config alongside source so the webview's sample
      // cache and SAMPLE_COUNT are correct on the first plot request.
      FlatPPLPanel.currentPanel.updateConfig(readVisualizationConfig());
    }
    if (opts.mode === 'module') {
      FlatPPLPanel.currentPanel.showModule(
        opts.source, opts.sourceUri || null, !!opts.pushHistory,
        !!opts.readOnly, opts.navOrigin || null);
    } else {
      FlatPPLPanel.currentPanel.updateSource(
        opts.source, opts.targetName || null, opts.sourceUri || null,
        !!opts.pushHistory,
        opts.readOnly ? { readOnly: true, navOrigin: opts.navOrigin || null }
                      : undefined);
    }
  }

  function showDAGForCursor(editor, pushHistory) {
    if (!editor || !isFlatPPLDoc(editor.document)) return false;
    const { bindings } = getParsed(editor.document);
    const pos = editor.selection.active;
    const binding = pickBindingForCursor(bindings, pos.line, pos.character);
    if (!binding) return false;
    renderToPanel({
      source: editor.document.getText(), mode: 'binding',
      targetName: binding.name, sourceUri: editor.document.uri,
      pushHistory: !!pushHistory,
    });
    return true;
  }


  // Read the current visualization-related settings into a plain
  // object that can be postMessage'd to the webview. Centralised here
  // so initial-load and onDidChangeConfiguration share one shape.
  function readVisualizationConfig() {
    const cfg = vscode.workspace.getConfiguration('flatppl.visualization');
    return {
      sampleCount: cfg.get('sampleCount', 100000),
      dagNavigationHistoryCap: cfg.get('dagNavigationHistoryCap', 1000),
      truncateRejectionBudget: cfg.get('truncateRejectionBudget', 1000),
    };
  }

  // --- Embedded support: session-scoped, opt-in lifecycle ---
  //
  // Nothing here runs until the user invokes a FlatPPL command in a
  // Python/Julia file (or the explicit Activate command). Arming sets
  // the `flatppl.embeddedActive` context key (menus/buttons gate on
  // it) and registers the on-demand bundle (cursor follower now;
  // diagnostics added in a later commit). Disarm tears it all down.
  // So Python/Julia users who never opt in pay nothing.
  let embeddedArmed = false;
  let embeddedDisposables = [];
  let embeddedFollow;          // the single active cursor-follow listener
  let embeddedFollowTimer;
  const embeddedDiagTimers = new Map();   // uriString → debounce timeout

  function isEmbeddingHost(doc) {
    return doc && (doc.languageId === 'python' || doc.languageId === 'julia');
  }

  // Parse every flatppl(…) block in a Python/Julia host doc and
  // publish FlatPPL diagnostics into the shared collection, shifted to
  // host coordinates. Each block is its own module (spec §05), so they
  // are parsed independently and their diagnostics unioned. Same
  // diagCollection as native .flatppl — keyed by host uri, disjoint
  // from .flatppl uris, so no conflict. No blocks ⇒ clear (stale).
  function publishEmbeddedDiagnostics(doc) {
    if (!isEmbeddingHost(doc)) return;
    const text = doc.getText();
    const blocks = findEmbeddedBlocks(text);
    if (blocks.length === 0) { diagCollection.delete(doc.uri); return; }
    const out = [];
    for (const block of blocks) {
      const baseLine = doc.positionAt(block.start).line;
      let diags;
      try {
        diags = processSource(block.source, { variant: 'flatppl' }).diagnostics;
      } catch (_) { continue; }               // never throw from a listener
      for (const d of engineToVsDiagnostics(vscode, diags, baseLine)) {
        out.push(d);
      }
    }
    diagCollection.set(doc.uri, out);
  }

  // Debounced per-doc refresh (cheap regex scan + parse only when
  // armed; opt-in, so uninterested users never reach here).
  function scheduleEmbeddedDiagnostics(doc) {
    if (!isEmbeddingHost(doc)) return;
    const key = doc.uri.toString();
    clearTimeout(embeddedDiagTimers.get(key));
    embeddedDiagTimers.set(key, setTimeout(() => {
      embeddedDiagTimers.delete(key);
      publishEmbeddedDiagnostics(doc);
    }, 250));
  }

  function clearAllEmbeddedDiagnostics() {
    for (const t of embeddedDiagTimers.values()) clearTimeout(t);
    embeddedDiagTimers.clear();
    for (const d of vscode.workspace.textDocuments) {
      if (isEmbeddingHost(d)) diagCollection.delete(d.uri);
    }
  }

  function armEmbeddedSupport() {
    if (embeddedArmed) return false;
    embeddedArmed = true;
    vscode.commands.executeCommand('setContext', 'flatppl.embeddedActive', true);

    // Embedded LSP bundle — registered lazily (only now that the user
    // opted in) and torn down by disarm. Same native provider impls,
    // served over a block shim; results shifted to host coords.
    const EMB = EMBEDDING_HOST_SELECTOR;
    embeddedDisposables.push(
      // Diagnostics.
      vscode.workspace.onDidChangeTextDocument(e => scheduleEmbeddedDiagnostics(e.document)),
      vscode.workspace.onDidOpenTextDocument(d => publishEmbeddedDiagnostics(d)),
      vscode.workspace.onDidCloseTextDocument(d => {
        if (isEmbeddingHost(d)) diagCollection.delete(d.uri);
      }),
      // Position-based, reuse impls via the shim.
      vscode.languages.registerHoverProvider(EMB,
        embedPositional(vscode, hoverImpl, 'provideHover', remapHover)),
      vscode.languages.registerCompletionItemProvider(EMB,
        embedPositional(vscode, completionImpl, 'provideCompletionItems',
          (_v, r) => r, /* emptyValue (additive, don't suppress host) */ [])),
      vscode.languages.registerDefinitionProvider(EMB,
        embedPositional(vscode, defImpl, 'provideDefinition', remapDefinition)),
      vscode.languages.registerReferenceProvider(EMB,
        embedPositional(vscode, referenceImpl, 'provideReferences', remapReferences)),
      vscode.languages.registerDocumentHighlightProvider(EMB,
        embedPositional(vscode, highlightImpl, 'provideDocumentHighlights', remapHighlights)),
      // Rename: two methods on one provider; the edit writes back into
      // the host .py/.jl at block-shifted ranges. Outside a block we
      // decline (undefined/null) so the host language's rename runs.
      vscode.languages.registerRenameProvider(EMB, {
        prepareRename(document, position) {
          const at = blockAt(document, position);
          if (!at) return undefined;
          const vdoc = makeBlockDoc(vscode, document, at.block);
          const vpos = new vscode.Position(
            position.line - at.baseLine, position.character);
          const res = renameImpl.prepareRename(vdoc, vpos);  // may throw
          return res == null ? res
            : remapPrepareRename(vscode, res, at.baseLine);
        },
        provideRenameEdits(document, position, newName) {
          const at = blockAt(document, position);
          if (!at) return null;
          const vdoc = makeBlockDoc(vscode, document, at.block);
          const vpos = new vscode.Position(
            position.line - at.baseLine, position.character);
          const edit = renameImpl.provideRenameEdits(vdoc, vpos, newName);
          return edit == null ? edit
            : remapWorkspaceEdit(vscode, edit, at.baseLine);
        },
      }),
      // Document symbols: no position — union every block's outline,
      // each shifted by its own host base line.
      vscode.languages.registerDocumentSymbolProvider(EMB, {
        provideDocumentSymbols(document) {
          const out = [];
          for (const b of findEmbeddedBlocks(document.getText())) {
            const baseLine = document.positionAt(b.start).line;
            const vdoc = makeBlockDoc(vscode, document, b);
            for (const s of symbolImpl.provideDocumentSymbols(vdoc) || []) {
              out.push(_remapSymbol(vscode, s, baseLine));
            }
          }
          return out;
        },
      }),
      // Selection range: per requested position, in its own block.
      vscode.languages.registerSelectionRangeProvider(EMB, {
        provideSelectionRanges(document, positions) {
          return positions.map(pos => {
            const at = blockAt(document, pos);
            if (!at) return null;
            const vdoc = makeBlockDoc(vscode, document, at.block);
            const vpos = new vscode.Position(
              pos.line - at.baseLine, pos.character);
            const r = selectionRangeImpl.provideSelectionRanges(vdoc, [vpos]);
            if (!r || !r[0]) return null;
            return _remapSelectionRange(vscode, r[0], at.baseLine);
          }).filter(Boolean);
        },
      }),
    );
    // Catch up: lint already-open host docs.
    for (const d of vscode.workspace.textDocuments) publishEmbeddedDiagnostics(d);
    vscode.window.showInformationMessage(
      'FlatPPL: embedded support enabled for this window — Visualize '
      + 'Binding / Model now work inside flatppl(…) blocks in '
      + 'Python/Julia files.');
    return true;
  }

  function disarmEmbeddedSupport() {
    if (!embeddedArmed) return;
    embeddedArmed = false;
    vscode.commands.executeCommand('setContext', 'flatppl.embeddedActive', false);
    if (embeddedFollow) { embeddedFollow.dispose(); embeddedFollow = undefined; }
    for (const d of embeddedDisposables) { try { d.dispose(); } catch (_) {} }
    embeddedDisposables = [];
    clearAllEmbeddedDiagnostics();   // remove host-doc squiggles too
  }

  // Resolve what to render for `editor` in `mode` ('binding'|'module')
  // across native .flatppl and embedded Python/Julia — parsing only,
  // no side effects (arming/following is runViz's job). One resolver
  // ⇒ the two surfaces and the follower can't drift.
  function resolveVizTarget(editor, mode) {
    if (!editor) return { kind: 'none', error: 'No active editor.' };
    const doc = editor.document;
    if (isFlatPPLDoc(doc)) {
      const { bindings } = getParsed(doc);
      if (bindings.size === 0) {
        return { kind: 'none', error: 'No bindings to visualize in this file.' };
      }
      const pos = editor.selection.active;
      const b = mode === 'binding'
        ? pickBindingForCursor(bindings, pos.line, pos.character) : null;
      return { kind: 'native', opts: {
        source: doc.getText(), mode, targetName: b ? b.name : null,
        sourceUri: doc.uri, readOnly: false, pushHistory: true } };
    }
    const lang = doc.languageId;
    if (lang === 'python' || lang === 'julia') {
      const block = extractEmbeddedFlatPPL(
        doc.getText(), doc.offsetAt(editor.selection.active));
      if (!block) {
        return { kind: 'none', error:
          'No embedded FlatPPL at the cursor — expected flatppl(r"""…""") '
          + '(Python) or flatppl"""…""" (Julia).' };
      }
      const baseLine = doc.positionAt(block.start).line;
      let targetName = null;
      if (mode === 'binding') {
        const { bindings } = processSource(block.source, { variant: 'flatppl' });
        const pos = editor.selection.active;
        const b = pickBindingForCursor(
          bindings, pos.line - baseLine, pos.character);
        targetName = b ? b.name : null;
      }
      return { kind: 'embedded', hostUri: doc.uri, opts: {
        source: block.source, mode, targetName,
        sourceUri: null, readOnly: true,
        navOrigin: { uri: doc.uri, baseLine }, pushHistory: true } };
    }
    return { kind: 'none', error:
      'Open a .flatppl file, or a Python/Julia file with embedded FlatPPL.' };
  }

  // (Re)arm the embedded source→DAG cursor follower for `hostUri`.
  // Lazy (user already opted in), self-disposes when the panel is
  // gone, replaced on re-arm. Reuses resolveVizTarget + renderToPanel
  // so the follow path is identical to the command path.
  function armEmbeddedFollower(hostUri) {
    const hostUriStr = hostUri.toString();
    if (embeddedFollow) embeddedFollow.dispose();
    let lastName = ' ';
    embeddedFollow = vscode.window.onDidChangeTextEditorSelection(ev => {
      if (!FlatPPLPanel.currentPanel) {
        embeddedFollow.dispose(); embeddedFollow = undefined; return;
      }
      if (ev.textEditor.document.uri.toString() !== hostUriStr) return;
      const k = ev.kind;
      const KIND = vscode.TextEditorSelectionChangeKind;
      if (k !== KIND.Keyboard && k !== KIND.Mouse) return;
      clearTimeout(embeddedFollowTimer);
      embeddedFollowTimer = setTimeout(() => {
        if (!FlatPPLPanel.currentPanel) return;
        const r = resolveVizTarget(ev.textEditor, 'binding');
        if (r.kind !== 'embedded') return;     // cursor left every block
        if (r.opts.targetName === lastName) return;
        lastName = r.opts.targetName;
        renderToPanel(r.opts);
      }, 150);
    });
    embeddedDisposables.push(embeddedFollow);
  }

  // Shared body of the two visualize commands. Native and embedded
  // differ only in resolveVizTarget; embedded additionally auto-arms
  // and follows.
  function runViz(mode) {
    const r = resolveVizTarget(vscode.window.activeTextEditor, mode);
    if (r.kind === 'none') {
      vscode.window.showInformationMessage(r.error);
      return;
    }
    if (r.kind === 'embedded') {
      armEmbeddedSupport();                    // first embedded use arms
      armEmbeddedFollower(r.hostUri);
    }
    renderToPanel(r.opts);
  }

  // --- Commands ---

  const showDagCmd = vscode.commands.registerCommand(
    'flatppl.visualize', () => runViz('binding'));

  const showModuleCmd = vscode.commands.registerCommand(
    'flatppl.visualizeModule', () => runViz('module'));

  const activateEmbeddedCmd = vscode.commands.registerCommand(
    'flatppl.activateEmbedded', () => {
      if (!armEmbeddedSupport()) {
        vscode.window.showInformationMessage(
          'FlatPPL embedded support is already enabled for this window.');
      }
    });

  const deactivateEmbeddedCmd = vscode.commands.registerCommand(
    'flatppl.deactivateEmbedded', () => {
      disarmEmbeddedSupport();
      vscode.window.showInformationMessage(
        'FlatPPL embedded support disabled for this window.');
    });

  // --- Live DAG update on cursor move ---

  let updateTimeout;
  let lastShownName = '';

  const selectionListener = vscode.window.onDidChangeTextEditorSelection(e => {
    if (!FlatPPLPanel.currentPanel) return;
    if (!isFlatPPLDoc(e.textEditor.document)) return;

    // Only react to user-driven cursor moves (Keyboard / Mouse).
    // Programmatic edits — including the WorkspaceEdit applied by
    // the panel's persistPreset handler — emit selection-change
    // events with kind=Command or kind=undefined. Treating those
    // as cursor navigation here would push the cursor's current
    // binding back to the webview, overriding whatever node the
    // user was focused on in the visualizer (e.g. they clicked
    // Persist while viewing forward_kernel; without this gate,
    // focus jumps to wherever the editor cursor happened to be).
    const k = e.kind;
    const KIND = vscode.TextEditorSelectionChangeKind;
    if (k !== KIND.Keyboard && k !== KIND.Mouse) return;

    clearTimeout(updateTimeout);
    updateTimeout = setTimeout(() => {
      const { bindings } = getParsed(e.textEditor.document);
      const pos = e.selections[0].active;
      const binding = findBindingAtLine(bindings, pos.line, pos.character);
      if (binding && binding.name !== lastShownName) {
        lastShownName = binding.name;
        // Cursor-driven navigation: push onto the visualizer's
        // history so the user can step back through their path.
        showDAGForCursor(e.textEditor, /* pushHistory */ true);
      }
    }, 150);
  });

  // --- Re-parse on document change (for diagnostics + visualizer) ---

  let changePushTimeout;

  const changeListener = vscode.workspace.onDidChangeTextDocument(e => {
    if (!isFlatPPLDoc(e.document)) return;
    getParsed(e.document);

    // Also push the updated source to the visualizer if it's open.
    // Without this, editing a binding's RHS doesn't refresh the
    // panel — the selection listener only fires on cursor moves and
    // gates on binding-name change, so RHS-only edits get missed.
    // Pass targetName=null so the webview keeps its current focus
    // (the user is editing the binding they're looking at; they
    // don't want their place reset).
    if (!FlatPPLPanel.currentPanel) return;
    const editor = vscode.window.activeTextEditor;
    if (!editor || editor.document !== e.document) return;
    clearTimeout(changePushTimeout);
    changePushTimeout = setTimeout(() => {
      FlatPPLPanel.currentPanel.updateSource(
        editor.document.getText(), null, editor.document.uri, /* pushHistory */ false);
    }, 200);
  });

  // --- Parse on open ---

  const openListener = vscode.workspace.onDidOpenTextDocument(doc => {
    if (isFlatPPLDoc(doc)) {
      getParsed(doc);
    }
  });

  // --- Go-to-definition ---

  const defImpl = {
    provideDefinition(document, position) {
      const { bindings } = parsedFor(document);
      const wordRange = document.getWordRangeAtPosition(position);
      if (!wordRange) return null;
      const word = document.getText(wordRange);
      const binding = bindings.get(word);
      if (!binding) return null;
      return new vscode.Location(document.uri,
        new vscode.Position(binding.nameLoc.start.line, binding.nameLoc.start.col));
    }
  };
  const defProvider = vscode.languages.registerDefinitionProvider(
    [...FLATPPL_LANGS], defImpl);

  // --- Hover ---

  const hoverImpl = {
    provideHover(document, position) {
      const { bindings } = parsedFor(document);
      const wordRange = document.getWordRangeAtPosition(position);
      if (!wordRange) return null;
      const word = document.getText(wordRange);
      const binding = bindings.get(word);
      if (!binding) return null;

      const phaseStr = binding.phase ? `, phase: *${binding.phase}*` : '';
      const lines = [
        `**${binding.name}** \u2014 *${binding.type}*${phaseStr}`,
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
  };
  const hoverProvider = vscode.languages.registerHoverProvider(
    [...FLATPPL_LANGS], hoverImpl);

  // --- Document symbols (outline) ---

  const symbolImpl = {
    provideDocumentSymbols(document) {
      const { symbols } = parsedFor(document);
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
  };
  const symbolProvider = vscode.languages.registerDocumentSymbolProvider(
    [...FLATPPL_LANGS], symbolImpl);

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

  // Snippets for special operations — drop in a body placeholder.
  const SPECIAL_SNIPPETS = {
    elementof: 'elementof(${1:reals})',
    external: 'external(${1:reals})',
    draw: 'draw(${1})',
    lawof: 'lawof(${1})',
    functionof: 'functionof(${1})',
    kernelof: 'kernelof(${1})',
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
    for (const name of builtins.SPECIAL_OPERATIONS) {
      const item = new vscode.CompletionItem(name, vscode.CompletionItemKind.Keyword);
      const snippet = SPECIAL_SNIPPETS[name];
      if (snippet) {
        item.insertText = new vscode.SnippetString(snippet);
      }
      item.detail = 'special operation';
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

  const completionImpl = {
    provideCompletionItems(document, position) {
      const { bindings } = parsedFor(document);
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
  };
  const completionProvider = vscode.languages.registerCompletionItemProvider(
    [...FLATPPL_LANGS], completionImpl);

  // --- Rename provider (F2) ---

  function locToRange(loc) {
    return new vscode.Range(
      loc.start.line, loc.start.col,
      loc.end.line, loc.end.col
    );
  }

  const renameImpl = {
    prepareRename(document, position) {
      const { ast, bindings } = parsedFor(document);
      const plan = planRename(ast, bindings, position.line, position.character);
      if (!plan) {
        // Throw to give VS Code a clear "not renameable" signal.
        throw new Error("This element cannot be renamed");
      }
      return {
        range: locToRange(plan.targetLoc),
        placeholder: plan.kind === 'placeholder'
          ? '_' + plan.oldName + '_'
          : plan.oldName,
      };
    },

    provideRenameEdits(document, position, newName) {
      const { ast, bindings } = parsedFor(document);
      const plan = planRename(ast, bindings, position.line, position.character);
      if (!plan) return null;

      // Validate new name based on rename kind.
      if (plan.kind === 'binding') {
        if (!isValidBindingName(newName)) {
          throw new Error(`'${newName}' is not a valid binding name`);
        }
        // Reject conflicts with existing bindings (except the one being renamed).
        if (newName !== plan.oldName && bindings.has(newName)) {
          throw new Error(`Name '${newName}' is already defined`);
        }
        // Reject built-in shadowing? Spec allows it ("built-ins are shadowable"),
        // so we don't block it here — the user is intentionally shadowing.
      } else if (plan.kind === 'placeholder') {
        if (!isValidPlaceholderText(newName)) {
          throw new Error(`'${newName}' is not a valid placeholder (must look like _name_)`);
        }
      }

      const edit = new vscode.WorkspaceEdit();
      const replaceText = newName; // already includes underscores for placeholder case
      for (const loc of plan.locs) {
        edit.replace(document.uri, locToRange(loc), replaceText);
      }
      return edit;
    }
  };
  const renameProvider = vscode.languages.registerRenameProvider(
    [...FLATPPL_LANGS], renameImpl);

  // --- Find All References (Shift+F12) ---

  const referenceImpl = {
    provideReferences(document, position, refContext) {
      const { ast, bindings } = parsedFor(document);
      const plan = planRename(ast, bindings, position.line, position.character);
      if (!plan) return null;
      // For binding renames, locs[0] is the LHS definition and the rest are
      // references. Honour includeDeclaration accordingly.
      let locs = plan.locs;
      if (plan.kind === 'binding' && refContext && refContext.includeDeclaration === false) {
        locs = locs.slice(1);
      }
      return locs.map(loc => new vscode.Location(document.uri, locToRange(loc)));
    }
  };
  const referenceProvider = vscode.languages.registerReferenceProvider(
    [...FLATPPL_LANGS], referenceImpl);

  // --- Document Highlight (auto, when cursor is on an identifier) ---

  const highlightImpl = {
    provideDocumentHighlights(document, position) {
      const { ast, bindings } = parsedFor(document);
      const plan = planRename(ast, bindings, position.line, position.character);
      if (!plan) return null;
      return plan.locs.map((loc, i) => {
        // For bindings: first loc is the LHS definition (Write); rest are refs (Read).
        // For placeholders: all are textually equivalent (Text).
        let kind = vscode.DocumentHighlightKind.Text;
        if (plan.kind === 'binding') {
          kind = (i === 0)
            ? vscode.DocumentHighlightKind.Write
            : vscode.DocumentHighlightKind.Read;
        }
        return new vscode.DocumentHighlight(locToRange(loc), kind);
      });
    }
  };
  const highlightProvider = vscode.languages.registerDocumentHighlightProvider(
    [...FLATPPL_LANGS], highlightImpl);

  // --- Selection Range (Shift+Alt+→ to expand selection) ---

  const selectionRangeImpl = {
    provideSelectionRanges(document, positions) {
      const { ast } = parsedFor(document);
      return positions.map(pos => {
        const ranges = findEnclosingRanges(ast, pos.line, pos.character);
        if (ranges.length === 0) {
          // Fall back to the word range so VS Code's default still works.
          const wordRange = document.getWordRangeAtPosition(pos);
          return wordRange ? new vscode.SelectionRange(wordRange) : null;
        }
        // Build a parent chain from outermost down to innermost.
        let current = null;
        for (let i = ranges.length - 1; i >= 0; i--) {
          current = new vscode.SelectionRange(locToRange(ranges[i]), current);
        }
        return current;
      }).filter(Boolean);
    }
  };
  const selectionRangeProvider = vscode.languages.registerSelectionRangeProvider(
    [...FLATPPL_LANGS], selectionRangeImpl);

  // --- Clean up diagnostics on close ---

  const closeListener = vscode.workspace.onDidCloseTextDocument(doc => {
    diagCollection.delete(doc.uri);
  });

  // Parse any already-open FlatPPL documents
  for (const doc of vscode.workspace.textDocuments) {
    if (isFlatPPLDoc(doc)) getParsed(doc);
  }

  // Push configuration changes to a live panel. Filter to our own
  // namespace so unrelated settings don't trigger a webview round-trip.
  const configListener = vscode.workspace.onDidChangeConfiguration(e => {
    if (!e.affectsConfiguration('flatppl.visualization')) return;
    if (FlatPPLPanel.currentPanel) {
      FlatPPLPanel.currentPanel.updateConfig(readVisualizationConfig());
    }
  });

  context.subscriptions.push(
    showDagCmd, showModuleCmd, activateEmbeddedCmd, deactivateEmbeddedCmd,
    selectionListener, changeListener, openListener, closeListener,
    defProvider, hoverProvider, symbolProvider, completionProvider,
    renameProvider, referenceProvider, highlightProvider, selectionRangeProvider,
    configListener,
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
