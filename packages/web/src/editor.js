// @flatppl/web — playground editor.
//
// FlatPPL-specific glue around CodeMirror 6 (loaded lazily from
// vendor/codemirror.min.js). Exposes window.FlatPPLWebEditor
// with two functions:
//
//   loadBundle(): Promise<bundle>
//     Lazy-loads the CodeMirror bundle if not already present.
//     Resolves with window.FlatPPLEditorBundle. Idempotent.
//
//   mountEditor(container, opts): EditorHandle
//     Builds a CodeMirror editor inside `container`, wires the
//     FlatPPL syntax-highlight plugin (which reuses the engine's
//     tokenizer and the same `tok-*` CSS classes the read-only
//     pane uses), and returns a small handle:
//       { setSource(text), getSource(), destroy() }
//     opts:
//       initialSource (string, '')
//       onChange(text) — fired on every document change.
//                        The caller is expected to debounce and
//                        re-trigger viewer.update.
//       onNavigate(name) — fired when the user Ctrl/Cmd-clicks an
//                          identifier whose name is a defined
//                          binding. Plain click is the editor's
//                          cursor placement (default behaviour);
//                          the modifier reserves "navigate to
//                          DAG" without fighting the cursor.
//
// Highlight plugin: a ViewPlugin walks the tokenizer output on
// every doc change, and emits Decoration.mark for each token with
// the same class names as the static highlighter (`tok-keyword`,
// `tok-ident-binding`, etc.). The CSS already in style.css covers
// both surfaces — read-only `<pre>` for non-playground deploys
// and the editor for playground deploys — without duplication.

'use strict';

(function (globalScope) {
  var BUNDLE_URL = 'vendor/codemirror.min.js';
  var loadPromise = null;

  function loadBundle() {
    if (globalScope.FlatPPLEditorBundle) return Promise.resolve(globalScope.FlatPPLEditorBundle);
    if (loadPromise) return loadPromise;
    loadPromise = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = BUNDLE_URL;
      s.async = true;
      s.onload = function () {
        if (globalScope.FlatPPLEditorBundle) resolve(globalScope.FlatPPLEditorBundle);
        else reject(new Error('CodeMirror bundle loaded but FlatPPLEditorBundle was not set'));
      };
      s.onerror = function () { reject(new Error('Failed to load ' + BUNDLE_URL)); };
      document.head.appendChild(s);
    });
    return loadPromise;
  }

  // Mirrors syntax.js classifyIdentifier — kept independent so
  // editor.js doesn't depend on syntax.js (which targets the
  // read-only pane). Same logic, different output sink.
  function classifyIdentifier(name, bindings, B) {
    if (bindings && bindings.has(name)) return 'tok-ident-binding';
    if (B.isSpecialOperation(name))     return 'tok-special';
    if (B.MEASURE_OPS.has(name))        return 'tok-mop';
    if (B.DISTRIBUTIONS.has(name))      return 'tok-dist';
    if (B.SET_CONSTRUCTORS.has(name))   return 'tok-set';
    if (B.isSet(name))                  return 'tok-set';
    if (B.BUILTIN_FUNCTIONS.has(name))  return 'tok-func';
    if (B.isConstant(name))             return 'tok-const';
    if (B.isReserved(name))             return 'tok-reserved';
    if (B.SPECIAL_BINDINGS.has(name))   return 'tok-reserved';
    return 'tok-ident';
  }

  function classifyToken(tok) {
    var t = tok.type;
    if (t === 'COMMENT')     return 'tok-comment';
    if (t === 'STRING')      return 'tok-string';
    if (t === 'NUMBER')      return 'tok-number';
    if (t === 'PLACEHOLDER') return 'tok-placeholder';
    if (t === 'HOLE')        return 'tok-hole';
    if (t === 'EQUALS' || t === 'EQEQ' || t === 'NEQ' ||
        t === 'LT' || t === 'GT' || t === 'LTE' || t === 'GTE' ||
        t === 'PLUS' || t === 'MINUS' || t === 'STAR' || t === 'SLASH') {
      return 'tok-op';
    }
    return 'tok-punct';
  }

  function computeLineStarts(src) {
    var starts = [0];
    for (var i = 0; i < src.length; i++) {
      if (src.charCodeAt(i) === 10 /* \n */) starts.push(i + 1);
    }
    return starts;
  }

  function offsetOf(loc, lineStarts) {
    var ls = lineStarts[loc.line];
    return (typeof ls === 'number' ? ls : 0) + loc.col;
  }

  /** Build the FlatPPL highlight ViewPlugin. Closure over the
      bundle so this module doesn't import CodeMirror directly
      (which would force the gallery to depend on it even in
      non-playground mode). */
  function makeHighlightPlugin(bundle) {
    var ViewPlugin = bundle.ViewPlugin;
    var Decoration = bundle.Decoration;
    var FE = globalScope.FlatPPLEngine;
    var B = FE && FE.builtins;

    function buildDecorations(view) {
      if (!FE || !B) return Decoration.none;
      var text = view.state.doc.toString();
      var bindings = null;
      try {
        var processed = FE.processSource(text);
        if (processed && processed.bindings) {
          bindings = new Set(processed.bindings.keys());
        }
      } catch (_) { bindings = null; }

      var tokens = FE.tokenize(text).tokens || [];
      var lineStarts = computeLineStarts(text);
      var ranges = [];
      for (var i = 0; i < tokens.length; i++) {
        var tok = tokens[i];
        if (tok.type === 'EOF' || tok.type === 'NEWLINE') continue;
        var from = offsetOf(tok.loc.start, lineStarts);
        var to   = offsetOf(tok.loc.end,   lineStarts);
        if (to <= from) continue;

        var cls;
        var attrs = null;
        if (tok.type === 'IDENT') {
          cls = classifyIdentifier(tok.value, bindings, B);
          if (bindings && bindings.has(tok.value)) {
            attrs = { 'data-binding': tok.value };
          }
        } else {
          cls = classifyToken(tok);
        }
        ranges.push(
          Decoration.mark({
            class: cls,
            attributes: attrs || undefined,
          }).range(from, to)
        );
      }
      return Decoration.set(ranges, true);
    }

    return ViewPlugin.fromClass(
      function (view) {
        this.decorations = buildDecorations(view);
        this.update = function (u) {
          if (u.docChanged || u.viewportChanged) {
            this.decorations = buildDecorations(u.view);
          }
        };
      },
      { decorations: function (v) { return v.decorations; } }
    );
  }

  /** Build a small EditorView.theme matching the gallery's dark
      palette so the editor blends with the surrounding panes
      instead of importing a separate CodeMirror theme. */
  function makeTheme(bundle) {
    return bundle.EditorView.theme({
      '&': {
        height: '100%',
        fontSize: '13px',
        backgroundColor: '#252526',
        color: '#cccccc',
      },
      '.cm-scroller': {
        fontFamily: "ui-monospace, 'Cascadia Code', 'JetBrains Mono', 'Source Code Pro', Menlo, Consolas, monospace",
        lineHeight: '1.45',
      },
      '.cm-content':  { caretColor: '#cccccc' },
      '.cm-gutters':  {
        backgroundColor: '#252526',
        borderRight: '1px solid #3c3c3c',
        color: '#858585',
      },
      '.cm-activeLine':       { backgroundColor: 'rgba(255,255,255,0.06)' },
      '.cm-activeLineGutter': { backgroundColor: 'rgba(255,255,255,0.08)' },
      '&.cm-focused .cm-cursor': { borderLeftColor: '#cccccc' },
      '&.cm-focused .cm-selectionBackground, ::selection': {
        backgroundColor: '#264f78',
      },
    }, { dark: true });
  }

  function mountEditor(container, opts) {
    opts = opts || {};
    var bundle = globalScope.FlatPPLEditorBundle;
    if (!bundle) {
      throw new Error('FlatPPLEditorBundle missing — call loadBundle() first');
    }

    // Click navigation: Ctrl/Cmd-click on a [data-binding] span
    // routes through opts.onNavigate(name); plain click stays the
    // editor's cursor placement. Symmetric to today's source-pane
    // behaviour minus the "plain click navigates" path (which is
    // taken by the editor cursor in playground mode).
    var domEventHandlers = {
      mousedown: function (ev) {
        if (!(ev.ctrlKey || ev.metaKey)) return false;
        var t = ev.target;
        var name = null;
        while (t) {
          if (t.dataset && t.dataset.binding) { name = t.dataset.binding; break; }
          t = t.parentNode;
        }
        if (name && typeof opts.onNavigate === 'function') {
          ev.preventDefault();
          opts.onNavigate(name);
          return true;
        }
        return false;
      },
    };

    // Suppress onChange for programmatic setSource calls (model
    // swaps, revert, …). The caller has the new text already and
    // owns whatever side-effect (viewer.update) it wants to trigger;
    // firing onChange here would cause a redundant re-render. User-
    // typed changes still fire normally because suppressOnChange is
    // only set during dispatch and restored immediately after.
    var suppressOnChange = false;
    // Last reported binding name from cursor-driven navigation. We
    // suppress repeats so the router doesn't see a flood of
    // identical navigateTo calls when the cursor sits on a single
    // identifier across multiple updates.
    var lastCursorBinding = null;

    function bindingAtCursor() {
      var FE = globalScope.FlatPPLEngine;
      if (!FE) return null;
      var head = view.state.selection.main.head;
      var doc = view.state.doc.toString();
      var bindings = null;
      try {
        var processed = FE.processSource(doc);
        if (processed && processed.bindings) {
          bindings = new Set(processed.bindings.keys());
        }
      } catch (_) { return null; }
      if (!bindings) return null;
      var tokens = FE.tokenize(doc).tokens || [];
      var lineStarts = computeLineStarts(doc);
      for (var i = 0; i < tokens.length; i++) {
        var tok = tokens[i];
        if (tok.type !== 'IDENT') continue;
        var from = offsetOf(tok.loc.start, lineStarts);
        var to   = offsetOf(tok.loc.end,   lineStarts);
        if (head >= from && head <= to && bindings.has(tok.value)) {
          return tok.value;
        }
      }
      return null;
    }

    var docChangeListener = bundle.EditorView.updateListener.of(function (u) {
      if (suppressOnChange) return;
      if (u.docChanged && typeof opts.onChange === 'function') {
        opts.onChange(u.state.doc.toString());
      }
      // Cursor-driven navigation: when the main cursor lands on an
      // identifier that resolves to a defined binding, fire
      // onNavigate. This subsumes the read-only pane's
      // "click-a-binding-to-focus" UX (a click both moves the
      // cursor and triggers selectionSet) plus keyboard cursor
      // movement, both of which feel natural in a code editor.
      // The router de-dupes identical states so repeated calls
      // with the same target are cheap.
      if ((u.selectionSet || u.docChanged) && typeof opts.onNavigate === 'function') {
        var binding = bindingAtCursor();
        if (binding !== lastCursorBinding) {
          lastCursorBinding = binding;
          if (binding) opts.onNavigate(binding);
        }
      }
    });

    var state = bundle.EditorState.create({
      doc: typeof opts.initialSource === 'string' ? opts.initialSource : '',
      extensions: [
        bundle.lineNumbers(),
        bundle.highlightActiveLine(),
        bundle.highlightActiveLineGutter(),
        bundle.history(),
        bundle.keymap.of(
          (bundle.defaultKeymap || []).concat(
            bundle.historyKeymap || [],
            bundle.searchKeymap || []
          )
        ),
        makeHighlightPlugin(bundle),
        makeTheme(bundle),
        bundle.EditorView.domEventHandlers(domEventHandlers),
        docChangeListener,
      ],
    });

    var view = new bundle.EditorView({ state: state, parent: container });

    return {
      setSource: function (text) {
        if (text === view.state.doc.toString()) return;
        suppressOnChange = true;
        try {
          view.dispatch({
            changes: { from: 0, to: view.state.doc.length, insert: text },
          });
        } finally {
          suppressOnChange = false;
        }
      },
      getSource: function () { return view.state.doc.toString(); },
      /** Scroll the editor to the given source line (zero-indexed,
          matching the engine's tokenizer + the read-only pane's
          data-line attributes) and place the cursor at that
          line's start. The DAG → source flow (host.revealSourceLine
          on Ctrl-click of a DAG node) lands here in playground
          mode. */
      revealLine: function (line) {
        var totalLines = view.state.doc.lines;
        // CodeMirror's doc.line() is 1-indexed; the engine and our
        // read-only pane are 0-indexed. Translate + clamp.
        var n = Math.max(1, Math.min(((line | 0) + 1), totalLines));
        var info = view.state.doc.line(n);
        view.dispatch({
          selection: { anchor: info.from },
          effects: bundle.EditorView.scrollIntoView(info.from, { y: 'center' }),
        });
        // Focus the editor so the cursor is visible. Without this
        // the editor only renders the cursor when focused, and a
        // Ctrl-click on a DAG node (which kept focus on the DAG
        // pane) leaves the user with no visual indication that the
        // editor cursor moved at all. The active-line highlight
        // (theme rule) is a secondary cue but is too subtle on
        // its own to substitute for a visible cursor.
        view.focus();
      },
      destroy: function () { try { view.destroy(); } catch (_) {} },
    };
  }

  globalScope.FlatPPLWebEditor = {
    loadBundle: loadBundle,
    mountEditor: mountEditor,
  };
})(typeof window !== 'undefined' ? window : globalThis);
