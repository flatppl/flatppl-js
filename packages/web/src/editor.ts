// @flatppl/web — source editor (view + edit, unified on CodeMirror).
//
// FlatPPL-specific glue around CodeMirror 6 (loaded eagerly from
// vendor/codemirror.min.js — see index.html). Replaces the previous
// dual surface (read-only `<pre>` + syntax.ts on one side, lazy
// CodeMirror on the other) with a single CodeMirror instance whose
// `readOnly` / `editable` flags toggle between view and edit modes
// via Compartment reconfiguration.
//
// Exposes window.FlatPPLWebEditor.mountEditor(container, opts) →
//   { setSource, getSource, setReadOnly, revealLine, replaceRange,
//     destroy }
//
// opts:
//   initialSource (string, '')
//   initialReadOnly (boolean, true) — start in view mode by default;
//                                     the gallery toggles to edit mode
//                                     when the user flips the source-
//                                     pane edit button.
//   onChange(text) — fired on every doc change; gallery debounces
//                    and re-renders the viewer.
//   onNavigate(name) — fired when the main cursor lands on an
//                      identifier that resolves to a defined binding,
//                      or when the user Ctrl/Cmd-clicks one.
//
// Visual differentiation between modes is the editor's natural
// affordance: view mode has no caret, no active-line highlight,
// no editable input area; edit mode shows the caret and the
// active-line stripe. Same text rendering, same syntax highlight,
// same hover — only the input behaviour changes.

'use strict';

(function (globalScope: any) {
  function computeLineStarts(src: any) {
    const starts = [0];
    for (let i = 0; i < src.length; i++) {
      if (src.charCodeAt(i) === 10 /* \n */) starts.push(i + 1);
    }
    return starts;
  }

  function offsetOf(loc: any, lineStarts: any) {
    const ls = lineStarts[loc.line];
    return (typeof ls === 'number' ? ls : 0) + loc.col;
  }

  /** Build the FlatPPL highlight ViewPlugin. Base lexical highlight comes from
      the canonical TextMate grammar (window.FlatPPLTextmate, async-loaded and
      added separately in the extensions list). This plugin adds ONLY the
      engine-derived semantic overlay: identifiers that resolve to a defined
      binding get `tok-ident-binding` + a `data-binding` attribute (the gallery's
      Ctrl-click navigation). Base + overlay are separate marks on the same
      range; `.tok-ident-binding` wins by CSS source order. */
  function makeHighlightPlugin(bundle: any) {
    const ViewPlugin = bundle.ViewPlugin;
    const Decoration = bundle.Decoration;
    const FE = globalScope.FlatPPLEngine;

    function bindingOverlay(view: any) {
      const B = FE && FE.builtins;
      if (!FE || !B) return Decoration.none;
      const text = view.state.doc.toString();
      let bindings: Set<unknown> | null = null;
      try {
        const processed = FE.processSource(text);
        if (processed && processed.bindings) bindings = new Set(processed.bindings.keys());
      } catch (_) { bindings = null; }
      if (!bindings) return Decoration.none;

      const tokens = FE.tokenize(text).tokens || [];
      const lineStarts = computeLineStarts(text);
      const ranges: any[] = [];
      for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        if (tok.type !== 'IDENT' || !bindings.has(tok.value)) continue;
        const from = offsetOf(tok.loc.start, lineStarts);
        const to   = offsetOf(tok.loc.end,   lineStarts);
        if (to <= from) continue;
        ranges.push(
          Decoration.mark({
            class: 'tok-ident-binding',
            attributes: { 'data-binding': tok.value },
          }).range(from, to)
        );
      }
      return Decoration.set(ranges, true);
    }

    return ViewPlugin.fromClass(
      function (this: any, view: any) {
        this.decorations = bindingOverlay(view);
        this.update = function (this: any, u: any) {
          if (u.docChanged || u.viewportChanged) {
            this.decorations = bindingOverlay(u.view);
          }
        };
      },
      { decorations: function (v: any) { return v.decorations; } }
    );
  }

  /** Hover-tooltip extension: on hover over an identifier that
      resolves to a defined binding, render the same content as the
      DAG-view tooltip (`name = expr` + attached doc-comment as
      Markdown + MathML via the viewer's renderDoc helper). For
      identifiers that don't resolve to a binding, no tooltip is
      shown — matching the DAG-view behaviour. */
  function makeHoverTooltip(bundle: any) {
    const hoverTooltip = bundle.hoverTooltip;
    const FE = globalScope.FlatPPLEngine;
    if (!FE || typeof hoverTooltip !== 'function') return [];

    // Resolve the FlatPPLViewer's renderDoc lazily — the viewer
    // bundle loads after the editor bundle, so at module-eval time
    // it may not yet be present. Lookup on each hover (cheap) lets
    // us pick it up as soon as it's available.
    function renderDoc(doc: any): string | null {
      const v = (globalScope.FlatPPLViewer || {});
      return typeof v.renderDoc === 'function' ? v.renderDoc(doc) : null;
    }

    return hoverTooltip(function (view: any, pos: number, side: number) {
      const text = view.state.doc.toString();
      let processed;
      try { processed = FE.processSource(text); }
      catch (_) { return null; }
      if (!processed || !processed.bindings) return null;
      const tokens = FE.tokenize(text).tokens || [];
      const lineStarts = computeLineStarts(text);
      // Locate the IDENT token under `pos`. CodeMirror's hover API
      // gives us a `side`: -1 means the caret is on the left edge
      // of `pos`, +1 the right edge. We accept any IDENT whose
      // range [from, to] contains pos in the half-open sense
      // [from, to), with the side handling the boundary case.
      for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        if (tok.type !== 'IDENT') continue;
        const from = offsetOf(tok.loc.start, lineStarts);
        const to   = offsetOf(tok.loc.end,   lineStarts);
        if (pos < from || pos > to) continue;
        if ((pos === from && side < 0) || (pos === to && side > 0)) continue;
        const name = tok.value;
        const binding = processed.bindings.get(name);
        if (!binding) return null;
        // Build the tooltip body. The `name = expr` line uses
        // textContent (set via the dom callback's element below)
        // so identifier-derived strings can't smuggle HTML; the
        // doc-comment HTML comes from the viewer's renderDoc which
        // owns its own escaping pipeline (marked + temml).
        return {
          pos: from,
          end: to,
          above: true,
          create: function () {
            const dom = document.createElement('div');
            dom.className = 'cm-source-hover';
            const exprLine = document.createElement('div');
            exprLine.className = 'cm-source-hover-expr';
            exprLine.textContent = name + ' = ' + (binding.rhs || '');
            dom.appendChild(exprLine);
            if (binding.node && binding.node.doc
                && binding.node.doc.lines
                && binding.node.doc.lines.length > 0) {
              const docBlock = document.createElement('div');
              docBlock.className = 'cm-source-hover-doc';
              const html = renderDoc(binding.node.doc);
              if (html) docBlock.innerHTML = html;
              else      docBlock.textContent = binding.node.doc.lines.join('\n');
              dom.appendChild(docBlock);
            }
            return { dom: dom };
          },
        };
      }
      return null;
    }, { hideOnChange: true });
  }

  /** Build a small EditorView.theme matching the gallery's dark
      palette so the editor blends with the surrounding panes. */
  function makeTheme(bundle: any) {
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

  function mountEditor(container: any, opts: any) {
    opts = opts || {};
    const bundle = globalScope.FlatPPLEditorBundle;
    if (!bundle) {
      throw new Error('FlatPPLEditorBundle missing — load vendor/codemirror.min.js first');
    }

    // readOnly + editable live in a Compartment so the view ↔ edit
    // toggle can reconfigure them on the fly without rebuilding the
    // editor. We toggle both together: `readOnly: true` blocks
    // programmatic edits via dispatch (transactions can still apply
    // selection-only changes), and `editable: false` blocks user
    // input + hides the caret. View mode wants both off; edit mode
    // wants both unset (the defaults are writable + editable).
    const initialReadOnly = opts.initialReadOnly !== false;
    const readOnlyCompartment = new bundle.Compartment();
    function readOnlyExtensions(ro: boolean) {
      return [
        bundle.EditorState.readOnly.of(ro),
        bundle.EditorView.editable.of(!ro),
      ];
    }

    // Click-navigation (Ctrl/Cmd-click on a defined binding → jump
    // to its definition). Works regardless of mode: in view mode
    // the editor isn't focusable for typing, but it still receives
    // mousedown events.
    function jumpToBindingDefinition(name: any) {
      const FE = globalScope.FlatPPLEngine;
      if (!FE) return;
      const doc = view.state.doc.toString();
      let processed;
      try { processed = FE.processSource(doc); } catch (_) { return; }
      if (!processed || !processed.bindings || !processed.bindings.has(name)) return;
      const b = processed.bindings.get(name);
      const nameLoc = b && b.nameLoc && b.nameLoc.start;
      if (!nameLoc) return;
      const lineStarts = computeLineStarts(doc);
      const pos = (lineStarts[nameLoc.line] || 0) + (nameLoc.col || 0);
      view.dispatch({
        selection: { anchor: pos },
        effects: bundle.EditorView.scrollIntoView(pos, { y: 'center' }),
      });
      view.focus();
    }

    const domEventHandlers = {
      mousedown: function (ev: any) {
        // Ctrl/Cmd-click on a defined-binding identifier: jump to its
        // definition. Works in both view and edit modes.
        if (ev.ctrlKey || ev.metaKey) {
          let t = ev.target;
          let name: any = null;
          while (t) {
            if (t.dataset && t.dataset.binding) { name = t.dataset.binding; break; }
            t = t.parentNode;
          }
          if (!name) return false;
          ev.preventDefault();
          jumpToBindingDefinition(name);
          if (typeof opts.onNavigate === 'function') {
            opts.onNavigate(name);
          }
          return true;
        }
        // Plain click in view mode: CodeMirror's editable=false stops
        // the caret from following the click — but the gallery's
        // click-to-focus-binding UX depends on cursor-driven
        // onNavigate firing when the new caret lands on a defined
        // binding. Dispatch the selection move manually here so the
        // selectionSet event fires; the docChangeListener picks it
        // up and routes onNavigate just like it would in edit mode.
        // We don't preventDefault so native text-selection (click +
        // drag) still works. In edit mode CodeMirror handles caret
        // movement natively, so the branch is a no-op there.
        if (view.state.readOnly) {
          const pos = view.posAtCoords({ x: ev.clientX, y: ev.clientY });
          if (pos != null && pos !== view.state.selection.main.head) {
            view.dispatch({ selection: { anchor: pos } });
          }
        }
        return false;
      },
    };

    let suppressOnChange = false;
    let suppressNavigate = false;
    let lastCursorBinding: any = null;

    function bindingAtCursor() {
      const FE = globalScope.FlatPPLEngine;
      if (!FE) return null;
      const head = view.state.selection.main.head;
      const doc = view.state.doc.toString();
      let bindings: Set<unknown> | null = null;
      try {
        const processed = FE.processSource(doc);
        if (processed && processed.bindings) {
          bindings = new Set(processed.bindings.keys());
        }
      } catch (_) { return null; }
      if (!bindings) return null;
      const tokens = FE.tokenize(doc).tokens || [];
      const lineStarts = computeLineStarts(doc);
      for (let i = 0; i < tokens.length; i++) {
        const tok = tokens[i];
        if (tok.type !== 'IDENT') continue;
        const from = offsetOf(tok.loc.start, lineStarts);
        const to   = offsetOf(tok.loc.end,   lineStarts);
        if (head >= from && head <= to && bindings.has(tok.value)) {
          return tok.value;
        }
      }
      return null;
    }

    const docChangeListener = bundle.EditorView.updateListener.of(function (u: any) {
      if (suppressOnChange) return;
      if (u.docChanged && typeof opts.onChange === 'function') {
        opts.onChange(u.state.doc.toString());
      }
      if ((u.selectionSet || u.docChanged)
          && !suppressNavigate
          && typeof opts.onNavigate === 'function') {
        const binding = bindingAtCursor();
        if (binding !== lastCursorBinding) {
          lastCursorBinding = binding;
          if (binding) opts.onNavigate(binding);
        }
      }
    });

    // The line-flash decoration set: a single mark on the target
    // line that the gallery's revealLine method installs, then
    // removes after ~1.5 s. Implementing the flash as a Decoration
    // (rather than mutating DOM directly) keeps it correct under
    // CodeMirror's virtual scrolling — the line can scroll out and
    // back without losing the highlight.
    const flashState = bundle.StateField
      ? null
      : null;
    // Use a ViewPlugin to manage flash decorations instead of
    // StateField (StateField requires importing more from
    // @codemirror/state into the bundle). The plugin holds a
    // mutable DecorationSet and a small API for installing /
    // clearing flashes.
    let flashView: any = null;
    const flashPlugin = bundle.ViewPlugin.fromClass(
      function (this: any, view: any) {
        flashView = this;
        this.view = view;
        this.decorations = bundle.Decoration.none;
        this.timer = null;
        this.flashLine = function (line1: number) {
          if (line1 < 1 || line1 > view.state.doc.lines) return;
          const info = view.state.doc.line(line1);
          const deco = bundle.Decoration.line({ class: 'cm-line-flash' })
            .range(info.from);
          this.decorations = bundle.Decoration.set([deco]);
          view.requestMeasure();
          if (this.timer) clearTimeout(this.timer);
          const self = this;
          this.timer = setTimeout(function () {
            self.decorations = bundle.Decoration.none;
            view.requestMeasure();
          }, 1500);
        };
        this.update = function () { /* nothing — decorations persist
          across viewport changes until the timer clears them. */ };
        this.destroy = function () {
          if (this.timer) clearTimeout(this.timer);
          flashView = null;
        };
      },
      { decorations: function (v: any) { return v.decorations; } }
    );

    const extensions: any[] = [
      bundle.lineNumbers(),
      bundle.highlightActiveLine(),
      bundle.highlightActiveLineGutter(),
      bundle.history(),
      bundle.keymap.of(
        // The Ctrl/Cmd-S binding gets `preventDefault: true` so the
        // browser's "save page" dialog doesn't fire on top of our
        // save action. The handler is callable in view mode too —
        // the gallery's onSave decides whether there's anything to
        // do (no-op when no editable buffer / no dirty changes).
        ([{
          key: 'Mod-s',
          preventDefault: true,
          run: function () {
            if (typeof opts.onSave === 'function') opts.onSave();
            return true;
          },
        }] as any[]).concat(
          bundle.defaultKeymap || [],
          bundle.historyKeymap || [],
          bundle.searchKeymap || []
        )
      ),
      ...(globalScope.FlatPPLTextmate
        ? [(globalScope.FlatPPLTextmate.init(), globalScope.FlatPPLTextmate.makeHighlightPlugin(bundle))]
        : []),
      makeHighlightPlugin(bundle),
      makeHoverTooltip(bundle),
      flashPlugin,
      makeTheme(bundle),
      bundle.EditorView.domEventHandlers(domEventHandlers),
      docChangeListener,
      readOnlyCompartment.of(readOnlyExtensions(initialReadOnly)),
    ];

    const state = bundle.EditorState.create({
      doc: typeof opts.initialSource === 'string' ? opts.initialSource : '',
      extensions: extensions,
    });

    var view = new bundle.EditorView({ state: state, parent: container });

    return {
      setSource: function (text: any) {
        if (text === view.state.doc.toString()) return;
        suppressOnChange = true;
        try {
          // readOnly: true blocks `changes` transactions by default.
          // Programmatic source loads bypass that via the
          // `userEvent: 'input'`-free dispatch — readOnly only
          // rejects *user* edits in CodeMirror's contract, but to
          // be defensive we temporarily reconfigure to writable.
          const wasRO = view.state.readOnly;
          if (wasRO) {
            view.dispatch({
              effects: readOnlyCompartment.reconfigure(readOnlyExtensions(false)),
            });
          }
          view.dispatch({
            changes: { from: 0, to: view.state.doc.length, insert: text },
          });
          if (wasRO) {
            view.dispatch({
              effects: readOnlyCompartment.reconfigure(readOnlyExtensions(true)),
            });
          }
        } finally {
          suppressOnChange = false;
        }
      },
      getSource: function () { return view.state.doc.toString(); },
      /** Toggle the editor between view (readOnly + non-editable)
       *  and edit (writable + editable). The doc, selection,
       *  history and decorations all persist across the toggle —
       *  same instance, different input behaviour. */
      setReadOnly: function (ro: boolean) {
        view.dispatch({
          effects: readOnlyCompartment.reconfigure(readOnlyExtensions(!!ro)),
        });
      },
      /** Scroll the editor to the given source line (zero-indexed,
          matching the engine's tokenizer's positions). Adds a
          short-lived line-flash decoration so the destination is
          visually obvious. The DAG → source flow lands here. */
      revealLine: function (line: any) {
        const totalLines = view.state.doc.lines;
        const n = Math.max(1, Math.min(((line | 0) + 1), totalLines));
        const info = view.state.doc.line(n);
        view.dispatch({
          selection: { anchor: info.from },
          effects: bundle.EditorView.scrollIntoView(info.from, { y: 'center' }),
        });
        if (flashView) flashView.flashLine(n);
        // Don't auto-focus in view mode — the caret isn't visible
        // there anyway, and focusing the editor steals focus from
        // the DAG pane (Ctrl-click target).
        if (view.state.readOnly === false) view.focus();
      },
      replaceRange: function (from: any, to: any, text: any) {
        suppressNavigate = true;
        try {
          const wasRO = view.state.readOnly;
          if (wasRO) {
            view.dispatch({
              effects: readOnlyCompartment.reconfigure(readOnlyExtensions(false)),
            });
          }
          view.dispatch({
            changes: { from: from, to: to, insert: text },
          });
          if (wasRO) {
            view.dispatch({
              effects: readOnlyCompartment.reconfigure(readOnlyExtensions(true)),
            });
          }
        } finally {
          suppressNavigate = false;
        }
      },
      destroy: function () { try { view.destroy(); } catch (_) {} },
    };
  }

  globalScope.FlatPPLWebEditor = {
    mountEditor: mountEditor,
  };
})(typeof window !== 'undefined' ? window : globalThis);
