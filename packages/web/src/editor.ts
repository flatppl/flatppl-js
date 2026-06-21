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
  // One-entry parse memo. Every per-keystroke consumer (binding overlay,
  // hover, cursor-binding, go-to-def) shares this so a single document
  // version is parsed + tokenized once, not once per consumer. Keyed by
  // exact text; replaced when the text changes. processed may be null
  // (parse failure) — callers already guard for that.
  let _memoText: string | null = null;
  let _memoProcessed: any = null;
  let _memoTokens: any[] | null = null;
  function parseCached(text: string): { processed: any; tokens: any[] } {
    if (text === _memoText && _memoTokens !== null) {
      return { processed: _memoProcessed, tokens: _memoTokens };
    }
    const FE = globalScope.FlatPPLEngine;
    let processed: any = null;
    let tokens: any[] = [];
    if (FE) {
      try { processed = FE.processSource(text); } catch (_) { processed = null; }
      try { tokens = (FE.tokenize(text).tokens) || []; } catch (_) { tokens = []; }
    }
    _memoText = text; _memoProcessed = processed; _memoTokens = tokens;
    return { processed, tokens };
  }

  // Fold multi-line `{ … }` regions (FlatPPL's main block delimiter:
  // module / record bodies). Heuristic depth-count over raw text — does not
  // special-case braces inside comments/strings (rare; folding is forgiving).
  function flatpplFold(bundle: any) {
    return bundle.foldService.of(function (state: any, lineStart: number, lineEnd: number) {
      const line = state.doc.sliceString(lineStart, lineEnd);
      const open = line.lastIndexOf('{');
      if (open < 0) return null;
      const docStr = state.doc.toString();
      let depth = 1;
      for (let i = lineStart + open + 1; i < docStr.length; i++) {
        const c = docStr.charCodeAt(i);
        if (c === 123 /* { */) depth++;
        else if (c === 125 /* } */) {
          depth--;
          if (depth === 0) {
            if (i <= lineEnd) return null;          // closes on the same line — not foldable
            return { from: lineStart + open + 1, to: i };
          }
        }
      }
      return null;
    });
  }

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
      the canonical TextMate grammar (window.FlatPPLTextmate, loaded eagerly via
      the <script> tag in index.html; its grammar parse is async and repaints
      when ready). This plugin adds ONLY the engine-derived semantic overlay:
      identifiers that resolve to a defined binding get `tok-ident-binding` + a
      `data-binding` attribute (the gallery's Ctrl-click navigation).

      Base + overlay are two separate mark decorations over the same range, so
      CodeMirror renders them as NESTED spans (not one element with two classes).
      The inner span wraps the text node directly, so its colour wins by the CSS
      cascade — and in CM6 the HIGHER-precedence decoration is the inner span.
      So for `.tok-ident-binding` to win over the TextMate base mark, this plugin
      must outrank FlatPPLTextmate's; the caller wraps it in `bundle.Prec.high(...)`
      to guarantee that regardless of extension-list order. (Verified in-browser:
      without the Prec bump the base scope, e.g. `tok-reserved`, nests inside and
      the binding colour is lost. Not "CSS source order" — decoration precedence
      drives the span nesting.) */
  function makeHighlightPlugin(bundle: any) {
    const ViewPlugin = bundle.ViewPlugin;
    const Decoration = bundle.Decoration;
    const FE = globalScope.FlatPPLEngine;

    function bindingOverlay(view: any) {
      const B = FE && FE.builtins;
      if (!FE || !B) return Decoration.none;
      const text = view.state.doc.toString();
      const { processed, tokens } = parseCached(text);
      let bindings: Set<unknown> | null = null;
      if (processed && processed.bindings) bindings = new Set(processed.bindings.keys());
      if (!bindings) return Decoration.none;
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
          if (u.docChanged) {
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
      const { processed, tokens } = parseCached(text);
      if (!processed || !processed.bindings) return null;
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

  // Surface engine diagnostics (tokenizer/parser/analyzer) as CM lint markers.
  // Reuses parseCached — adds no new parse. CM debounces linter runs itself.
  function makeLinter(bundle: any) {
    return bundle.linter(function (view: any) {
      const text = view.state.doc.toString();
      const { processed } = parseCached(text);
      const diags = (processed && processed.diagnostics) || [];
      if (!diags.length) return [];
      const lineStarts = computeLineStarts(text);
      const out: any[] = [];
      for (let i = 0; i < diags.length; i++) {
        const d = diags[i];
        if (!d || !d.loc || !d.loc.start) continue;
        let from = offsetOf(d.loc.start, lineStarts);
        let to = d.loc.end ? offsetOf(d.loc.end, lineStarts) : from;
        from = Math.max(0, Math.min(from, text.length));
        to = Math.max(0, Math.min(to, text.length));
        if (to <= from) to = Math.min(from + 1, text.length);
        const sev = (d.severity === 'warning' || d.severity === 'info') ? d.severity : 'error';
        out.push({ from: from, to: to, severity: sev, message: String(d.message || 'error') });
      }
      return out;
    });
  }

  // Identifier completions: in-scope bindings (from parseCached) first, then
  // FlatPPL builtins from FlatPPLEngine.builtins. Lists may be arrays or Sets.
  function makeCompletion(bundle: any) {
    function names(src: any): string[] {
      if (!src) return [];
      if (Array.isArray(src)) return src;
      if (typeof src.forEach === 'function' && typeof src.size === 'number') return Array.from(src as Set<string>);
      return Object.keys(src);
    }
    return bundle.autocompletion({
      override: [function (context: any) {
        const word = context.matchBefore(/[A-Za-z_][A-Za-z0-9_]*/);
        if (!word || (word.from === word.to && !context.explicit)) return null;
        const seen: Record<string, boolean> = {};
        const options: any[] = [];
        function add(list: any, type: string) {
          const arr = names(list);
          for (let i = 0; i < arr.length; i++) {
            const n = arr[i];
            if (n && !seen[n]) { seen[n] = true; options.push({ label: n, type: type }); }
          }
        }
        // Bindings first (most relevant), then builtins by category.
        const { processed } = parseCached(context.state.doc.toString());
        if (processed && processed.bindings) add(Array.from(processed.bindings.keys()), 'variable');
        const FE = globalScope.FlatPPLEngine;
        const B = FE && FE.builtins;
        if (B) {
          add(B.DISTRIBUTIONS, 'class');
          add(B.BUILTIN_FUNCTIONS, 'function');
          add(B.MEASURE_OPS, 'keyword');
          add(B.MEASURE_PRODUCING, 'keyword');
          add(B.SPECIAL_OPERATIONS, 'keyword');
          add(B.CONSTANTS, 'constant');
          add(B.BOOL_LITERALS, 'constant');
          add(B.SETS, 'type');
          add(B.SET_CONSTRUCTORS, 'type');
        }
        if (options.length === 0) return null;
        return { from: word.from, options: options };
      }],
    });
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
      const { processed } = parseCached(doc);
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
      const { processed, tokens } = parseCached(doc);
      if (!processed || !processed.bindings) return null;
      const bindings = new Set(processed.bindings.keys());
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

    // TextMate base highlighter, when its bundle loaded. init() kicks off the
    // async grammar/WASM load (fire-and-forget; the plugin repaints on ready).
    // Must precede the binding overlay below — see makeHighlightPlugin's doc on
    // span-nesting precedence.
    let textmateExt: any[] = [];
    if (globalScope.FlatPPLTextmate) {
      globalScope.FlatPPLTextmate.init();
      textmateExt = [globalScope.FlatPPLTextmate.makeHighlightPlugin(bundle)];
    }

    const extensions: any[] = [
      bundle.lineNumbers(),
      bundle.highlightActiveLine(),
      bundle.highlightActiveLineGutter(),
      bundle.bracketMatching(),
      bundle.codeFolding(),
      bundle.foldGutter(),
      flatpplFold(bundle),
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
        }, {
          key: 'Mod-/',
          preventDefault: true,
          run: function () { return toggleLineComment(); },
        }] as any[]).concat(
          bundle.defaultKeymap || [],
          bundle.historyKeymap || [],
          bundle.searchKeymap || [],
          bundle.foldKeymap || [],
          bundle.completionKeymap || []
        )
      ),
      ...textmateExt,
      // Prec.high so the binding overlay nests INSIDE the TextMate base mark
      // (higher precedence => inner span => its colour wins). See
      // makeHighlightPlugin's doc comment.
      bundle.Prec.high(makeHighlightPlugin(bundle)),
      makeHoverTooltip(bundle),
      makeLinter(bundle),
      makeCompletion(bundle),
      bundle.lintGutter(),
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

    // Toggle FlatPPL line comments (`#`) on the active line / selection.
    // Standard editor behaviour: if every non-blank affected line is already
    // commented, uncomment; otherwise comment. One transaction → one undo
    // step. No-op in view mode. FlatPPL's line comment is `#`; `%` doc-comments
    // and `###` block fences are treated as ordinary text (no special-casing).
    function toggleLineComment(): boolean {
      if (view.state.readOnly) return false;
      const state = view.state;
      const ranges = state.selection.ranges;
      const lineNums: number[] = [];
      const seen: Record<number, boolean> = {};
      for (let r = 0; r < ranges.length; r++) {
        const fromLine = state.doc.lineAt(ranges[r].from).number;
        const toLine   = state.doc.lineAt(ranges[r].to).number;
        for (let ln = fromLine; ln <= toLine; ln++) {
          if (!seen[ln]) { seen[ln] = true; lineNums.push(ln); }
        }
      }
      if (lineNums.length === 0) return true;

      // Remove only if EVERY non-blank affected line is already commented.
      let anyNonBlank = false;
      let allCommented = true;
      for (let i = 0; i < lineNums.length; i++) {
        const text = state.doc.line(lineNums[i]).text;
        const firstNW = text.search(/\S/);
        if (firstNW < 0) continue;
        anyNonBlank = true;
        if (text.charAt(firstNW) !== '#') { allCommented = false; break; }
      }
      if (!anyNonBlank) return true;

      const changes: Array<{ from: number; to: number; insert?: string }> = [];
      for (let i = 0; i < lineNums.length; i++) {
        const line = state.doc.line(lineNums[i]);
        const text = line.text;
        const firstNW = text.search(/\S/);
        if (allCommented) {
          if (firstNW < 0 || text.charAt(firstNW) !== '#') continue;
          const hashPos = line.from + firstNW;
          const hasSpace = text.charAt(firstNW + 1) === ' ';
          changes.push({ from: hashPos, to: hashPos + (hasSpace ? 2 : 1), insert: '' });
        } else {
          if (firstNW < 0) continue;             // skip blank lines on add
          changes.push({ from: line.from + firstNW, to: line.from + firstNW, insert: '# ' });
        }
      }
      if (changes.length === 0) return true;
      view.dispatch({ changes });
      return true;
    }

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
