// Entry point for the CodeMirror 6 bundle that powers the
// gallery's source viewer + editor (unified surface). Imported
// through esbuild and emitted as dist/vendor/codemirror.min.js —
// a single IIFE that publishes the curated subset of CodeMirror
// APIs the gallery needs onto window.FlatPPLEditorBundle.
//
// CodeMirror is now used in BOTH view and edit modes (toggled via
// a Compartment-reconfigured `readOnly` + `editable` pair); the
// previous custom-HTML view (`<pre>` + syntax.ts) was replaced so
// hover / find / decorations / virtual-scrolling work uniformly
// at any file size. See packages/web/src/editor.ts for the wiring.

import { EditorState, Compartment, Prec } from '@codemirror/state';
import {
  EditorView, keymap, lineNumbers, highlightActiveLine,
  highlightActiveLineGutter, ViewPlugin, Decoration, hoverTooltip,
} from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { searchKeymap } from '@codemirror/search';
import { linter, lintGutter } from '@codemirror/lint';
import { autocompletion, completionKeymap } from '@codemirror/autocomplete';
import {
  bracketMatching, foldGutter, codeFolding, foldKeymap, foldService,
  syntaxHighlighting, defaultHighlightStyle,
} from '@codemirror/language';
import { json as langJson } from '@codemirror/lang-json';
import { markdown as langMarkdown } from '@codemirror/lang-markdown';

window.FlatPPLEditorBundle = {
  EditorState, Compartment, Prec,
  EditorView, keymap, lineNumbers, highlightActiveLine,
  highlightActiveLineGutter, ViewPlugin, Decoration, hoverTooltip,
  defaultKeymap, history, historyKeymap,
  searchKeymap,
  linter, lintGutter,
  autocompletion, completionKeymap,
  bracketMatching, foldGutter, codeFolding, foldKeymap, foldService,
  // Non-FlatPPL languages: real syntax highlighting + native (foldNodeProp)
  // folding for .json / .md, so they no longer ride the FlatPPL TextMate
  // grammar. syntaxHighlighting(defaultHighlightStyle) is what actually
  // paints the lang parsers' highlight tags.
  syntaxHighlighting, defaultHighlightStyle,
  langJson, langMarkdown,
};
