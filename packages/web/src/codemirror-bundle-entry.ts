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

window.FlatPPLEditorBundle = {
  EditorState, Compartment, Prec,
  EditorView, keymap, lineNumbers, highlightActiveLine,
  highlightActiveLineGutter, ViewPlugin, Decoration, hoverTooltip,
  defaultKeymap, history, historyKeymap,
  searchKeymap,
};
