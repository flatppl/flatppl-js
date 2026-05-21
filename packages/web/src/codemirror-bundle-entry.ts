// Entry point for the CodeMirror 6 bundle that powers the
// playground's source editor. Imported through esbuild and
// emitted as dist/vendor/codemirror.min.js — a single IIFE that
// publishes the curated subset of CodeMirror APIs the gallery
// needs onto window.FlatPPLEditorBundle.
//
// The bundle is loaded **lazily** by the gallery (only when
// __FLATPPL_CONFIG__.playground is true), so non-playground
// deploys never fetch it. See packages/web/src/editor.js for the
// FlatPPL-specific glue (highlight plugin, click handlers,
// debounced re-render).

import { EditorState, Compartment } from '@codemirror/state';
import {
  EditorView, keymap, lineNumbers, highlightActiveLine,
  highlightActiveLineGutter, ViewPlugin, Decoration,
} from '@codemirror/view';
import { defaultKeymap, history, historyKeymap } from '@codemirror/commands';
import { searchKeymap } from '@codemirror/search';

window.FlatPPLEditorBundle = {
  EditorState, Compartment,
  EditorView, keymap, lineNumbers, highlightActiveLine,
  highlightActiveLineGutter, ViewPlugin, Decoration,
  defaultKeymap, history, historyKeymap,
  searchKeymap,
};
