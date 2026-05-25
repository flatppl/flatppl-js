// @flatppl/viewer — doc-comment rendering for hover tooltips.
//
// FlatPPL doc-comments (% / %%%) carry a `markup` tag (`md` for
// CommonMark+math, `typ` for Typst); the parser surfaces them on
// each binding as `{ markup, lines }`. This module turns that into
// HTML for the DAG-view tooltip (and is reused by the VS Code
// extension's hover provider on the Node side — see
// packages/vscode-extension/extension.ts).
//
// Renderer choices:
//   - Markdown skeleton: `marked` (CommonMark + GFM, ~37 KB minified).
//   - Math:              `temml` (LaTeX → MathML, ~57 KB minified, no
//                        fonts required — modern browsers ship MathML
//                        Core).
//   - Typst:             no client-side renderer ships; we surface
//                        the literal source inside <pre>. The VS Code
//                        hover handles `typ` separately via a
//                        ```typst fenced code block (Tinymist
//                        highlights when installed).
//
// The math grammar is the Pandoc convention: `$inline$` and
// `$$display$$`. Math tokens are added via a marked extension so
// they survive the CommonMark pipeline without being chewed up as
// emphasis (`*x*`) or being HTML-escaped before Temml sees them.

import { marked, type MarkedExtension } from 'marked';
import * as temmlNs from 'temml';

// Temml's CJS export shape is `{ default: { renderToString, ... } }`
// under esbuild's bundle; the ESM default is the renderer namespace
// itself. Normalize so we always call `.renderToString`.
const temml: { renderToString: (tex: string, opts?: any) => string } =
  (temmlNs as any).default ?? (temmlNs as any);

function renderTexSafe(tex: string, displayMode: boolean): string {
  try {
    return temml.renderToString(tex, { displayMode, throwOnError: false });
  } catch (err) {
    // Temml's `throwOnError: false` already swallows parse errors and
    // returns an inline error node; this catch covers anything else
    // (loader failure, malformed Unicode) so a single bad doc-comment
    // doesn't blank the whole tooltip.
    return '<span class="tml-error">' + escapeHtml(tex) + '</span>';
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => {
    switch (c) {
      case '&': return '&amp;';
      case '<': return '&lt;';
      case '>': return '&gt;';
      case '"': return '&quot;';
      default:  return '&#39;';
    }
  });
}

// `marked` extension wiring math at the tokenizer level. Display
// math (`$$…$$`) is a block token so it can span lines; inline math
// is an inline token. Both render via Temml at the renderer step.
function mathExtension(): MarkedExtension {
  return {
    extensions: [
      {
        name: 'mathBlock',
        level: 'block',
        start(src: string) {
          const i = src.indexOf('$$');
          return i < 0 ? undefined : i;
        },
        tokenizer(src: string) {
          const m = /^\$\$([\s\S]+?)\$\$/.exec(src);
          if (!m) return undefined;
          return {
            type: 'mathBlock',
            raw: m[0],
            text: m[1].trim(),
          } as any;
        },
        renderer(token: any) {
          return renderTexSafe(token.text, /* displayMode */ true);
        },
      },
      {
        name: 'mathInline',
        level: 'inline',
        start(src: string) {
          const i = src.indexOf('$');
          return i < 0 ? undefined : i;
        },
        tokenizer(src: string) {
          // Reject `$$` (handled by the block tokenizer) and require
          // a non-empty body that doesn't span newlines — keeps the
          // delimiter tight so a single stray `$` in prose doesn't
          // eat the rest of a paragraph.
          if (src.startsWith('$$')) return undefined;
          const m = /^\$([^$\n]+?)\$/.exec(src);
          if (!m) return undefined;
          return {
            type: 'mathInline',
            raw: m[0],
            text: m[1],
          } as any;
        },
        renderer(token: any) {
          return renderTexSafe(token.text, /* displayMode */ false);
        },
      },
    ],
  };
}

let _mathWired = false;
function ensureMarkedConfigured() {
  if (_mathWired) return;
  marked.use(mathExtension());
  // Multi-line doc-comments rarely intend hard line breaks; treat the
  // collected `lines` array as standard CommonMark paragraphs.
  marked.setOptions({ gfm: true, breaks: false });
  _mathWired = true;
}

export interface DocComment {
  markup: string;
  lines: string[];
}

/** Render a doc-comment to an HTML fragment. Returns null when there
 *  is no content to show. */
export function renderDoc(doc: DocComment | null | undefined): string | null {
  if (!doc || !doc.lines || doc.lines.length === 0) return null;
  ensureMarkedConfigured();
  const text = doc.lines.join('\n');
  if (doc.markup === 'md') {
    return marked.parse(text, { async: false }) as string;
  }
  if (doc.markup === 'typ') {
    // No client-side Typst renderer. Show the source verbatim — same
    // fallback the VS Code hover uses when Tinymist isn't installed.
    return '<pre class="tooltip-typst-src">' + escapeHtml(text) + '</pre>';
  }
  // Unknown markup tag — should already have errored at parse time,
  // but if a custom tag slips through we still want something visible.
  return '<pre>' + escapeHtml(text) + '</pre>';
}

/** Render just a single math expression (used by the VS Code hover
 *  provider when embedding MathML into a MarkdownString that already
 *  handles the surrounding Markdown skeleton). */
export function renderMath(tex: string, displayMode: boolean): string {
  return renderTexSafe(tex, displayMode);
}
