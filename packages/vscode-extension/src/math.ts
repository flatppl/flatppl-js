// @flatppl/vscode-extension — math fallback for the editor hover.
//
// VS Code's `MarkdownString` hover renderer handles CommonMark + GFM
// natively but does NOT run KaTeX/MathJax on `$…$` / `$$…$$` — and
// its HTML sanitiser strips MathML even with `supportHtml = true`
// (the allowlist covers semantic HTML but not `<math>` and friends).
// So actually rendering math in the editor hover would require an
// SVG-image pipeline (MathJax-on-Node, ~300 KB bundle).
//
// Rather than pay that cost OR let inline `$…$` regions silently
// disappear from the rendered hover (which is what happens when
// stripped MathML eats them and leaves the surrounding prose
// concatenated), this module substitutes math regions with a
// CommonMark fallback that preserves the LaTeX source verbatim:
//
//   $x^2$            → `x^2`         (inline code)
//   $$\frac{a}{b}$$  → ```math
//                      \frac{a}{b}
//                      ```           (fenced code block)
//
// Result: the math notation stays visible and copy-paste-friendly,
// and the prose around it reads correctly. The DAG-view tooltip on
// both web and the VS Code webview still renders the full
// Temml-MathML — see packages/viewer/src/markdown.ts. Same Markdown
// source, two surfaces, two fidelities: rich rendering wherever the
// renderer cooperates, readable LaTeX fallback otherwise.

/** Replace `$$…$$` (display) and `$…$` (inline) math regions in the
 *  given text with CommonMark code fallbacks. Fenced code blocks
 *  (```…```) and inline code (`…`) in the input are skipped so
 *  dollar signs inside literal code don't accidentally trigger math. */
export function replaceMathWithCodeFallback(text: string): string {
  const CODE_RE = /```[\s\S]*?```|`[^`\n]*`/g;
  let out = '';
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = CODE_RE.exec(text)) !== null) {
    out += substituteMath(text.slice(last, m.index));
    out += m[0];
    last = m.index + m[0].length;
  }
  out += substituteMath(text.slice(last));
  return out;
}

function substituteMath(segment: string): string {
  // Display math first — emit a fenced code block with the `math`
  // language id so VS Code picks up its syntax-highlight grammar
  // (`Markdown All in One` and a few other extensions install one;
  // unrecognised falls back to plain monospace, still readable).
  // The fence needs surrounding blank lines for CommonMark to treat
  // it as a block, so we conservatively pad with `\n\n` on both
  // sides — the hover renderer collapses repeated blanks anyway.
  segment = segment.replace(/\$\$([\s\S]+?)\$\$/g, (_full, tex) => {
    return '\n\n```math\n' + tex.trim() + '\n```\n\n';
  });
  // Inline math — wrap in backticks. If the LaTeX itself contains
  // a backtick (uncommon) we escape it by switching to double
  // backticks, the standard CommonMark trick for inline code with
  // an embedded backtick.
  segment = segment.replace(/\$([^$\n]+?)\$/g, (_full, tex) => {
    if (tex.indexOf('`') >= 0) return '`` ' + tex + ' ``';
    return '`' + tex + '`';
  });
  return segment;
}
