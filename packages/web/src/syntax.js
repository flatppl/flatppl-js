// @flatppl/web — syntax highlighter for the source pane.
//
// FlatPPL-aware highlighting backed by the engine's own tokenizer
// (no third-party Prism / Shiki / etc grammar to maintain). The
// tokenizer hands us classified tokens with line/col positions; we
// walk them and emit a <span> per token plus the literal source text
// for inter-token whitespace, so the source pane is byte-for-byte
// identical to the original aside from the wrapping spans.
//
// Each emitted token span carries:
//   class="tok-<kind>"          coarse classification (used by CSS)
//   data-line / data-col        token start position (zero-indexed)
//   data-binding="<name>"       on identifier tokens whose name
//                               appears as a binding in the analyzed
//                               source — these are the click targets
//                               for the source ↔ DAG navigation in
//                               step 1.6.
//
// Lives on globalThis as window.FlatPPLWebSyntax.highlight.

'use strict';

(function (globalScope) {
  // Token-class mapping. Falls through to 'tok-ident' for unknown
  // identifiers (user names that don't appear in any builtins
  // catalogue and aren't bindings — typically nothing, since every
  // user-introduced name should be a binding, but the parser
  // tolerates references to undefined names with a warning).
  // Names that act as keywords / operators at the surface level
  // across variants. `and`/`or`/`not` are FlatPPY logical keywords;
  // `in` is the membership comparison operator in all three variants;
  // `true`/`false`/`True`/`False` are boolean keywords (variant-
  // specific spelling, but treated uniformly by the highlighter).
  var KEYWORD_NAMES = new Set([
    'and', 'or', 'not', 'in',
    'true', 'false', 'True', 'False',
  ]);

  function classifyIdentifier(name, bindings, B) {
    if (KEYWORD_NAMES.has(name))        return 'tok-keyword';
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

  // Token types that map to the generic operator class. Includes the
  // new variant operators: TILDE (`~`), CARET (`^`), AMPAMP (`&&`),
  // PIPEPIPE (`||`), BANG (`!`). SEMI (`;`) is treated as punctuation.
  var OP_TOKEN_TYPES = new Set([
    'EQUALS', 'EQEQ', 'NEQ',
    'LT', 'GT', 'LTE', 'GTE',
    'PLUS', 'MINUS', 'STAR', 'SLASH',
    'TILDE', 'CARET', 'AMPAMP', 'PIPEPIPE', 'BANG',
  ]);

  function classifyToken(tok) {
    var t = tok.type;
    if (t === 'COMMENT')     return 'tok-comment';
    if (t === 'STRING')      return 'tok-string';
    if (t === 'NUMBER')      return 'tok-number';
    if (t === 'PLACEHOLDER') return 'tok-placeholder';
    if (t === 'HOLE')        return 'tok-hole';
    if (OP_TOKEN_TYPES.has(t)) return 'tok-op';
    return 'tok-punct';
  }

  // Build a mapping line-index → char offset of that line's start so
  // we can convert the tokenizer's (line, col) positions to source
  // offsets in O(1).
  function computeLineStarts(src) {
    var starts = [0];
    for (var i = 0; i < src.length; i++) {
      if (src.charCodeAt(i) === 10 /* \n */) starts.push(i + 1);
    }
    return starts;
  }

  function escape(s) {
    return s.replace(/[&<>]/g, function (ch) {
      if (ch === '&') return '&amp;';
      if (ch === '<') return '&lt;';
      if (ch === '>') return '&gt;';
      return ch;
    });
  }

  /**
   * Render a FlatPPL source string as syntax-highlighted HTML.
   *
   * @param {string} source  the source text
   * @param {Set<string>} [bindings]  set of names known to be defined
   *        bindings in this source — typically derived by running
   *        `processSource(source).bindings.keys()`. Identifier tokens
   *        matching any of these names get `data-binding="<name>"` and
   *        `class="tok-ident-binding"` so the cross-pane navigation
   *        can target them without re-tokenizing on click.
   * @returns {string} HTML safe to insert via innerHTML
   */
  function highlight(source, bindings, opts) {
    var FE = globalScope.FlatPPLEngine;
    if (!FE || typeof FE.tokenize !== 'function') {
      // Engine missing — degrade gracefully to escaped raw text so
      // the pane still shows something useful.
      return escape(source);
    }
    // Resolve the variant the tokenizer should use. The tokenizer
    // itself is variant-agnostic for the operator set we currently
    // care about (it always emits TILDE / AMPAMP / etc), but passing
    // the variant keeps the public API future-proof.
    var variant = FE.variants && FE.variants.resolveVariant
      ? FE.variants.resolveVariant(opts) : undefined;
    var B = FE.builtins;
    var tokenizeResult = FE.tokenize(source, variant);
    var tokens = tokenizeResult.tokens || [];
    var lineStarts = computeLineStarts(source);

    function offsetOf(line, col) {
      var ls = lineStarts[line];
      return (typeof ls === 'number' ? ls : 0) + col;
    }

    // Group tokens by start line. Tokens are single-line by tokenizer
    // construction (strings end at \n, comments end at \n, numbers
    // never contain whitespace), so a single map keyed by start.line
    // is enough to drive a per-line wrapper layer.
    var tokensByLine = new Map();
    for (var t = 0; t < tokens.length; t++) {
      var tk = tokens[t];
      if (tk.type === 'EOF' || tk.type === 'NEWLINE') continue;
      var ln = tk.loc.start.line;
      if (!tokensByLine.has(ln)) tokensByLine.set(ln, []);
      tokensByLine.get(ln).push(tk);
    }

    // Render token at index `tk` into its classified span. Returns the
    // HTML string; uses `bindings` to decide which idents get the
    // data-binding hook for cross-pane navigation.
    function renderToken(tok) {
      var start = offsetOf(tok.loc.start.line, tok.loc.start.col);
      var end   = offsetOf(tok.loc.end.line,   tok.loc.end.col);
      var cls, extraAttr = '';
      if (tok.type === 'IDENT') {
        cls = classifyIdentifier(tok.value, bindings, B);
        if (bindings && bindings.has(tok.value)) {
          extraAttr = ' data-binding="' + escape(tok.value) + '"';
        }
      } else {
        cls = classifyToken(tok);
      }
      return '<span class="' + cls + '"'
        + ' data-line="' + tok.loc.start.line + '"'
        + ' data-col="'  + tok.loc.start.col  + '"'
        + extraAttr + '>'
        + escape(source.slice(start, end))
        + '</span>';
    }

    // Compute the total line count. lineStarts has one entry per line;
    // a trailing newline produces an empty final line which we still
    // wrap so its `data-line` attribute exists for scroll lookups.
    var totalLines = lineStarts.length;
    if (source.length === 0) totalLines = 1;

    var out = [];
    for (var line = 0; line < totalLines; line++) {
      var lineStart = lineStarts[line];
      // Position of the trailing \n, or source.length for the last line.
      var nextLineStart = (line + 1 < lineStarts.length) ? lineStarts[line + 1] : source.length + 1;
      var lineContentEnd = (nextLineStart > source.length) ? source.length : nextLineStart - 1;

      out.push('<span class="src-line" data-line="' + line + '">');

      var pos = lineStart;
      var lineTokens = tokensByLine.get(line) || [];
      for (var j = 0; j < lineTokens.length; j++) {
        var tok = lineTokens[j];
        var ts = offsetOf(tok.loc.start.line, tok.loc.start.col);
        var te = offsetOf(tok.loc.end.line,   tok.loc.end.col);
        if (ts > pos) out.push(escape(source.slice(pos, ts)));
        out.push(renderToken(tok));
        pos = te;
      }
      // Trailing whitespace within the line (rare, but possible: e.g.
      // a line with only spaces).
      if (pos < lineContentEnd) out.push(escape(source.slice(pos, lineContentEnd)));

      out.push('</span>');

      // Emit the line-terminating newline outside the wrapper so the
      // pre's `white-space: pre` does the line break and copy-paste
      // preserves newlines.
      if (lineContentEnd < source.length) {
        out.push('\n');
      }
    }
    return out.join('');
  }

  globalScope.FlatPPLWebSyntax = {
    highlight: highlight,
  };
})(typeof window !== 'undefined' ? window : globalThis);
