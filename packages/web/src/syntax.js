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
  function highlight(source, bindings) {
    var FE = globalScope.FlatPPLEngine;
    if (!FE || typeof FE.tokenize !== 'function') {
      // Engine missing — degrade gracefully to escaped raw text so
      // the pane still shows something useful.
      return escape(source);
    }
    var B = FE.builtins;
    var tokenizeResult = FE.tokenize(source);
    var tokens = tokenizeResult.tokens || [];
    var lineStarts = computeLineStarts(source);

    function offsetOf(line, col) {
      var ls = lineStarts[line];
      return (typeof ls === 'number' ? ls : 0) + col;
    }

    var out = [];
    var pos = 0;
    for (var i = 0; i < tokens.length; i++) {
      var tok = tokens[i];
      // EOF and NEWLINE tokens have zero width or carry only '\n' —
      // we keep them out of the span loop and let the inter-token
      // gap path emit the actual whitespace verbatim.
      if (tok.type === 'EOF') continue;

      var start = offsetOf(tok.loc.start.line, tok.loc.start.col);
      var end   = offsetOf(tok.loc.end.line,   tok.loc.end.col);

      if (start > pos) out.push(escape(source.slice(pos, start)));

      if (tok.type === 'NEWLINE') {
        // Render the newline character as itself. Wrapping it in a
        // span would still work but bloats the markup for no gain.
        out.push(escape(source.slice(start, end)));
        pos = end;
        continue;
      }

      var cls;
      var extraAttr = '';
      if (tok.type === 'IDENT') {
        cls = classifyIdentifier(tok.value, bindings, B);
        if (bindings && bindings.has(tok.value)) {
          extraAttr = ' data-binding="' + escape(tok.value) + '"';
        }
      } else {
        cls = classifyToken(tok);
      }
      out.push('<span class="' + cls + '"'
        + ' data-line="' + tok.loc.start.line + '"'
        + ' data-col="'  + tok.loc.start.col  + '"'
        + extraAttr + '>'
        + escape(source.slice(start, end))
        + '</span>');
      pos = end;
    }
    if (pos < source.length) out.push(escape(source.slice(pos)));
    return out.join('');
  }

  globalScope.FlatPPLWebSyntax = {
    highlight: highlight,
  };
})(typeof window !== 'undefined' ? window : globalThis);
