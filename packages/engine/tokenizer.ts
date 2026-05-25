'use strict';

// Token types
const T = {
  IDENT: 'IDENT',
  NUMBER: 'NUMBER',
  STRING: 'STRING',
  PLACEHOLDER: 'PLACEHOLDER', // _name_
  HOLE: 'HOLE',               // bare _

  LPAREN: 'LPAREN', RPAREN: 'RPAREN',
  LBRACKET: 'LBRACKET', RBRACKET: 'RBRACKET',
  COMMA: 'COMMA',
  SEMI: 'SEMI',         // ;  (FlatPPJ semicolon kwargs)
  DOT: 'DOT',
  COLON: 'COLON',
  EQUALS: 'EQUALS',     // =
  TILDE: 'TILDE',       // ~  (FlatPPL/FlatPPJ tilde bindings)
  EQEQ: 'EQEQ',        // ==
  NEQ: 'NEQ',           // !=
  LT: 'LT', GT: 'GT',
  LTE: 'LTE', GTE: 'GTE',
  PLUS: 'PLUS', MINUS: 'MINUS',
  STAR: 'STAR', SLASH: 'SLASH',
  CARET: 'CARET',       // ^  (FlatPPL/FlatPPJ exponentiation)
  AMPAMP: 'AMPAMP',     // && (FlatPPL/FlatPPJ logical AND)
  PIPEPIPE: 'PIPEPIPE', // || (FlatPPL/FlatPPJ logical OR)
  BANG: 'BANG',         // !  (FlatPPL/FlatPPJ logical NOT)
  ARROW: 'ARROW',       // -> (lambda; spec §05 Lambda syntax)
  COLON_EQ: 'COLON_EQ', // := (aggregate shorthand; spec §05 Axis names)

  NEWLINE: 'NEWLINE',
  COMMENT: 'COMMENT',
  EOF: 'EOF',
};

function token(type: string, value: any, startLine: number, startCol: number, endLine: number, endCol: number): any {
  return {
    type, value,
    loc: { start: { line: startLine, col: startCol }, end: { line: endLine, col: endCol } },
  };
}

// At the start of a line iff there's no preceding token on this line.
// We track this through the emitted-token stream: a NEWLINE at the
// tail means we're starting a fresh line; an empty stream means
// start of source.
function _atLineStart(tokens: any[]) {
  if (tokens.length === 0) return true;
  return tokens[tokens.length - 1].type === 'NEWLINE';
}

function isAlpha(ch: string) { return /[a-zA-Z]/.test(ch); }
function isDigit(ch: string) { return /[0-9]/.test(ch); }
function isHexDigit(ch: string) { return /[0-9a-fA-F]/.test(ch); }
function isAlphaNum(ch: string) { return /[a-zA-Z0-9]/.test(ch); }
function isIdentStart(ch: string) { return isAlpha(ch) || ch === '_'; }
function isIdentChar(ch: string) { return isAlphaNum(ch) || ch === '_'; }
function isWhitespace(ch: string) { return ch === ' ' || ch === '\t' || ch === '\r'; }

/**
 * Tokenize FlatPPL source text into an array of tokens.
 * Returns { tokens: Token[], diagnostics: Diagnostic[] }.
 *
 * `variant` is the surface-syntax variant (see ./variants). The base
 * tokenizer is variant-agnostic; per-variant token additions live in
 * later commits in this series. `variant` is accepted (and ignored)
 * starting now so call sites can pass it without churn.
 */
function tokenize(source: string, variant: any) {  // eslint-disable-line no-unused-vars
  const tokens: any[] = [];
  const diagnostics: any[] = [];
  let pos = 0;
  let line = 0;
  let col = 0;
  let depth = 0; // paren/bracket nesting depth

  function peek(offset: number) { return source[pos + (offset || 0)] || ''; }
  function advance() {
    const ch = source[pos++];
    if (ch === '\n') { line++; col = 0; } else { col++; }
    return ch;
  }
  function at(offset?: number) { return pos + (offset || 0) < source.length ? source[pos + (offset || 0)] : ''; }

  while (pos < source.length) {
    const startLine = line, startCol = col;
    const ch = at();

    // Whitespace (not newline)
    if (isWhitespace(ch)) {
      advance();
      continue;
    }

    // Newline
    if (ch === '\n') {
      advance();
      if (depth === 0) {
        tokens.push(token(T.NEWLINE, '\n', startLine, startCol, line, col));
      }
      continue;
    }

    // Comments (spec §05). Two forms:
    //   - Block comment `###` on its own line (with optional leading
    //     horizontal whitespace), closed by a matching `###` on its
    //     own line. Content between fences is consumed and discarded.
    //   - Line comment `#`, terminated by the next newline OR
    //     semicolon — whichever comes first — so source survives
    //     newline-collapsing transport channels (spec §05 Statements:
    //     newlines and `;` are equivalent separators).
    if (ch === '#') {
      // Block-comment fence detection: `###` AT THE START OF A LINE
      // (only horizontal whitespace before it on the line). We
      // approximate "start of line" by checking that the previous
      // emitted token is NEWLINE (or there is no previous token —
      // start of source). Tab/space already consumed by the
      // whitespace branch above; line/col still point at the `#`.
      if (peek(1) === '#' && peek(2) === '#'
          && (peek(3) === '\n' || peek(3) === '' ||
              (peek(3) === ' ' || peek(3) === '\t'))
          && _atLineStart(tokens)) {
        // Consume the opening fence and any trailing whitespace
        // through the newline.
        advance(); advance(); advance();
        while (pos < source.length && at() !== '\n') advance();
        if (at() === '\n') advance();
        // Consume body lines until a closing `###` on its own line.
        while (pos < source.length) {
          // Save start of this line.
          const lineStart = pos;
          // Skip leading horizontal whitespace.
          let p = pos;
          while (p < source.length && (source[p] === ' ' || source[p] === '\t')) p++;
          if (source[p] === '#' && source[p + 1] === '#' && source[p + 2] === '#'
              && (p + 3 === source.length
                  || source[p + 3] === '\n'
                  || source[p + 3] === ' '
                  || source[p + 3] === '\t')) {
            // Closing fence; consume through end-of-line.
            while (pos < source.length && at() !== '\n') advance();
            if (at() === '\n') advance();
            break;
          }
          // Not a fence — consume the whole line.
          while (pos < source.length && at() !== '\n') advance();
          if (at() === '\n') advance();
        }
        // Block comment is transparent to the parser — emit no
        // token. The block spanned ≥1 newline(s) which would have
        // been emitted as NEWLINE if we'd taken the whitespace path,
        // so we synthesise a single NEWLINE here at depth 0 to keep
        // statement-separator behaviour intact.
        if (depth === 0) {
          tokens.push(token(T.NEWLINE, '\n', startLine, startCol, line, col));
        }
        continue;
      }
      // Line comment.
      let text = '';
      while (pos < source.length && at() !== '\n' && at() !== ';') {
        text += advance();
      }
      tokens.push(token(T.COMMENT, text, startLine, startCol, line, col));
      continue;
    }

    // String literal
    if (ch === '"') {
      advance(); // opening quote
      let value = '';
      let raw = '"';
      while (pos < source.length && at() !== '"' && at() !== '\n') {
        if (at() === '\\' && pos + 1 < source.length) {
          const escStartLine = line, escStartCol = col;
          raw += advance(); // backslash
          raw += advance(); // escaped char
          const esc = raw[raw.length - 1];
          if (esc === 'n') value += '\n';
          else if (esc === 't') value += '\t';
          else if (esc === 'r') value += '\r';
          else if (esc === '0') value += '\0';
          else if (esc === '\\') value += '\\';
          else if (esc === '"') value += '"';
          else {
            diagnostics.push({
              severity: 'error',
              message: `Invalid escape sequence '\\${esc}'`,
              loc: { start: { line: escStartLine, col: escStartCol }, end: { line, col } },
            });
            value += esc;
          }
        } else {
          const c = advance();
          raw += c;
          value += c;
        }
      }
      if (at() === '"') {
        raw += advance(); // closing quote
      } else {
        diagnostics.push({ severity: 'error', message: 'Unterminated string literal', loc: { start: { line: startLine, col: startCol }, end: { line, col } } });
      }
      tokens.push(token(T.STRING, value, startLine, startCol, line, col));
      continue;
    }

    // Number literal (decimal, hex, with optional underscores between digits)
    if (isDigit(ch) || (ch === '.' && isDigit(at(1)))) {
      let num = '';
      // Hex: 0x HexDigit (_? HexDigit)*  — at least one hex digit
      // required. Lowercase `0x` only, per spec §05 HexIntLit (matches
      // Julia/Rust; the uppercase `0X` C-family form is intentionally
      // not canonical FlatPPL).
      if (ch === '0' && at(1) === 'x') {
        num += advance(); // 0
        num += advance(); // x
        if (!isHexDigit(at())) {
          diagnostics.push({
            severity: 'error',
            message: 'Hex literal requires at least one hex digit after 0x',
            loc: { start: { line: startLine, col: startCol }, end: { line, col } },
          });
        }
        while (pos < source.length && (isHexDigit(at()) || (at() === '_' && isHexDigit(at(1))))) {
          num += advance();
        }
        tokens.push(token(T.NUMBER, num, startLine, startCol, line, col));
        continue;
      }
      // Decimal integer part
      while (pos < source.length && (isDigit(at()) || (at() === '_' && isDigit(at(1))))) {
        num += advance();
      }
      // Fractional part
      if (at() === '.' && at(1) !== '.') {
        num += advance(); // dot
        while (pos < source.length && (isDigit(at()) || (at() === '_' && isDigit(at(1))))) {
          num += advance();
        }
      }
      // Exponent
      if (at() === 'e' || at() === 'E') {
        num += advance(); // e/E
        if (at() === '+' || at() === '-') num += advance();
        while (pos < source.length && (isDigit(at()) || (at() === '_' && isDigit(at(1))))) {
          num += advance();
        }
      }
      tokens.push(token(T.NUMBER, num, startLine, startCol, line, col));
      continue;
    }

    // Identifier, placeholder, hole, or keyword
    if (isIdentStart(ch)) {
      let ident = '';
      const iStart = pos;
      while (pos < source.length && isIdentChar(at())) ident += advance();

      // Distinguish: bare _ (hole) vs _name_ (placeholder) vs regular ident
      if (ident === '_') {
        tokens.push(token(T.HOLE, '_', startLine, startCol, line, col));
      } else if (ident.length >= 3 && ident[0] === '_' && ident[ident.length - 1] === '_'
                 && isAlpha(ident[1])) {
        // _name_ pattern — placeholder
        const innerName = ident.slice(1, -1);
        tokens.push(token(T.PLACEHOLDER, innerName, startLine, startCol, line, col));
      } else {
        tokens.push(token(T.IDENT, ident, startLine, startCol, line, col));
      }
      continue;
    }

    // Two-character operators. Order matters: check `==` before `=`,
    // `!=` before `!`, `<=` before `<`, `>=` before `>`, `&&` before
    // any bitwise-and (which FlatPPL doesn't have but we'd want to
    // reject cleanly), and `||` before any bitwise-or.
    if (ch === '=' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.EQEQ, '==', startLine, startCol, line, col));
      continue;
    }
    if (ch === '!' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.NEQ, '!=', startLine, startCol, line, col));
      continue;
    }
    if (ch === '<' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.LTE, '<=', startLine, startCol, line, col));
      continue;
    }
    if (ch === '>' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.GTE, '>=', startLine, startCol, line, col));
      continue;
    }
    if (ch === '&' && at(1) === '&') {
      advance(); advance();
      tokens.push(token(T.AMPAMP, '&&', startLine, startCol, line, col));
      continue;
    }
    if (ch === '|' && at(1) === '|') {
      advance(); advance();
      tokens.push(token(T.PIPEPIPE, '||', startLine, startCol, line, col));
      continue;
    }
    // `->` — lambda arrow. Must be checked before the single-char `-`
    // path. Per spec §05 Lambda syntax, `(arg, …) -> expr` lowers to
    // `functionof(expr-with-placeholders, arg = _arg_, …)`.
    if (ch === '-' && at(1) === '>') {
      advance(); advance();
      tokens.push(token(T.ARROW, '->', startLine, startCol, line, col));
      continue;
    }
    // `:=` — aggregate shorthand (spec §05 Axis names). Must be checked
    // before the single-char `:` (used for slicing). `C[.i, .k] := expr`
    // lowers to `C = aggregate(sum, [.i, .k], expr)`.
    if (ch === ':' && at(1) === '=') {
      advance(); advance();
      tokens.push(token(T.COLON_EQ, ':=', startLine, startCol, line, col));
      continue;
    }

    // Dot-prefixed (broadcast) operators, spec §05 Broadcasting syntax:
    // `.+ .- .* ./ .^ .< .> .<= .>= .== .!= .&& .|| .!`. Each is a
    // single token recognized by maximal munch (longest match — the
    // 3-char forms are tried before their 2-char prefixes). They carry
    // the SAME token type and value as the plain operator plus a
    // `dotted:true` marker, so the precedence parser matches them with
    // zero change and only the desugaring branches inspect the marker.
    //
    // Reaching here means `ch === '.'` was NOT consumed by the number
    // branch (which already ate `.`+digit, so `1./x` munched `1.` as
    // the real literal `1.0` — the Julia rule, spec §05). `.(`
    // (dot-call) and `.name` (field access) are left as a bare DOT for
    // the parser, since `(` and identifier-starts are not operator
    // chars below.
    if (ch === '.') {
      const DOT_OPS_3: Record<string, string> = {
        '.==': T.EQEQ, '.!=': T.NEQ, '.<=': T.LTE, '.>=': T.GTE,
        '.&&': T.AMPAMP, '.||': T.PIPEPIPE,
      };
      const DOT_OPS_2: Record<string, string> = {
        '.<': T.LT, '.>': T.GT, '.+': T.PLUS, '.-': T.MINUS,
        '.*': T.STAR, './': T.SLASH, '.^': T.CARET, '.!': T.BANG,
      };
      const three = ch + at(1) + at(2);
      const two = ch + at(1);
      let lexeme: string | null = null, ttype: any = null;
      if (DOT_OPS_3[three]) { lexeme = three; ttype = DOT_OPS_3[three]; }
      else if (DOT_OPS_2[two]) { lexeme = two; ttype = DOT_OPS_2[two]; }
      if (lexeme) {
        for (let k = 0; k < lexeme.length; k++) advance();
        const tok = token(ttype, lexeme.slice(1), startLine, startCol, line, col);
        tok.dotted = true;
        tokens.push(tok);
        continue;
      }
    }

    // Semicolon at depth 0: statement separator (spec §05 Statements:
    // newline and `;` are equivalent). Emitted as NEWLINE so the
    // parser's existing statement-separator path applies unchanged.
    // Inside parens/brackets (depth > 0), `;` stays SEMI for FlatPPJ's
    // positional/kwargs split (`f(x, y; a = 1)`).
    if (ch === ';' && depth === 0) {
      advance();
      tokens.push(token(T.NEWLINE, ';', startLine, startCol, line, col));
      continue;
    }

    // Single-character tokens
    const singleMap: Record<string, string> = {
      '(': T.LPAREN, ')': T.RPAREN,
      '[': T.LBRACKET, ']': T.RBRACKET,
      ',': T.COMMA, ';': T.SEMI,
      '.': T.DOT, ':': T.COLON,
      '=': T.EQUALS, '~': T.TILDE,
      '<': T.LT, '>': T.GT,
      '+': T.PLUS, '-': T.MINUS,
      '*': T.STAR, '/': T.SLASH,
      '^': T.CARET, '!': T.BANG,
    };
    if (singleMap[ch]) {
      const tt = singleMap[ch];
      advance();
      if (tt === T.LPAREN || tt === T.LBRACKET) depth++;
      else if (tt === T.RPAREN || tt === T.RBRACKET) depth = Math.max(0, depth - 1);
      tokens.push(token(tt, ch, startLine, startCol, line, col));
      continue;
    }

    // Unknown character
    const uch = advance();
    diagnostics.push({
      severity: 'error',
      message: `Unexpected character '${uch}'`,
      loc: { start: { line: startLine, col: startCol }, end: { line, col } },
    });
  }

  tokens.push(token(T.EOF, '', line, col, line, col));
  return { tokens, diagnostics };
}

module.exports = { T, tokenize };
