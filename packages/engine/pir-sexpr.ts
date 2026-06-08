'use strict';

// =====================================================================
// pir-sexpr.ts — FlatPIR S-expression printer & reader (spec §11)
// =====================================================================
//
// The engine's in-memory FlatPIR representation is JSON-shaped (see
// pir.ts and lower.ts). Spec §11 defines a canonical S-expression
// surface form for FlatPIR (.flatpir files). This module bridges the
// two:
//
//   toSexpr(loweredModule, opts?) -> string
//     Prints a LoweredModule as a `(%module ...)` form. Indent-pretty
//     by default; pass { compact: true } for a single-line form.
//
//   fromSexpr(text) -> { module, diagnostics }
//     Parses a `.flatpir` text into a LoweredModule. Designed to
//     round-trip toSexpr output and to consume hand-written .flatpir.
//
// Not yet: cross-module loading, %meta annotations (we print the call
// shape but no meta slot today — type/phase are computed by passes
// that consume the in-memory IR, not the S-expr form). Adding meta
// later is a straightforward extension; the parser already accepts it
// (skips %meta forms it doesn't recognise).
//
// Spec mapping (high level):
//   IR { kind: 'lit', value: x }      → bare literal (number / string /
//                                       true / false / null)
//   IR { kind: 'ref', ns, name }      → (%ref <ns> <name>)
//   IR { kind: 'hole' }               → _
//   IR { kind: 'const', name }        → bare symbol (pi, inf, im, etc.)
//   IR { kind: 'axis', name }         → (%axis <name>)
//   IR { kind: 'call', op, args,
//         kwargs, fields, assigns,
//         params, body }              → (<op> [%meta]? args... %kwarg...
//                                              %field... %assign...
//                                              %params... body?)

const pir = require('./pir.ts');

// =====================================================================
// Printer
// =====================================================================

function toSexpr(mod: any, opts?: any) {
  opts = opts || {};
  const indent = opts.compact ? null : '  ';
  const lines: string[] = ['(%module'];
  // %public list
  const publics = Array.from(mod.publicSet || []);
  if (publics.length > 0) {
    lines.push((indent || '') + '(%public ' + publics.join(' ') + ')');
  }
  // bindings
  for (const [name, binding] of mod.bindings) {
    const rhsText = _exprToSexpr(binding.rhs, indent ? '  ' : '');
    // Optional `(%doc <markup> <line>...)` trailing sub-form (spec
    // §11 Documentation). Absent when the binding has no doc OR when
    // the doc has zero content lines (canonical form for undocumented).
    const docText = _docToSexpr(binding.doc);
    const bodyText = docText ? rhsText + ' ' + docText : rhsText;
    if (indent) {
      lines.push('');  // blank line between bindings
      lines.push(indent + '(%bind ' + name + ' ' + bodyText + ')');
    } else {
      lines.push('(%bind ' + name + ' ' + bodyText + ')');
    }
  }
  lines.push(')');
  return indent ? lines.join('\n') : lines.join(' ');
}

function _exprToSexpr(e: any, ind: string): string {
  if (e == null) return 'null';
  if (typeof e !== 'object') return _atomToSexpr(e);
  switch (e.kind) {
    case 'lit':   return _atomToSexpr(e.value);
    case 'hole':  return '_';
    case 'const': return String(e.name);
    case 'axis': {
      // Bare axis (no variance marker) → `(%axis <name>)`. Variance-
      // marked axes inside `metricsum(...)` lower to per-variance heads
      // per spec §11 (FlatPIR): `(%uaxis <name>)` for upper /
      // contravariant, `(%laxis <name>)` for lower / covariant.
      if (e.variance === 'upper') return '(%uaxis ' + e.name + ')';
      if (e.variance === 'lower') return '(%laxis ' + e.name + ')';
      return '(%axis ' + e.name + ')';
    }
    case 'ref':   return '(%ref ' + e.ns + ' ' + e.name + ')';
    case 'call':  return _callToSexpr(e, ind);
    default:
      throw new Error('pir-sexpr: unknown IR kind ' + JSON.stringify(e.kind));
  }
}

// Build the optional `(%doc <markup> <line>...)` sub-form. Returns
// the string OR null when the binding has no documented content.
// Per spec §11: `(%doc md)` with zero content lines is semantically
// the same as omitting the sub-form, and the absent form is canonical
// — so we elide both no-doc bindings and zero-content `%doc md` ones.
function _docToSexpr(doc: any): string | null {
  if (!doc) return null;
  const lines = Array.isArray(doc.lines) ? doc.lines : [];
  if (lines.length === 0) return null;
  const markup = typeof doc.markup === 'string' ? doc.markup : 'md';
  const parts = ['%doc', markup];
  for (const ln of lines) parts.push(JSON.stringify(ln));
  return '(' + parts.join(' ') + ')';
}

function _atomToSexpr(v: any) {
  if (v === true) return 'true';
  if (v === false) return 'false';
  if (v === null) return 'null';
  if (typeof v === 'string') return JSON.stringify(v);
  if (typeof v === 'number') {
    if (!Number.isFinite(v)) {
      if (v === Infinity)  return 'inf';
      if (v === -Infinity) return '(neg inf)';
      return 'nan';
    }
    return String(v);
  }
  // Anything else (an object that's not an IR node) — emit as JSON
  // string for now; spec-strict callers can post-process.
  return JSON.stringify(v);
}

function _callToSexpr(e: any, ind: string): string {
  // FlatPIR (spec §11) distinguishes built-in operations from calls to
  // user-defined callables by expression SHAPE, not name:
  //   - Built-ins are bare-headed: `(add x y)`, `(Normal …)`. lower.ts
  //     records these with `e.op` set to the built-in name.
  //   - User-defined calls use `(%call <ref-head> args…)` where the head is
  //     a `(%ref <ns> <name>)`. lower.ts records these with `e.target =
  //     { ns, name }` and leaves `e.op` undefined.
  // Emit the matching surface form for each so a rewriter's `(%call ?head …)`
  // pattern fires only on user calls and `(add ?x ?y)` only on the built-in.
  const parts: string[] = [];
  if (e.target && typeof e.target === 'object') {
    const ns = e.target.ns != null ? e.target.ns : 'self';
    parts.push('%call');
    parts.push('(%ref ' + ns + ' ' + e.target.name + ')');
  } else {
    parts.push(e.op);
  }
  // Positional args
  if (e.args) {
    for (const a of e.args) parts.push(_exprToSexpr(a, ind));
  }
  // params (functionof / kernelof). Spec §11: the parameter list is a NESTED
  // list `(%params (x y z))`, not a flat `(%params x y z)`.
  if (e.params && e.params.length > 0) {
    parts.push('(%params (' + e.params.join(' ') + '))');
  }
  // kwargs
  if (e.kwargs) {
    const names = Object.keys(e.kwargs);
    for (const k of names) {
      parts.push('(%kwarg ' + k + ' ' + _exprToSexpr(e.kwargs[k], ind) + ')');
    }
  }
  // record / joint / cartprod / table fields
  if (e.fields) {
    for (const f of e.fields) {
      parts.push('(%field ' + f.name + ' ' + _exprToSexpr(f.value, ind) + ')');
    }
  }
  // load_module / standard_module assigns
  if (e.assigns) {
    for (const a of e.assigns) {
      parts.push('(%assign ' + a.name + ' ' + _exprToSexpr(a.value, ind) + ')');
    }
  }
  // body (functionof / kernelof / fn / aggregate)
  if (e.body) {
    parts.push(_exprToSexpr(e.body, ind));
  }
  return '(' + parts.join(' ') + ')';
}

// =====================================================================
// Reader
// =====================================================================
//
// Tokenizer that recognises:
//   ( )          — list parens
//   ;… <eol>     — line comments
//   "..."        — string literal (JSON escapes)
//   123 / 1.5e2  — number literal (int vs real by lexical form)
//   true false   — boolean literal
//   _            — hole
//   %name        — structural keyword (%module, %ref, %kwarg, %field,
//                  %assign, %params, %bind, %public, %meta, %axis, %local,
//                  %deferred, %fixed, %parameterized, %stochastic, …)
//   name         — bare symbol (built-in op name, set name, constant)
//
// The parser builds the same JSON-shaped IR that lower.ts emits, plus
// a list of bindings into a LoweredModule shape consumable by the rest
// of the engine.

// Recursion guard for the descent reader (NIT-2). Adversarial input such
// as `(add (add (add … 1 …)))` nested thousands deep would otherwise
// overflow the JS call stack with a RangeError. We cap the nesting and
// surface a diagnostic instead.
const MAX_DEPTH = 2000;

function fromSexpr(text: any) {
  const tokens = _tokenize(text);
  const diagnostics: any[] = [];
  let pos = 0;
  let depth = 0;
  let depthTripped = false;
  function peek() { return tokens[pos]; }
  function eat() { return tokens[pos++]; }
  function eof() { return pos >= tokens.length; }

  function readForm(): any {
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unexpected EOF' });
      return null;
    }
    const t = peek();
    if (t.type === '(') return readList();
    if (t.type === ')') {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unexpected )' });
      eat();
      return null;
    }
    eat();
    return _atomFromToken(t);
  }

  function readList(): any {
    eat(); // '('
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed (' });
      return null;
    }
    // NIT-2: bound recursion depth. Past the cap we stop descending and
    // consume the rest of this subtree's tokens iteratively (balancing
    // parens) so the token stream stays consistent for any trailing forms.
    if (depth >= MAX_DEPTH) {
      if (!depthTripped) {
        depthTripped = true;
        diagnostics.push({ severity: 'error', message: 'pir-sexpr: nesting too deep' });
      }
      let nest = 1; // we've already consumed this list's opening '('
      while (!eof() && nest > 0) {
        const tk = eat();
        if (tk.type === '(') nest++;
        else if (tk.type === ')') nest--;
      }
      return null;
    }
    depth++;
    try {
      return readListBody();
    } finally {
      depth--;
    }
  }

  function readListBody(): any {
    const head = peek();
    // Head of list determines the form.
    if (head.type === ')') {
      eat();
      return [];  // empty list — invalid in spec, but parse defensively
    }
    if (head.type === 'sym' || head.type === '%kw' || head.type === 'str') {
      if (head.value === '%module')   return readModuleBody();
      if (head.value === '%bind')     return readBindForm();
      if (head.value === '%public')   return readPublicForm();
      if (head.value === '%ref')      return readRefForm();
      if (head.value === '%axis')     return readAxisForm();
      // Variance-marked axes (spec §11): `(%uaxis <name>)` is the
      // contravariant / upper-variance form; `(%laxis <name>)` is the
      // covariant / lower. Both lower to the same `{kind:'axis'}` IR
      // shape, with the `variance` field set, mirroring the AST
      // representation.
      if (head.value === '%uaxis')    return readUAxisForm();
      if (head.value === '%laxis')    return readLAxisForm();
      if (head.value === '%kwarg')    return readKwargForm();
      if (head.value === '%field')    return readFieldForm();
      if (head.value === '%assign')   return readAssignForm();
      if (head.value === '%params')   return readParamsForm();
      if (head.value === '%meta')     return readMetaForm();
      if (head.value === '%call')     return readCallForm();
    }
    // Default: parse as call (head is op name).
    eat();
    const op = head.value;
    // NIT-1: kwargs is a null-prototype map so a crafted `(%kwarg __proto__ …)`
    // becomes an ordinary own key instead of mutating Object.prototype or
    // hiding behind the prototype chain.
    const callShape: any = { kind: 'call', op, args: [], kwargs: Object.create(null) };
    while (!eof() && peek().type !== ')') {
      const part = readForm();
      _absorbCallPart(callShape, part);
    }
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed list (op ' + op + ')' });
    } else {
      eat();  // ')'
    }
    if (Object.keys(callShape.kwargs).length === 0) delete callShape.kwargs;
    if (callShape.args.length === 0) delete callShape.args;
    return callShape;
  }

  function readModuleBody(): any {
    eat(); // %module
    const mod = pir.loweredModule();
    while (!eof() && peek().type !== ')') {
      const sub = readForm();
      if (sub == null) continue;
      if (sub.__kind === 'public') {
        for (const n of sub.names) mod.publicSet.add(n);
      } else if (sub.__kind === 'bind') {
        mod.bindings.set(sub.name,
          pir.loweredBinding(sub.name, sub.rhs, { doc: sub.doc || null }));
      } else {
        diagnostics.push({ severity: 'error', message: 'pir-sexpr: stray form inside %module' });
      }
    }
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed %module' });
    } else {
      eat();
    }
    return { __kind: 'module', module: mod };
  }

  function readBindForm(): any {
    eat(); // %bind
    const nameTok = eat();
    if (!nameTok || nameTok.type !== 'sym') {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: %bind expects a name' });
      return { __kind: 'bind', name: '?', rhs: null };
    }
    const rhs = readForm();
    // Optional trailing `(%doc <markup> <line>...)` sub-form.
    let doc: any = null;
    if (peek() && peek().type === '(') {
      // Peek inside: head must be the `%doc` keyword (lexer emits
      // `%`-prefixed names as type '%kw').
      const save = pos;
      eat(); // (
      const headTok = peek();
      if (headTok && headTok.type === '%kw' && headTok.value === '%doc') {
        eat(); // %doc
        const markupTok = peek();
        const markup = (markupTok && markupTok.type === 'sym')
          ? (eat() && markupTok.value) : 'md';
        const lines: string[] = [];
        while (!eof() && peek().type !== ')') {
          const t = eat();
          if (t.type === 'str') lines.push(t.value);
          else diagnostics.push({
            severity: 'error',
            message: `pir-sexpr: %doc line must be a string, got ${t.type}`,
          });
        }
        if (peek() && peek().type === ')') eat();
        doc = { markup, lines };
      } else {
        // Not a %doc form — rewind so the outer loop sees the `(`.
        pos = save;
      }
    }
    if (peek() && peek().type === ')') eat();
    return { __kind: 'bind', name: nameTok.value, rhs, doc };
  }

  function readPublicForm(): any {
    eat(); // %public
    const names: string[] = [];
    while (!eof() && peek().type !== ')') {
      const t = eat();
      if (t.type === 'sym') names.push(t.value);
    }
    if (peek() && peek().type === ')') eat();
    return { __kind: 'public', names };
  }

  function readRefForm(): any {
    eat(); // %ref
    const ns = eat();
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'ref',
      ns: ns ? ns.value : '?',
      name: name ? name.value : '?' };
  }

  function readAxisForm(): any {
    eat(); // %axis
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'axis', name: name ? name.value : '?' };
  }

  function readUAxisForm(): any {
    eat(); // %uaxis
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'axis', name: name ? name.value : '?', variance: 'upper' };
  }

  function readLAxisForm(): any {
    eat(); // %laxis
    const name = eat();
    if (peek() && peek().type === ')') eat();
    return { kind: 'axis', name: name ? name.value : '?', variance: 'lower' };
  }

  function readKwargForm(): any {
    eat(); // %kwarg
    const name = eat();
    const value = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'kwarg',
      name: name ? name.value : '?',
      value };
  }

  function readFieldForm(): any {
    eat(); // %field
    const name = eat();
    const value = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'field',
      name: name ? name.value : '?',
      value };
  }

  function readAssignForm(): any {
    eat(); // %assign
    const name = eat();
    const value = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'assign',
      name: name ? name.value : '?',
      value };
  }

  function readParamsForm(): any {
    eat(); // %params
    const names: string[] = [];
    // Spec §11: the parameter list is a NESTED list `(%params (x y z))`.
    // Read the inner `(...)` list of bare parameter symbols. We tolerate a
    // legacy flat `(%params x y z)` here too, for any pre-spec input.
    if (peek() && peek().type === '(') {
      eat(); // inner (
      while (!eof() && peek().type !== ')') {
        const t = eat();
        if (t.type === 'sym') names.push(t.value);
      }
      if (peek() && peek().type === ')') eat(); // inner )
    } else {
      while (!eof() && peek().type !== ')') {
        const t = eat();
        if (t.type === 'sym') names.push(t.value);
      }
    }
    if (peek() && peek().type === ')') eat();
    return { __kind: 'params', names };
  }

  function readMetaForm(): any {
    // Spec §11: (%meta <type> <phase>). We round-trip the raw forms
    // but don't currently re-feed them to the inference pipeline.
    eat(); // %meta
    const typ = readForm();
    const ph  = readForm();
    if (peek() && peek().type === ')') eat();
    return { __kind: 'meta', type: typ, phase: ph };
  }

  // Spec §11: `(%call <ref-head> args…)` — a call to a user-defined
  // callable. The head is a `(%ref <ns> <name>)`; we record it as
  // `target: { ns, name }` to mirror lower.ts' user-call shape (the
  // inverse of `_callToSexpr`'s `%call` emission). `%kwarg`/`%field`/
  // `%assign`/`%meta` parts are absorbed the same way as for built-ins.
  function readCallForm(): any {
    eat(); // %call
    // NIT-1: null-prototype kwargs map (see readListBody for rationale).
    const callShape: any = { kind: 'call', args: [], kwargs: Object.create(null) };
    // Head: a `(%ref ns name)` form.
    if (peek() && peek().type === '(') {
      const headForm = readForm();
      if (headForm && headForm.kind === 'ref') {
        // L4: spec §11 allows exactly three `%call` head namespaces:
        // `self`, `%local`, and a module-alias symbol. A `%local` ref
        // names a parameter — not a callable — so it is invalid as a
        // `%call` head and must be rejected with a diagnostic. `self`
        // and module-alias symbols are accepted; we stay lenient about
        // the exact alias spelling so legitimate module heads parse.
        if (headForm.ns === '%local') {
          diagnostics.push({ severity: 'error',
            message: 'pir-sexpr: %call head cannot reference %local '
              + '(parameters are not callable)' });
        }
        callShape.target = { ns: headForm.ns, name: headForm.name };
      } else {
        diagnostics.push({ severity: 'error',
          message: 'pir-sexpr: %call head must be a (%ref …) form' });
      }
    } else {
      diagnostics.push({ severity: 'error',
        message: 'pir-sexpr: %call expects a (%ref …) head' });
    }
    while (!eof() && peek().type !== ')') {
      const part = readForm();
      _absorbCallPart(callShape, part);
    }
    if (eof()) {
      diagnostics.push({ severity: 'error', message: 'pir-sexpr: unclosed (%call …)' });
    } else {
      eat(); // ')'
    }
    if (Object.keys(callShape.kwargs).length === 0) delete callShape.kwargs;
    if (callShape.args.length === 0) delete callShape.args;
    return callShape;
  }

  function _absorbCallPart(call: any, part: any) {
    if (part == null) return;
    if (part.__kind === 'kwarg')  { call.kwargs[part.name] = part.value; return; }
    if (part.__kind === 'field')  { (call.fields ||= []).push({ name: part.name, value: part.value }); return; }
    if (part.__kind === 'assign') { (call.assigns ||= []).push({ name: part.name, value: part.value }); return; }
    if (part.__kind === 'params') { call.params = part.names; return; }
    if (part.__kind === 'meta')   { call.meta = { type: part.type, phase: part.phase }; return; }
    // The body of functionof/kernelof/fn/aggregate is the trailing
    // non-structural expression. We treat the LAST non-structural arg
    // as the body when the op is one of those forms; otherwise add
    // to positional args.
    if ((call.op === 'functionof' || call.op === 'kernelof'
         || call.op === 'fn'        || call.op === 'aggregate')
        && (call.params || call.fields)) {
      call.body = part;
      return;
    }
    call.args.push(part);
  }

  // ---- Top-level: parse a sequence of forms (typically one %module) --

  const out: any = { module: null, diagnostics };
  while (!eof()) {
    const f = readForm();
    if (f && f.__kind === 'module') {
      out.module = f.module;
    }
  }
  if (!out.module) {
    diagnostics.push({ severity: 'warning', message: 'pir-sexpr: no %module form found' });
  }
  return out;
}

function _atomFromToken(t: any) {
  if (t.type === 'str') return { kind: 'lit', value: t.value };
  if (t.type === 'num') return { kind: 'lit', value: t.value };
  if (t.type === 'bool') return { kind: 'lit', value: t.value };
  if (t.type === '_') return { kind: 'hole' };
  if (t.type === 'sym' || t.type === '%kw') {
    return { kind: 'const', name: t.value };
  }
  return null;
}

// ----- Tokenizer ------------------------------------------------------

function _tokenize(text: any) {
  const tokens: any[] = [];
  const s = String(text);
  let i = 0;
  while (i < s.length) {
    const c = s[i];
    if (c === ' ' || c === '\t' || c === '\n' || c === '\r') { i++; continue; }
    if (c === ';') {
      // Comment to EOL
      while (i < s.length && s[i] !== '\n') i++;
      continue;
    }
    if (c === '(') { tokens.push({ type: '(' }); i++; continue; }
    if (c === ')') { tokens.push({ type: ')' }); i++; continue; }
    if (c === '"') {
      let j = i + 1;
      let str = '';
      while (j < s.length && s[j] !== '"') {
        if (s[j] === '\\' && j + 1 < s.length) {
          const esc = s[j + 1];
          if (esc === 'n')      { str += '\n'; j += 2; continue; }
          if (esc === 't')      { str += '\t'; j += 2; continue; }
          if (esc === 'r')      { str += '\r'; j += 2; continue; }
          if (esc === '\\')     { str += '\\'; j += 2; continue; }
          if (esc === '"')      { str += '"';  j += 2; continue; }
          str += esc; j += 2; continue;
        }
        str += s[j]; j++;
      }
      tokens.push({ type: 'str', value: str });
      i = j + 1;
      continue;
    }
    // Atom — read until whitespace or paren.
    let j = i;
    while (j < s.length) {
      const cc = s[j];
      if (cc === ' ' || cc === '\t' || cc === '\n' || cc === '\r'
          || cc === '(' || cc === ')' || cc === ';') break;
      j++;
    }
    const tok = s.slice(i, j);
    i = j;
    if (tok === '_')          { tokens.push({ type: '_' }); continue; }
    if (tok === 'true')       { tokens.push({ type: 'bool', value: true }); continue; }
    if (tok === 'false')      { tokens.push({ type: 'bool', value: false }); continue; }
    // Numeric literal: matches integer or real (incl. scientific notation).
    if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(tok)) {
      tokens.push({ type: 'num', value: Number(tok) });
      continue;
    }
    if (tok[0] === '%') {
      tokens.push({ type: '%kw', value: tok });
      continue;
    }
    tokens.push({ type: 'sym', value: tok });
  }
  return tokens;
}

module.exports = {
  toSexpr,
  fromSexpr,
  // Exported for tests
  _internal: { _tokenize },
};
