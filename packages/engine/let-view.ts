'use strict';
// let-view — read IR that holds engine-internal `let` graphs the way a
// recognizer reads the inlined tree.
//
// A `let(assigns, body)` binds each shared value once (derivations.ts,
// materialiser-shared.ts). A recognizer that matches a node's shape, or reads
// a marker on it, sees a ref to a let name where the inlined tree had the
// value. The view renames every let name to a unique one, resolves a name to
// its value, and closes an expression it takes out of the graph over the
// assigns it reads, so the expression evaluates without the enclosing let.

// A scope while renaming: the first `upto` assigns of a let (their fresh
// names in `names`), or the parameters of a `functionof`, which hide outer
// names.
interface RenameScope {
  assigns?: any[]; names?: string[]; upto?: number; hides?: Set<string>;
  parent: RenameScope | null; memo: WeakMap<object, any>;
}

// `body` with every `let` name made unique, so no assign shadows another (two
// calls of one function bind its formals twice). Each assign joins `bound`
// after the assigns its value reads.
function renameLets(body: any, bound: Map<string, { value: any; order: number }>): any {
  const lookup = (s: RenameScope | null, name: string): string | null => {
    for (; s; s = s.parent) {
      if (s.hides) { if (s.hides.has(name)) return null; continue; }
      for (let j = s.upto! - 1; j >= 0; j--) if (s.assigns![j].name === name) return s.names![j];
    }
    return null;
  };
  const rootMemo = new WeakMap<object, any>();
  const walk = (n: any, s: RenameScope | null): any => {
    if (!n || typeof n !== 'object') return n;
    if (Array.isArray(n)) {
      let out: any[] | null = null;
      n.forEach((c, i) => { const w = walk(c, s); if (w !== c) { out = out || n.slice(); out[i] = w; } });
      return out || n;
    }
    const memo = s ? s.memo : rootMemo;
    if (memo.has(n)) return memo.get(n);
    let result: any;
    if (n.kind === 'ref') {
      const name = lookup(s, n.name);
      result = name ? { ...n, ns: '%local', name } : n;
    } else if (n.kind === 'call' && n.op === 'let') {
      const names: string[] = [];
      const assigns: any[] = [];
      for (let i = 0; i < n.assigns.length; i++) {
        const value = walk(n.assigns[i].value,
          { assigns: n.assigns, names, upto: i, parent: s, memo: new WeakMap() });
        const name = '%wb:' + bound.size + ':' + n.assigns[i].name;
        names.push(name);
        bound.set(name, { value, order: bound.size });
        assigns.push({ name, value });
      }
      const inner = { assigns: n.assigns, names, upto: n.assigns.length, parent: s, memo: new WeakMap() };
      result = { ...n, assigns, body: walk(n.body, inner) };
    } else if (s && n.kind === 'call' && n.op === 'functionof' && Array.isArray(n.params)) {
      result = { ...n, body: walk(n.body, { hides: new Set(n.params), parent: s, memo: new WeakMap() }) };
    } else {
      let out: any = null;
      for (const k in n) {
        if (k === 'loc' || k === 'meta') continue;
        const w = walk(n[k], s);
        if (w !== n[k]) { out = out || { ...n }; out[k] = w; }
      }
      result = out || n;
    }
    memo.set(n, result);
    return result;
  };
  return walk(body, null);
}

type Bound = Map<string, { value: any; order: number }>;

interface LetView {
  // `ir` with unique let names.
  ir: any;
  bound: Bound;
  isBound(n: any): boolean;
  // `n` with let names and let wrappers stripped from its top.
  resolve(n: any): any;
  // `e` wrapped in a `let` of the assigns it reads, in dependency order.
  close(e: any): any;
  // The resolved top node with each direct child (args, kwargs, fields)
  // closed: a shape matcher reads it as the inlined node. Without a let it
  // is the input itself.
  top(): any;
  // `base`, a recognizer's own "does this subtree read the variate", also
  // true of a let name whose value reads it. Memoized.
  mentions(base: (n: any) => boolean): (n: any) => boolean;
  // `ir` with each let name replaced by its value, one object per value. For
  // a constant folder, whose expressions are small.
  expand(): any;
}


function letView(ir: any): LetView {
  const bound: Bound = new Map();
  const renamed = renameLets(ir, bound);
  const isBound = (n: any) => !!n && n.kind === 'ref' && bound.has(n.name);
  const resolve = (n: any): any => {
    for (;;) {
      if (isBound(n)) n = bound.get(n.name)!.value;
      else if (n && n.kind === 'call' && n.op === 'let') n = n.body;
      else return n;
    }
  };
  const close = (e: any): any => {
    if (bound.size === 0) return e;
    const need = new Set<string>();
    const seen = new WeakSet<object>();
    const collect = (n: any) => {
      if (!n || typeof n !== 'object' || seen.has(n)) return;
      seen.add(n);
      if (Array.isArray(n)) { n.forEach(collect); return; }
      if (isBound(n) && !need.has(n.name)) { need.add(n.name); collect(bound.get(n.name)!.value); return; }
      for (const k in n) if (k !== 'loc' && k !== 'meta') collect(n[k]);
    };
    collect(e);
    if (need.size === 0) return e;
    const assigns = [...need].sort((p, q) => bound.get(p)!.order - bound.get(q)!.order)
      .map((name) => ({ name, value: bound.get(name)!.value }));
    return { kind: 'call', op: 'let', assigns, body: e };
  };
  const top = (): any => {
    if (bound.size === 0) return ir;
    const n = resolve(renamed);
    if (!n || n.kind !== 'call') return n;
    const out: any = { ...n };
    if (Array.isArray(n.args)) out.args = n.args.map(close);
    if (n.kwargs) {
      out.kwargs = {};
      for (const k in n.kwargs) out.kwargs[k] = close(n.kwargs[k]);
    }
    if (Array.isArray(n.fields)) out.fields = n.fields.map((f: any) => ({ ...f, value: close(f.value) }));
    return out;
  };
  const mentions = (base: (n: any) => boolean) => {
    if (bound.size === 0) return base;
    const memo = new WeakMap<object, boolean>();
    const viaBound = (n: any): boolean => {
      if (!n || typeof n !== 'object') return false;
      if (Array.isArray(n)) return n.some(viaBound);
      if (memo.has(n)) return memo.get(n)!;
      memo.set(n, false);
      let r = false;
      if (isBound(n)) r = m(bound.get(n.name)!.value);
      else for (const k in n) if (k !== 'loc' && k !== 'meta' && viaBound(n[k])) { r = true; break; }
      memo.set(n, r);
      return r;
    };
    const m = (n: any): boolean => base(n) || viaBound(n);
    return m;
  };
  const memo = new Map<any, any>();
  const expandNode = (n: any): any => {
    if (!n || typeof n !== 'object') return n;
    if (memo.has(n)) return memo.get(n);
    let out: any;
    if (isBound(n)) out = expandNode(bound.get(n.name)!.value);
    else if (n.kind === 'call' && n.op === 'let') out = expandNode(n.body);
    else if (Array.isArray(n)) {
      out = n;
      n.forEach((c: any, i: number) => { const w = expandNode(c); if (w !== c) { if (out === n) out = n.slice(); out[i] = w; } });
    } else {
      out = n;
      for (const k in n) {
        if (k === 'loc' || k === 'meta') continue;
        const w = expandNode(n[k]);
        if (w !== n[k]) { if (out === n) out = { ...n }; out[k] = w; }
      }
    }
    memo.set(n, out);
    return out;
  };
  const expand = (): any => (bound.size === 0 ? ir : expandNode(renamed));
  return { ir: renamed, bound, isBound, resolve, close, top, mentions, expand };
}

// The top node of a set or support IR a shape matcher reads: `top()` of a
// `let`, otherwise the IR itself.
function topOf(ir: any): any {
  return ir && ir.kind === 'call' && ir.op === 'let' ? letView(ir).top() : ir;
}

module.exports = { letView, topOf };
