// @ts-nocheck — test support only; never bundled (index.ts does not import it).
//
// A small serialisable DOM for whole-mount tests in Node. It records every
// element mount() creates, with its attributes, classes, inline style and
// text, so a test can compare what a host receives before and after a change.
// The viewer touches only this surface; a missing method fails loudly.

import { createHash } from 'node:crypto';

export interface TestDom {
  document: any;
  window: any;
  listeners: { window: string[]; document: string[] };
  serialize(root: any): any;
  restore(): void;
}

export function installTestDom(): TestDom {
  const ids = new Map();
  const listeners = { window: [] as string[], document: [] as string[] };

  class Style {
    [key: string]: any;
    setProperty(name: string, value: string) { this[name] = String(value); }
    getPropertyValue(name: string) { return this[name] || ''; }
    removeProperty(name: string) { delete this[name]; }
  }

  class Element extends EventTarget {
    tagName: string; children: any[] = []; parentNode: any = null; style = new Style();
    attributes = new Map<string, string>(); classes = new Set<string>();
    _text = ''; _html: string | null = null; _id = '';
    hidden = false; disabled = false; title = ''; scrollTop = 0; tabIndex = -1;
    ownerDocument: any;
    _listeners = new Map<string, any[]>();
    constructor(tag: string) { super(); this.tagName = tag.toUpperCase(); this.ownerDocument = doc; }
    // Own listener registry so events bubble through the element tree with
    // the original target, as in a browser.
    addEventListener(type: string, fn: any) {
      if (!fn) return;
      const list = this._listeners.get(type) || [];
      if (!list.includes(fn)) list.push(fn);
      this._listeners.set(type, list);
    }
    removeEventListener(type: string, fn: any) {
      const list = this._listeners.get(type);
      if (list) this._listeners.set(type, list.filter((f) => f !== fn));
    }
    dispatchEvent(ev: any) {
      let stopped = false;
      const stop = ev.stopPropagation.bind(ev);
      Object.defineProperty(ev, 'stopPropagation', { value: () => { stopped = true; stop(); }, configurable: true });
      Object.defineProperty(ev, 'target', { value: this, configurable: true });
      for (let node: any = this; node && node._listeners; node = node.parentNode) {
        if (node !== this && !ev.bubbles) break;
        Object.defineProperty(ev, 'currentTarget', { value: node, configurable: true });
        for (const fn of (node._listeners.get(ev.type) || []).slice()) {
          if (typeof fn === 'function') fn.call(node, ev); else fn.handleEvent(ev);
        }
        if (stopped) break;
      }
      return !ev.defaultPrevented;
    }
    get id() { return this._id; }
    set id(v) { this._id = v; ids.set(v, this); }
    get className() { return Array.from(this.classes).join(' '); }
    set className(v) { this.classes = new Set(String(v).split(/\s+/).filter(Boolean)); }
    get classList() {
      const c = this.classes;
      return {
        add: (...n: string[]) => n.forEach((x) => c.add(x)),
        remove: (...n: string[]) => n.forEach((x) => c.delete(x)),
        contains: (n: string) => c.has(n),
        toggle: (n: string, on?: boolean) => {
          const want = on === undefined ? !c.has(n) : !!on;
          if (want) c.add(n); else c.delete(n);
          return want;
        },
      };
    }
    get textContent() { return this._html != null ? this._html.replace(/<[^>]*>/g, '') : this._text + this.children.map((c) => c.textContent).join(''); }
    set textContent(v) { this._text = String(v); this._html = null; this.children.forEach((c) => { c.parentNode = null; }); this.children = []; }
    get innerHTML() { return this._html != null ? this._html : this._text; }
    set innerHTML(html) {
      this._html = String(html); this._text = '';
      this.children.forEach((c) => { c.parentNode = null; });
      this.children = [];
      // Elements with an id become real children, so $() and listeners work.
      for (const m of this._html.matchAll(/<([a-z][a-z0-9]*)([^>]*?)\sid="([^"]+)"([^>]*)>/g)) {
        const child = new Element(m[1]);
        const attrs = (m[2] + ' ' + m[4]).matchAll(/([a-z-]+)="([^"]*)"/g);
        for (const a of attrs) {
          if (a[1] === 'class') child.className = a[2];
          else child.setAttribute(a[1], a[2]);
        }
        if (/\shidden(\s|$|>)/.test(' ' + m[2] + ' ' + m[4] + ' ')) child.hidden = true;
        child.id = m[3];
        this.appendChild(child);
      }
    }
    get firstChild() { return this.children[0] || null; }
    get lastChild() { return this.children[this.children.length - 1] || null; }
    get childNodes() { return this.children; }
    get parentElement() { return this.parentNode; }
    get nextSibling() {
      const p = this.parentNode; if (!p) return null;
      return p.children[p.children.indexOf(this) + 1] || null;
    }
    get isConnected() { let n = this; while (n.parentNode) n = n.parentNode; return n === doc.body || n === doc.head; }
    setAttribute(n: string, v: any) { this.attributes.set(n, String(v)); if (n === 'id') this.id = String(v); }
    getAttribute(n: string) { return this.attributes.has(n) ? this.attributes.get(n) : null; }
    hasAttribute(n: string) { return this.attributes.has(n); }
    removeAttribute(n: string) { this.attributes.delete(n); }
    appendChild(c: any) {
      if (c.tagName === 'FRAGMENT') { c.children.slice().forEach((x) => this.appendChild(x)); c.children = []; return c; }
      if (c.parentNode) c.parentNode.removeChild(c);
      c.parentNode = this; this.children.push(c); return c;
    }
    append(...cs: any[]) { cs.forEach((c) => this.appendChild(typeof c === 'string' ? doc.createTextNode(c) : c)); }
    insertBefore(c: any, ref: any) {
      if (!ref) return this.appendChild(c);
      if (c.parentNode) c.parentNode.removeChild(c);
      c.parentNode = this; this.children.splice(this.children.indexOf(ref), 0, c); return c;
    }
    insertAdjacentHTML(_pos: string, html: string) { const tmp = new Element('div'); tmp.innerHTML = html; tmp.children.slice().forEach((c) => this.appendChild(c)); }
    removeChild(c: any) { const i = this.children.indexOf(c); if (i >= 0) this.children.splice(i, 1); c.parentNode = null; return c; }
    remove() { if (this.parentNode) this.parentNode.removeChild(this); }
    replaceChildren(...cs: any[]) { this.children.forEach((c) => { c.parentNode = null; }); this.children = []; this._text = ''; this._html = null; this.append(...cs); }
    contains(n: any) { return this === n || this.children.some((c) => c.contains && c.contains(n)); }
    matches(sel: string) { return sel.split(',').some((s) => matchOne(this, s.trim())); }
    closest(sel: string) { let n = this; while (n && n.matches) { if (n.matches(sel)) return n; n = n.parentNode; } return null; }
    querySelectorAll(sel: string) {
      const out: any[] = [];
      const walk = (n: any) => n.children.forEach((c) => { if (c.matches && c.matches(sel)) out.push(c); if (c.children) walk(c); });
      walk(this); return out;
    }
    querySelector(sel: string) { return this.querySelectorAll(sel)[0] || null; }
    focus() { doc.activeElement = this; }
    blur() { if (doc.activeElement === this) doc.activeElement = doc.body; }
    click() { this.dispatchEvent(new Event('click', { bubbles: true, cancelable: true })); }
    getBoundingClientRect() { return { top: 0, left: 0, right: 400, bottom: 300, width: 400, height: 300 }; }
    getClientRects() { return [this.getBoundingClientRect()]; }
    get clientWidth() { return 400; }
    get clientHeight() { return 300; }
    get offsetWidth() { return 400; }
    get offsetHeight() { return 300; }
    scrollIntoView() {}
    getContext() { return null; }
    setPointerCapture() {}
    releasePointerCapture() {}
    hasPointerCapture() { return false; }
  }

  function matchOne(el: any, s: string): boolean {
    if (s.startsWith('#')) return el.id === s.slice(1);
    if (s.startsWith('.')) return el.classes.has(s.slice(1));
    const attr = s.match(/^([a-z0-9]*)\[([a-z-]+)(?:="([^"]*)")?\]$/i);
    if (attr) {
      if (attr[1] && el.tagName !== attr[1].toUpperCase()) return false;
      if (!el.attributes.has(attr[2])) return false;
      return attr[3] === undefined || el.attributes.get(attr[2]) === attr[3];
    }
    const cls = s.match(/^([a-z0-9]+)\.([a-z0-9_-]+)$/i);
    if (cls) return el.tagName === cls[1].toUpperCase() && el.classes.has(cls[2]);
    return el.tagName === s.toUpperCase();
  }

  class Text { nodeType = 3; parentNode: any = null; data: string; constructor(data: string) { this.data = data; } get textContent() { return this.data; } }

  const doc: any = new EventTarget();
  doc.getElementById = (id: string) => {
    const el = ids.get(id);
    return el && el.isConnected ? el : null;
  };
  doc.createElement = (tag: string) => new Element(tag);
  doc.createElementNS = (_ns: string, tag: string) => new Element(tag);
  doc.createDocumentFragment = () => new Element('fragment');
  doc.createTextNode = (t: string) => new Text(String(t));
  doc.head = new Element('head');
  doc.body = new Element('body');
  doc.documentElement = new Element('html');
  doc.activeElement = doc.body;
  doc.readyState = 'complete';
  doc.baseURI = 'https://viewer.test/';
  const docAdd = doc.addEventListener.bind(doc);
  doc.addEventListener = (type: string, fn: any, opts?: any) => { listeners.document.push(type); docAdd(type, fn, opts); };
  doc.querySelector = (sel: string) => doc.body.querySelector(sel) || doc.head.querySelector(sel);
  doc.querySelectorAll = (sel: string) => [...doc.head.querySelectorAll(sel), ...doc.body.querySelectorAll(sel)];

  const win: any = new EventTarget();
  const winAdd = win.addEventListener.bind(win);
  win.addEventListener = (type: string, fn: any, opts?: any) => { listeners.window.push(type); winAdd(type, fn, opts); };
  win.location = { origin: 'https://viewer.test', href: 'https://viewer.test/' };
  win.innerWidth = 1200; win.innerHeight = 800;
  win.document = doc;

  const saved = new Map<string, PropertyDescriptor | undefined>();
  const globals: Record<string, any> = {
    document: doc, window: win,
    requestAnimationFrame: (fn: any) => { fn(); return 0; },
    cancelAnimationFrame: () => {},
    getComputedStyle: () => new Style(),
    ResizeObserver: undefined, MutationObserver: undefined,
  };
  for (const [k, v] of Object.entries(globals)) {
    saved.set(k, Object.getOwnPropertyDescriptor(globalThis, k));
    Object.defineProperty(globalThis, k, { value: v, configurable: true, writable: true });
  }

  function hash(s: string) { return createHash('sha256').update(s).digest('hex').slice(0, 16); }

  function serialize(n: any): any {
    if (n instanceof Text) return { text: n.data };
    const out: any = { tag: n.tagName.toLowerCase() };
    if (n.id) out.id = n.id;
    const attrs = Array.from(n.attributes.entries()).filter(([k]) => k !== 'id').sort();
    if (attrs.length) out.attrs = Object.fromEntries(attrs);
    if (n.classes.size) out.classes = Array.from(n.classes).sort();
    if (n.hidden) out.hidden = true;
    if (n.disabled) out.disabled = true;
    if (n.title) out.title = n.title;
    const style = Object.entries(n.style).filter(([, v]) => typeof v !== 'function' && v !== '' && v != null).sort();
    if (style.length) out.style = Object.fromEntries(style);
    if (n._html != null) out.html = { length: n._html.length, sha256: hash(n._html) };
    else if (n._text) out.text = n._text.length > 200 ? { length: n._text.length, sha256: hash(n._text) } : n._text;
    if (n.children.length) out.children = n.children.map(serialize);
    return out;
  }

  return {
    document: doc, window: win, listeners,
    serialize,
    restore() {
      for (const [k, d] of saved) {
        if (d) Object.defineProperty(globalThis, k, d); else delete (globalThis as any)[k];
      }
    },
  };
}
