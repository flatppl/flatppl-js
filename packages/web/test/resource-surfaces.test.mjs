import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';
import vm from 'node:vm';

function surfaces() {
  function element(tag) {
    return {
      tag, children: [], attributes: {},
      set innerHTML(html) {
        this.children = [];
        if (html.includes('surface-html')) this.appendChild(element('div'));
      },
      appendChild(child) { child.parent = this; this.children.push(child); },
      remove() { this.parent.children = this.parent.children.filter(child => child !== this); },
      querySelector() { return this.children[0]; },
      setAttribute(key, value) { this.attributes[key] = value; },
    };
  }
  let updates = 0, disposals = 0;
  const window = {
    FlatPPLFileTypes: { typeForPath: () => 'flatppl' },
    FlatPPLViewer: { mount: () => ({
      update() { updates++; }, dispose() { disposals++; },
    }) },
  };
  vm.runInNewContext(stripTypeScriptTypes(readFileSync(new URL('../src/surfaces.ts', import.meta.url), 'utf8')),
    { window, document: { createElement: element } });
  return { api: window.FlatPPLWebSurfaces, container: element('div'),
    updates: () => updates, disposals: () => disposals };
}

test('HTML resources require a click while the sandbox remains empty', () => {
  const s = surfaces();
  const html = '<!doctype html><style>p{color:red}</style><img src="https://remote.invalid/image.png">';
  s.api.resolveSurface('html')(s.container).update({ source: html, fileName: 'report.html' });
  const host = s.container.children[0];
  const [button, blocked] = host.children;
  assert.equal(blocked.attributes.sandbox, '');
  assert.ok(blocked.srcdoc.endsWith(html));
  const policy = frame => frame.srcdoc.match(/Content-Security-Policy" content="([^"]+)"/)[1];
  assert.match(policy(blocked), /default-src 'none'/);
  assert.doesNotMatch(policy(blocked), /https?:/);
  assert.match(policy(blocked), /style-src 'unsafe-inline'/);
  button.onclick();
  assert.equal(host.children.length, 1);
  const allowed = host.children[0];
  assert.equal(allowed.attributes.sandbox, '');
  assert.equal(allowed.srcdoc, html);
});

test('a source preview disposes the old viewer and runs only from its button', () => {
  const s = surfaces();
  const pane = s.api.createPaneController(s.container, {}, () => 'model.flatppl');
  pane.update('x=1');
  let runs = 0;
  pane.update('x=ones(1000000000)', null, { preview: true, run: () => { runs++; } });
  assert.equal(s.updates(), 1);
  assert.equal(s.disposals(), 1);
  assert.equal(runs, 0);
  s.container.children[1].onclick();
  assert.equal(runs, 1);
});
