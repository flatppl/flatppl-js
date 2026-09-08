// @ts-nocheck — Node ports stand in for browser window event targets.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MessageChannel } from 'node:worker_threads';
import { defaultVscodeHost, hostMessageListener } from './host-adapter.ts';

test('standalone host messages require the same window and origin', (t) => {
  const { port1: self, port2: foreign } = new MessageChannel();
  self.location = { origin: 'https://viewer.example' };
  const previousWindow = globalThis.window;
  globalThis.window = self;
  t.after(() => { globalThis.window = previousWindow; self.close(); foreign.close(); });
  const seen = [];
  const listener = hostMessageListener(event => seen.push(event.data));
  self.addEventListener('message', listener);
  for (const [source, origin, label] of [
    [foreign, 'https://foreign.example', 'foreign window'],
    [foreign, self.location.origin, 'same-origin foreign window'],
    [null, self.location.origin, 'no source'],
    [self, 'https://foreign.example', 'wrong origin'],
    [self, 'null', 'opaque foreign origin'],
    [self, self.location.origin, 'self'],
  ]) {
    self.dispatchEvent(new MessageEvent('message', { source, origin, data: label }));
  }
  assert.deepEqual(seen, ['self']);
  self.removeEventListener('message', listener);
  self.dispatchEvent(new MessageEvent('message', {
    source: self, origin: self.location.origin, data: 'after disposal',
  }));
  assert.deepEqual(seen, ['self']);
});

test('the VS Code channel preserves host messages without acquiring its API twice', (t) => {
  const { port1: self, port2: other } = new MessageChannel();
  const previousWindow = globalThis.window;
  const previousFactory = globalThis.acquireVsCodeApi;
  globalThis.window = self;
  globalThis.acquireVsCodeApi = () => {
    throw new Error('the host already acquired the one-shot API');
  };
  t.after(() => {
    globalThis.window = previousWindow;
    globalThis.acquireVsCodeApi = previousFactory;
    self.close(); other.close();
  });
  const seen = [];
  const listener = hostMessageListener(event => seen.push(event.data));
  // Some hosts hide the factory after setup. The chosen channel stays fixed.
  globalThis.acquireVsCodeApi = undefined;
  self.addEventListener('message', listener);
  self.dispatchEvent(new MessageEvent('message', { data: 'source update' }));
  assert.deepEqual(seen, ['source update']);
});

test('VS Code messages survive a factory hidden during host creation', (t) => {
  const { port1: self, port2: other } = new MessageChannel();
  const previousWindow = globalThis.window;
  const previousFactory = globalThis.acquireVsCodeApi;
  globalThis.window = self;
  globalThis.acquireVsCodeApi = () => {
    globalThis.acquireVsCodeApi = undefined;
    return { postMessage() {} };
  };
  t.after(() => {
    globalThis.window = previousWindow;
    globalThis.acquireVsCodeApi = previousFactory;
    self.close(); other.close();
  });
  // mount creates the host before installing its window listener.
  defaultVscodeHost();
  const seen = [];
  self.addEventListener('message', hostMessageListener(event => seen.push(event.data)));
  self.dispatchEvent(new MessageEvent('message', { data: 'source update' }));
  assert.deepEqual(seen, ['source update']);
});
