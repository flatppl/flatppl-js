# FlatPPL viewer

A web/JS-based viewer for FlatPPL, the Flat Portable Probabilistic Language.

The viewer visualizes and allows the user to explore the directed acyclic
graph (DAG) of FlatPPL modules. The viewer can also show plots of the
deterministic and stochastic values of graph nodes.

> Note: The FlatPPL viewer is in early development and may be
> unstable. It is not yet published to the npm registry; it currently
> ships only as part of the [`flatppl-js`](https://github.com/flatppl/flatppl-js)
> monorepo and is consumed via npm workspace symlinks by sibling packages.

## About FlatPPL

FlatPPL is a minimal, inference-agnostic stochastic language for specifying
probabilistic models.

## Testing

```sh
# from the flatppl-js workspace root, first time only (or after a clean):
npm install

# then in this package:
cd packages/viewer
npm run build  # populates ./vendor/ with all assets
npm run serve  # http://localhost:8000/  (defaults to embed-test.html)
```

`npm run watch` keeps esbuild in watch mode for the engine + sampler-
worker bundles, so edits to `packages/engine/` re-bundle automatically.

## Embedding in your own page

Mirror what `embed-test.html` does:

1. Drop a container element where the viewer should render:

   ```html
   <div id="flatppl-viewer-root" style="width: 100vw; height: 100vh;"></div>
   ```

2. Load cytoscape, dagre extensions, echarts, the engine bundle, and
   the viewer (in that order). The `vendor/` directory after
   `npm run build` contains everything you need:

   ```html
   <script src="vendor/cytoscape.min.js"></script>
   <script src="vendor/dagre.min.js"></script>
   <script src="vendor/cytoscape-dagre.js"></script>
   <script src="vendor/cytoscape-layers.min.js"></script>
   <script src="vendor/cytoscape-bubblesets.min.js"></script>
   <script src="vendor/echarts.min.js"></script>
   <script src="vendor/engine.min.js"></script>
   ```

3. Tell the viewer where the sampler-worker bundle lives, then load
   the viewer itself:

   ```html
   <script>
     window.__FLATPPL_CONFIG__ = {
       samplerWorkerUrl: new URL('vendor/sampler-worker.min.js', document.baseURI).href,
     };
   </script>
   <script src="vendor/viewer.js"></script>
   ```

4. The viewer auto-mounts inside `#flatppl-viewer-root` on
   `DOMContentLoaded`. Feed it source by `postMessage`:

   ```html
   <script>
     document.addEventListener('DOMContentLoaded', function() {
       window.postMessage({
         type: 'sourceUpdate',
         source: '<your FlatPPL source>',
         targetName: '<binding to focus>',
       }, '*');
     });
   </script>
   ```

   Or call `FlatPPLViewer.mount(container, opts)` directly (with
   `opts.source`, `opts.target`, and optionally `opts.host`) and
   ignore the auto-mount marker.

## Host adapter

The viewer delegates IDE-only concerns to an optional host adapter
(`opts.host`):

```js
{
  revealSourceLine?(line)  // jump editor to source line
  setTitle?(name)          // host updates surrounding panel title
  saveState?(state)        // persist viewer state across reloads
  loadState?()             // load previously-persisted state
}
```

All methods are optional; missing methods are no-ops at the call site.
A standalone embed can pass `{}` (or omit `host` entirely) and the
viewer renders fine — just without the cross-pane navigation niceties.

When no `host` is supplied AND the page is inside a VS Code webview
(`acquireVsCodeApi` is defined), a default adapter bridges to VS Code's
`postMessage` / `setState` / `getState`, which is how the existing
extension wrapper continues to work without per-call branching.

## Public API

```js
var view = FlatPPLViewer.mount(container, {
  source: '...flatppl source...',
  target: 'someBinding',           // optional initial focus
  host: { revealSourceLine, ... }, // optional; see above
});

view.update(newSource, newTarget);  // re-parse and re-render
view.dispose();                      // teardown placeholder (currently a no-op)
```

## Status

Functional but young — the package is private (not published to npm)
and ships as workspace-internal source. The VS Code extension consumes
`viewer.js` directly via its own build step; an online publishing
story (CDN bundle, npm-published assets) is left for whenever it's
needed.
