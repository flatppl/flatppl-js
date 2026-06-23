// @flatppl/web — right-pane surface registry (Phase 1).
//
// The gallery's right pane is no longer "the FlatPPL viewer" — it is a
// file-type-dependent SURFACE chosen from a registry. This is the
// mechanism that keeps the gallery framework-neutral: FlatPPL is ONE
// registered surface (wrapping the shared viewer bundle), Markdown is
// another (a live preview), and any unrecognised type falls back to a
// neutral placeholder — the reserved slot for future native pyhf / HS3 /
// Stan / … surfaces. No type is privileged or hard-wired; the FlatPPL
// surface happens to be the rich one only because FlatPPL ships a JS
// engine, which is a capability difference, not a structural one.
//
// Contract (mirrors the viewer's own mount→{update,dispose} lifecycle):
//
//   SurfaceFactory(container, ctx) -> SurfaceHandle
//   SurfaceHandle { update(input), dispose() }
//   SurfaceInput  { source, target?, fileName?, opts? }
//   SurfaceCtx    { host?, config? }
//
// The PaneController owns the single right-pane container and dispatches
// by file type: a SAME-type switch just calls update() (cheap — no
// re-mount); a CROSS-type switch disposes the active surface and mounts
// the new one. The dispose() is load-bearing — without it a FlatPPL→
// Markdown switch would orphan a sampler worker + cytoscape/echarts +
// the window 'message' listener (see the viewer's mount() teardown).
//
// Loaded as a plain <script> installing window.FlatPPLWebSurfaces, like
// every other gallery sub-module (no ES module graph — see globals.d.ts).

(function (globalScope: any) {
  'use strict';

  // ---- file-type detection -------------------------------------------
  //
  // Derived from the path extension. The classification itself lives in the
  // shared file-types module (window.FlatPPLFileTypes, bundled to
  // vendor/file-types.js and loaded before this script) so build.mjs and
  // the runtime can't drift. This wrapper stays the gallery's public API
  // (window.FlatPPLWebSurfaces.typeForPath) and degrades gracefully to
  // 'flatppl' if the global is somehow absent.
  function typeForPath(path: any): string {
    const ft = (globalScope as any).FlatPPLFileTypes;
    return ft && typeof ft.typeForPath === 'function' ? ft.typeForPath(path) : 'flatppl';
  }

  // ---- surfaces ------------------------------------------------------

  // FlatPPL: the existing viewer bundle is one surface. mount() injects
  // its own markup into `container`; we adapt its (source, target, opts)
  // update signature to the registry's input object and delegate dispose
  // to the viewer's real teardown.
  function flatpplSurface(container: any, ctx: any): any {
    const V = globalScope.FlatPPLViewer;
    if (!V || typeof V.mount !== 'function') {
      container.innerHTML =
        '<div class="surface-placeholder"><p>FlatPPL viewer unavailable.</p></div>';
      return { update: function () {}, dispose: function () {} };
    }
    const handle = V.mount(container, { host: ctx && ctx.host });
    return {
      update: function (input: any) {
        handle.update(input.source, input.target || null, input.opts || {});
      },
      dispose: function () {
        try { if (handle && handle.dispose) handle.dispose(); } catch (_) {}
      },
    };
  }

  // Markdown: a live preview over the viewer's renderDoc pipeline
  // (marked + Temml MathML) — the SAME renderer that drives the DAG
  // tooltips, so styling stays consistent and the gallery bundles no
  // extra markdown library. Content is trusted gallery/demo text;
  // sanitising arbitrary user markdown is a later hardening step.
  function markdownSurface(container: any, _ctx: any): any {
    container.innerHTML = '<div class="surface-markdown markdown-body" tabindex="0"></div>';
    const host = container.querySelector('.surface-markdown');
    const renderDoc = globalScope.FlatPPLViewer && globalScope.FlatPPLViewer.renderDoc;
    return {
      update: function (input: any) {
        const md = input && input.source != null ? String(input.source) : '';
        if (!host) return;
        // renderDoc takes a DocComment ({ markup, lines }), NOT a raw
        // string — it is the FlatPPL doc-comment renderer (marked + Temml
        // MathML). Wrap the file body as a `md` doc-comment so the whole
        // document (headings / tables / code / $math$) rides the same
        // pipeline. It returns null for empty content → plain-text fallback.
        if (typeof renderDoc === 'function') {
          try {
            const html = renderDoc({ markup: 'md', lines: md.split(/\r?\n/) });
            if (html != null) { host.innerHTML = html; return; }
          } catch (_) { /* fall through to plain text */ }
        }
        host.textContent = md;  // no renderer / empty / error → readable fallback
      },
      dispose: function () { try { container.innerHTML = ''; } catch (_) {} },
    };
  }

  // Placeholder: neutral "nothing to visualise yet" pane for any type
  // without a registered surface (pyhf / HS3 / Stan / Julia / …). The
  // reserved slot a future native surface drops into.
  function placeholderSurface(container: any, _ctx: any): any {
    return {
      update: function (_input: any) {
        container.innerHTML =
          '<div class="surface-placeholder"><p>No visualization available.</p></div>';
      },
      dispose: function () { try { container.innerHTML = ''; } catch (_) {} },
    };
  }

  // ---- registry ------------------------------------------------------
  //
  // type → factory. The SOLE right-pane dispatch — no FlatPPL fast-path
  // anywhere else. New surfaces register here (later: lazily, so a
  // deployment only ships what it registers).
  const REGISTRY: Record<string, any> = {
    flatppl: flatpplSurface,
    markdown: markdownSurface,
  };

  function resolveSurface(type: any): any {
    return REGISTRY[type] || placeholderSurface;
  }

  // ---- pane controller -----------------------------------------------
  //
  // Owns the single right-pane container + the active surface. update()
  // keeps the existing `viewer.update(source, target, opts)` signature so
  // every call site is unchanged; it derives the active file from
  // getActivePath() (the router is the source of truth for "what's
  // shown"), picks the surface, disposes-and-remounts on a type change,
  // and forwards the update.
  function createPaneController(container: any, ctx: any, getActivePath: any): any {
    let currentType: string | null = null;
    let handle: any = null;

    function ensure(type: string) {
      if (type === currentType && handle) return;
      if (handle) { try { handle.dispose(); } catch (_) {} handle = null; }
      handle = resolveSurface(type)(container, ctx || {});
      currentType = type;
    }

    return {
      update: function (source: any, target: any, opts: any) {
        const path = (typeof getActivePath === 'function') ? getActivePath() : null;
        ensure(typeForPath(path));
        if (handle) {
          handle.update({ source: source, target: target || null, fileName: path, opts: opts });
        }
      },
      dispose: function () {
        if (handle) { try { handle.dispose(); } catch (_) {} handle = null; }
        currentType = null;
      },
      // Exposed for tests / diagnostics — the active surface type.
      currentType: function () { return currentType; },
    };
  }

  globalScope.FlatPPLWebSurfaces = {
    typeForPath: typeForPath,
    resolveSurface: resolveSurface,
    createPaneController: createPaneController,
  };
})(typeof window !== 'undefined' ? window : globalThis);
