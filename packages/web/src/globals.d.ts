// Ambient declarations for the @flatppl/web gallery shell.
//
// The src/*.ts files in this package are loaded into the page as plain
// <script src="..."> tags (NOT type="module") and share state via
// window.FlatPPL* globals. Each file is its own IIFE; there's no ES
// module graph. This means tsc treats them as scripts and the shared
// runtime contract — engine, viewer, web sub-modules, vendor libs —
// has to be declared globally here.
//
// Mirror of packages/viewer/src/types.d.ts in spirit; intentionally
// kept permissive (`any` everywhere) for the initial migration so we
// don't repeat the engine surface here.

// Runtime globals injected by <script> tags BEFORE the gallery
// scripts load — see src/index.html.
declare const FlatPPLEngine: any;
declare const echarts: any;
declare const cytoscape: any;
declare const dagre: any;

interface Window {
  // Engine & viewer bundles (injected by vendor/<bundle>.js).
  FlatPPLEngine?: any;
  FlatPPLViewer?: any;
  FlatPPLEditorBundle?: any;

  // Host config consumed by the gallery + viewer on boot.
  __FLATPPL_CONFIG__?: {
    samplerWorkerUrl?: string;
    allowEdit?: boolean;
    playground?: boolean;
    [extra: string]: any;
  };

  // Web sub-modules — each src/*.ts file installs its own surface here.
  FlatPPLWebRouter?: any;
  FlatPPLWebManifest?: any;
  FlatPPLWebResolver?: any;
  FlatPPLWebLayout?: any;
  FlatPPLWebEditor?: any;
  FlatPPLWebEphemeral?: any;
  FlatPPLWeb?: any;
}
