// @ts-check
// @flatppl/viewer — the math pane —
//
// Third panel of the vertical split (panels.ts): the focused model as
// mathematical notation. The FlatPPL → mathematics conversion is done
// by flatppl-rust (`flatppl_wasm_api.render_math`, loaded lazily from
// the host-supplied `__FLATPPL_CONFIG__.wasmApiUrl`); this module owns
// the pane's DOM, the request/response plumbing and the navigation
// hooks. Design + contract: flatppl-dev/math-view-design.md.
//
// This is the pane scaffold; the renderer seam and the per-binding rows
// land in the next step. Until then the pane states plainly that no
// renderer is wired, rather than showing an empty box.

import { $ } from './util.js';
import type { Ctx } from './types';

/** Render (or re-render) the math pane for the current model. */
export function renderMathForCurrent(ctx: Ctx) {
  const el = $('math-content');
  if (!el) return;
  el.innerHTML = '<div class="math-empty">Math view is not available in this build.</div>';
}
