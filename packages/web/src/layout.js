// @flatppl/web — three-pane layout manager.
//
// Owns the column widths of the gallery's CSS Grid (#app), the
// file-pane collapse state, and the drag-to-resize handles between
// adjacent panes. All state lives in localStorage so a user's
// preferred layout survives reloads.
//
// Grid columns track three logical widths:
//   - filesWidth  : px width of the file tree
//   - sourceFrac  : fr-units of the source pane
//   - viewerFrac  : fr-units of the viewer pane
// Plus two ~5px drag handles. When the file pane is collapsed
// (toggle button), filesWidth is overridden to 0 and the handle
// before the source pane is hidden; the user's chosen filesWidth
// is preserved in storage so toggling re-expand restores it.
//
// The viewer's own ResizeObserver picks up container size changes
// from this module's grid mutations, so cytoscape re-fits the DAG
// without an explicit re-render call from here.
//
// Lives on globalThis as window.FlatPPLWebLayout.

'use strict';

(function (globalScope) {
  var STORAGE_KEY = 'flatppl-web-layout-v1';

  // Sane defaults — match the original step-1.2 grid.
  var DEFAULTS = Object.freeze({
    filesWidth: 240,
    sourceFrac: 1.0,
    viewerFrac: 1.5,
    collapsed: false,
  });

  // Width clamps so the user can't drag a pane to disappear.
  var MIN_FILES_WIDTH  = 120;
  var MAX_FILES_WIDTH  = 600;
  var MIN_PANE_PX      = 200;   // applied to source / viewer post-handle

  var HANDLE_PX = 5;

  function loadState() {
    try {
      var raw = globalScope.localStorage && globalScope.localStorage.getItem(STORAGE_KEY);
      if (!raw) return Object.assign({}, DEFAULTS);
      var parsed = JSON.parse(raw);
      return {
        filesWidth: clamp(parsed.filesWidth, MIN_FILES_WIDTH, MAX_FILES_WIDTH, DEFAULTS.filesWidth),
        sourceFrac: clamp(parsed.sourceFrac, 0.2, 10,            DEFAULTS.sourceFrac),
        viewerFrac: clamp(parsed.viewerFrac, 0.2, 10,            DEFAULTS.viewerFrac),
        collapsed:  parsed.collapsed === true,
      };
    } catch (_) {
      return Object.assign({}, DEFAULTS);
    }
  }

  function saveState(state) {
    try {
      if (globalScope.localStorage) {
        globalScope.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      }
    } catch (_) { /* quota / disabled — ignore */ }
  }

  function clamp(n, lo, hi, fallback) {
    if (typeof n !== 'number' || !isFinite(n)) return fallback;
    if (n < lo) return lo;
    if (n > hi) return hi;
    return n;
  }

  /** Install the layout manager. Mutates the existing #app grid to
      include drag-handle children, registers click + drag listeners,
      and applies the persisted state. Returns an object with
      { toggleFiles, isCollapsed }. */
  function install(opts) {
    opts = opts || {};
    var app          = document.getElementById('app');
    var filesPane    = document.getElementById('files-pane');
    var sourcePane   = document.getElementById('source-pane');
    var viewerPane   = document.getElementById('viewer-pane');
    var toggleButton = opts.toggleButton || null;
    if (!app || !filesPane || !sourcePane || !viewerPane) {
      console.warn('[@flatppl/web] layout.install: required elements missing');
      return null;
    }

    var state = loadState();

    // Insert two drag handles into the grid: one between files and
    // source, one between source and viewer. Each is a thin element
    // the user can grab to redistribute width.
    var handleFS = createHandle('files-source');
    var handleSV = createHandle('source-viewer');
    sourcePane.parentNode.insertBefore(handleFS, sourcePane);
    viewerPane.parentNode.insertBefore(handleSV, viewerPane);

    function createHandle(id) {
      var el = document.createElement('div');
      el.className = 'resize-handle';
      el.dataset.resize = id;
      el.title = 'Drag to resize';
      return el;
    }

    function applyGrid() {
      var filesPart = state.collapsed ? '0px' : (state.filesWidth + 'px');
      var handleFSPart = state.collapsed ? '0px' : (HANDLE_PX + 'px');
      app.style.gridTemplateColumns =
        filesPart + ' ' +
        handleFSPart + ' ' +
        state.sourceFrac + 'fr ' +
        HANDLE_PX + 'px ' +
        state.viewerFrac + 'fr';
      filesPane.style.display = state.collapsed ? 'none' : '';
      handleFS.style.display  = state.collapsed ? 'none' : '';
      if (toggleButton) {
        toggleButton.setAttribute('aria-expanded', state.collapsed ? 'false' : 'true');
        toggleButton.title = state.collapsed ? 'Show file list' : 'Hide file list';
      }
    }

    applyGrid();

    // Drag handlers.
    [handleFS, handleSV].forEach(function (handle) {
      handle.addEventListener('mousedown', function (ev) {
        ev.preventDefault();
        startDrag(handle.dataset.resize, ev.clientX);
      });
    });

    function startDrag(which, startX) {
      // Capture starting metrics so the drag is purely additive.
      var rect = app.getBoundingClientRect();
      var totalW = rect.width;
      // Width consumed by file pane + first handle (px-sized).
      var fixedPx = (state.collapsed ? 0 : state.filesWidth + HANDLE_PX);
      // Width consumed by the second handle.
      var trailingPx = HANDLE_PX;
      // The source + viewer share (totalW - fixedPx - trailingPx).
      // We map that to fr-units 1.0 baseline so total fr =
      // sourceFrac + viewerFrac.
      var totalFr = state.sourceFrac + state.viewerFrac;
      var startFilesW = state.filesWidth;
      var startSourceFrac = state.sourceFrac;
      var startViewerFrac = state.viewerFrac;

      function onMove(mv) {
        var dx = mv.clientX - startX;
        if (which === 'files-source' && !state.collapsed) {
          // Adjust the file-pane width directly.
          state.filesWidth = clamp(startFilesW + dx, MIN_FILES_WIDTH, MAX_FILES_WIDTH, startFilesW);
        } else if (which === 'source-viewer') {
          // Convert dx (px) to fr (relative to source+viewer area).
          var area = totalW - fixedPx - trailingPx;
          if (area <= 0) return;
          var frPerPx = totalFr / area;
          var newSourceFrac = startSourceFrac + dx * frPerPx;
          var newViewerFrac = startViewerFrac - dx * frPerPx;
          // Apply min-width clamp by converting back: each pane must
          // be ≥ MIN_PANE_PX of pixel width.
          var sourcePx = newSourceFrac / totalFr * area;
          var viewerPx = newViewerFrac / totalFr * area;
          if (sourcePx < MIN_PANE_PX) {
            newSourceFrac = MIN_PANE_PX / area * totalFr;
            newViewerFrac = totalFr - newSourceFrac;
          } else if (viewerPx < MIN_PANE_PX) {
            newViewerFrac = MIN_PANE_PX / area * totalFr;
            newSourceFrac = totalFr - newViewerFrac;
          }
          state.sourceFrac = newSourceFrac;
          state.viewerFrac = newViewerFrac;
        }
        applyGrid();
      }
      function onUp() {
        document.removeEventListener('mousemove', onMove);
        document.removeEventListener('mouseup', onUp);
        document.body.style.cursor = '';
        document.body.style.userSelect = '';
        saveState(state);
      }
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
      document.addEventListener('mousemove', onMove);
      document.addEventListener('mouseup', onUp);
    }

    function toggleFiles() {
      state.collapsed = !state.collapsed;
      applyGrid();
      saveState(state);
    }

    if (toggleButton) {
      toggleButton.addEventListener('click', toggleFiles);
    }

    return {
      toggleFiles: toggleFiles,
      isCollapsed: function () { return state.collapsed; },
    };
  }

  globalScope.FlatPPLWebLayout = { install: install };
})(typeof window !== 'undefined' ? window : globalThis);
