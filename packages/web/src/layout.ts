// @flatppl/web — responsive workspace layout.
//
// Wide screens retain the user's saved pane widths. Narrow screens use one
// main pane and a modal Files drawer, without overwriting those desktop widths.
// The viewer observes its container, so changing panes needs no data reload.

'use strict';

(function (globalScope: any) {
  const STORAGE_KEY = 'flatppl-web-layout-v1';
  const MIN_FILES_WIDTH = 120;
  const MAX_FILES_WIDTH = 600;
  const MIN_PANE_PX = 200;
  const HANDLE_PX = 12;

  interface LayoutState {
    filesWidth: number;
    sourceFrac: number;
    viewerFrac: number;
    collapsed: boolean;
  }
  type MainPane = 'source' | 'viewer';
  type Divider = 'files-source' | 'source-viewer';
  const DEFAULTS: LayoutState = {
    filesWidth: 240, sourceFrac: 1, viewerFrac: 1.5, collapsed: false,
  };

  /** Ignore malformed or unavailable preferences without preventing startup. */
  function loadState(): LayoutState {
    try {
      const parsed = JSON.parse(globalScope.localStorage.getItem(STORAGE_KEY));
      return {
        filesWidth: clamp(parsed.filesWidth, MIN_FILES_WIDTH, MAX_FILES_WIDTH, DEFAULTS.filesWidth),
        sourceFrac: clamp(parsed.sourceFrac, 0.01, 10, DEFAULTS.sourceFrac),
        viewerFrac: clamp(parsed.viewerFrac, 0.01, 10, DEFAULTS.viewerFrac),
        collapsed: parsed.collapsed === true,
      };
    } catch (_) {
      return { ...DEFAULTS };
    }
  }

  function saveState(state: LayoutState) {
    try {
      globalScope.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    } catch (_) { /* Preferences are optional when browser storage is unavailable. */ }
  }

  function clamp(n: number, lo: number, hi: number, fallback = lo) {
    return typeof n === 'number' && Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : fallback;
  }

  /** Own the workspace geometry and focus lifecycle, independent of file routing. */
  function install(opts?: { toggleButton?: HTMLElement | null }) {
    const app = document.getElementById('app');
    const filesPane = document.getElementById('files-pane');
    const sourcePane = document.getElementById('source-pane');
    const viewerPane = document.getElementById('viewer-pane');
    if (!app || !filesPane || !sourcePane || !viewerPane) {
      console.warn('[@flatppl/web] layout.install: required elements missing');
      return null;
    }

    const toggleButton = opts?.toggleButton;
    const closeButton = document.getElementById('close-files-btn');
    const backdrop = document.getElementById('files-backdrop');
    const sourceButton = document.getElementById('show-source-btn');
    const viewerButton = document.getElementById('show-viewer-btn');
    const splitButton = document.getElementById('show-split-btn');
    const media = globalScope.matchMedia('(max-width: 900px)') as MediaQueryList;
    const state = loadState();
    let compact = media.matches;
    let mainPane: MainPane = 'source';
    let focusedPane: MainPane | null = null;
    let drawerOpen = false;
    let drawerOpener: HTMLElement | null = null;
    let finishResize: (() => void) | null = null;

    const handleFS = createHandle('files-source', 'Resize Files', 'files-pane source-pane');
    const handleSV = createHandle('source-viewer', 'Resize Source and Visualization', 'source-pane viewer-pane');
    app.insertBefore(handleFS, sourcePane);
    app.insertBefore(handleSV, viewerPane);

    /** Expose the same constrained resize operation to pointer and keyboard users. */
    function createHandle(which: Divider, label: string, controls: string) {
      const el = document.createElement('div');
      el.className = 'resize-handle';
      el.dataset.resize = which;
      el.tabIndex = 0;
      el.setAttribute('role', 'separator');
      el.setAttribute('aria-label', label);
      el.setAttribute('aria-orientation', 'vertical');
      el.setAttribute('aria-controls', controls);
      el.title = 'Drag or use Left and Right arrow keys to resize';
      el.addEventListener('pointerdown', (ev) => startResize(el, which, ev));
      el.addEventListener('keydown', (ev) => {
        if (compact || (which === 'source-viewer' && focusedPane)) return;
        const { filesWidth, area, maxFiles } = measure();
        const current = which === 'files-source' ? filesWidth : sourceWidth(area);
        const min = which === 'files-source' ? MIN_FILES_WIDTH : MIN_PANE_PX;
        const max = which === 'files-source' ? maxFiles : area - MIN_PANE_PX;
        const step = ev.shiftKey ? 50 : 20;
        let next: number;
        if (ev.key === 'ArrowLeft') next = current - step;
        else if (ev.key === 'ArrowRight') next = current + step;
        else if (ev.key === 'Home') next = min;
        else if (ev.key === 'End') next = max;
        else return;
        ev.preventDefault();
        resize(which, next);
        saveState(state);
      });
      return el;
    }

    /** Clamp display widths against this viewport, while retaining saved preferences. */
    function measure() {
      const style = globalScope.getComputedStyle(app);
      const width = app!.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight);
      const mainMinimum = focusedPane ? MIN_PANE_PX : MIN_PANE_PX * 2 + HANDLE_PX;
      const maxFiles = Math.max(MIN_FILES_WIDTH, Math.min(MAX_FILES_WIDTH, width - mainMinimum - HANDLE_PX));
      const filesWidth = state.collapsed ? 0 : clamp(state.filesWidth, MIN_FILES_WIDTH, maxFiles);
      const area = width - (state.collapsed ? 0 : filesWidth + HANDLE_PX) - (focusedPane ? 0 : HANDLE_PX);
      return { filesWidth, area, maxFiles };
    }

    function sourceWidth(area: number) {
      return clamp(area * state.sourceFrac / (state.sourceFrac + state.viewerFrac), MIN_PANE_PX, area - MIN_PANE_PX);
    }

    /** Keep hidden panes out of both keyboard navigation and the accessibility tree. */
    function applyGrid() {
      const { filesWidth, area, maxFiles } = measure();
      const sourcePx = sourceWidth(area);
      const sourcePart = focusedPane === 'viewer' ? '0px' : focusedPane === 'source' ? '1fr' : sourcePx + 'fr';
      const viewerPart = focusedPane === 'source' ? '0px' : focusedPane === 'viewer' ? '1fr' : (area - sourcePx) + 'fr';
      app!.style.gridTemplateColumns = compact ? 'minmax(0, 1fr)'
        : `${filesWidth}px ${state.collapsed ? 0 : HANDLE_PX}px minmax(0, ${sourcePart}) ${focusedPane ? 0 : HANDLE_PX}px minmax(0, ${viewerPart})`;
      app!.dataset.layout = compact ? 'compact' : 'wide';
      filesPane!.hidden = compact ? !drawerOpen : state.collapsed;
      sourcePane!.hidden = compact ? mainPane !== 'source' : focusedPane === 'viewer';
      viewerPane!.hidden = compact ? mainPane !== 'viewer' : focusedPane === 'source';
      handleFS.hidden = compact || state.collapsed;
      handleSV.hidden = compact || focusedPane !== null;
      if (backdrop) backdrop.hidden = !compact || !drawerOpen;
      if (compact && drawerOpen) {
        filesPane!.setAttribute('role', 'dialog');
        filesPane!.setAttribute('aria-modal', 'true');
      } else {
        filesPane!.removeAttribute('role');
        filesPane!.removeAttribute('aria-modal');
      }
      for (const child of Array.from(app!.children)) {
        if (child !== filesPane && child !== backdrop) (child as HTMLElement).inert = compact && drawerOpen;
      }
      const filesVisible = compact ? drawerOpen : !state.collapsed;
      toggleButton?.setAttribute('aria-expanded', String(filesVisible));
      if (toggleButton) toggleButton.title = filesVisible ? 'Hide file list' : 'Show file list';
      sourceButton?.setAttribute('aria-pressed', String(compact ? mainPane === 'source' : focusedPane === 'source'));
      viewerButton?.setAttribute('aria-pressed', String(compact ? mainPane === 'viewer' : focusedPane === 'viewer'));
      splitButton?.setAttribute('aria-pressed', String(!focusedPane));
      if (sourceButton) sourceButton.title = compact ? 'Show Source' : 'Focus Source';
      if (viewerButton) viewerButton.title = compact ? 'Show Visualization' : 'Focus Visualization';
      if (!handleFS.hidden) setRange(handleFS, MIN_FILES_WIDTH, maxFiles, filesWidth, 'Files');
      if (!handleSV.hidden) setRange(handleSV, MIN_PANE_PX, area - MIN_PANE_PX, sourcePx, 'Source');
    }

    function setRange(handle: HTMLElement, min: number, max: number, value: number, name: string) {
      handle.setAttribute('aria-valuemin', String(Math.round(min)));
      handle.setAttribute('aria-valuemax', String(Math.round(max)));
      handle.setAttribute('aria-valuenow', String(Math.round(value)));
      handle.setAttribute('aria-valuetext', `${name} width ${Math.round(value)} pixels`);
    }

    /** Convert pixel input to the existing saved ratio, shared by both input methods. */
    function resize(which: Divider, width: number) {
      const { area, maxFiles } = measure();
      if (which === 'files-source') {
        state.filesWidth = clamp(width, MIN_FILES_WIDTH, maxFiles);
      } else {
        const fraction = clamp(width, MIN_PANE_PX, area - MIN_PANE_PX) / area;
        state.sourceFrac = fraction * 2.5;
        state.viewerFrac = (1 - fraction) * 2.5;
      }
      applyGrid();
    }

    /** Pointer capture also handles touch, pen, and a drag leaving its divider. */
    function startResize(handle: HTMLElement, which: Divider, ev: PointerEvent) {
      if (ev.button !== 0 || compact) return;
      ev.preventDefault();
      finishResize?.();
      handle.focus();
      const { filesWidth, area } = measure();
      const startWidth = which === 'files-source' ? filesWidth : sourceWidth(area);
      const onMove = (move: PointerEvent) => {
        if (move.pointerId === ev.pointerId) resize(which, startWidth + move.clientX - ev.clientX);
      };
      const onEnd = (end: PointerEvent) => {
        if (end.pointerId === ev.pointerId) finishResize?.();
      };
      const oldCursor = app!.style.cursor;
      const oldSelect = app!.style.userSelect;
      finishResize = () => {
        document.removeEventListener('pointermove', onMove);
        document.removeEventListener('pointerup', onEnd);
        document.removeEventListener('pointercancel', onEnd);
        handle.removeEventListener('lostpointercapture', onEnd);
        if (handle.hasPointerCapture(ev.pointerId)) handle.releasePointerCapture(ev.pointerId);
        app!.style.cursor = oldCursor;
        app!.style.userSelect = oldSelect;
        finishResize = null;
        saveState(state);
      };
      handle.setPointerCapture(ev.pointerId);
      app!.style.cursor = 'col-resize';
      app!.style.userSelect = 'none';
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onEnd);
      document.addEventListener('pointercancel', onEnd);
      handle.addEventListener('lostpointercapture', onEnd);
    }

    /** Opening the finder must work even when the desktop sidebar was collapsed. */
    function revealFiles() {
      if (compact) {
        if (!drawerOpen) drawerOpener = document.activeElement as HTMLElement | null;
        drawerOpen = true;
      } else if (state.collapsed) {
        state.collapsed = false;
        saveState(state);
      }
      applyGrid();
      document.getElementById('file-filter')?.focus();
    }

    /** File selection closes only the mobile drawer, preserving the desktop sidebar. */
    function closeFiles() {
      if (!drawerOpen) return;
      drawerOpen = false;
      applyGrid();
      if (drawerOpener?.isConnected && !drawerOpener.closest('[hidden]')) drawerOpener.focus();
      else toggleButton?.focus();
      drawerOpener = null;
    }

    function toggleFiles() {
      if (compact) {
        if (drawerOpen) closeFiles();
        else revealFiles();
      } else {
        state.collapsed = !state.collapsed;
        applyGrid();
        saveState(state);
      }
    }

    /** Reveal cross-pane navigation targets without changing a desktop split view. */
    function showPane(pane: MainPane) {
      const otherPane = pane === 'source' ? viewerPane : sourcePane;
      const moveFocus = (compact || focusedPane !== null) && otherPane!.contains(document.activeElement);
      mainPane = pane;
      if (focusedPane) focusedPane = pane;
      closeFiles();
      applyGrid();
      if (moveFocus) (pane === 'source' ? sourceButton : viewerButton)?.focus();
    }

    function focusPane(pane: MainPane) {
      mainPane = pane;
      if (!compact) focusedPane = pane;
      applyGrid();
    }

    toggleButton?.addEventListener('click', toggleFiles);
    closeButton?.addEventListener('click', closeFiles);
    backdrop?.addEventListener('click', closeFiles);
    sourceButton?.addEventListener('click', () => focusPane('source'));
    viewerButton?.addEventListener('click', () => focusPane('viewer'));
    splitButton?.addEventListener('click', () => { focusedPane = null; applyGrid(); });

    // The modal drawer keeps focus within Files. Inert siblings also block
    // pointer activation and screen-reader navigation through the backdrop.
    document.addEventListener('keydown', (ev) => {
      if (!compact || !drawerOpen) return;
      if (ev.key === 'Escape') {
        ev.preventDefault();
        closeFiles();
      } else if (ev.key === 'Tab') {
        const controls = Array.from(filesPane!.querySelectorAll<HTMLElement>(
          'button:not([disabled]), input:not([disabled]), a[href], [tabindex="0"]',
        )).filter((el) => el.getClientRects().length > 0);
        const first = controls[0];
        const last = controls[controls.length - 1];
        if (ev.shiftKey && document.activeElement === first) { ev.preventDefault(); last?.focus(); }
        else if (!ev.shiftKey && document.activeElement === last) { ev.preventDefault(); first?.focus(); }
      }
    });
    media.addEventListener('change', () => {
      finishResize?.();
      closeFiles();
      compact = media.matches;
      if (viewerPane!.contains(document.activeElement)) mainPane = 'viewer';
      else if (sourcePane!.contains(document.activeElement)) mainPane = 'source';
      applyGrid();
    });
    globalScope.addEventListener('resize', () => { finishResize?.(); applyGrid(); });
    applyGrid();

    return {
      toggleFiles, revealFiles, closeFiles, showPane,
      isCollapsed: () => compact ? !drawerOpen : state.collapsed,
    };
  }

  globalScope.FlatPPLWebLayout = { install };
})(typeof window !== 'undefined' ? window : globalThis);
