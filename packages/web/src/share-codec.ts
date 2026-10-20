/**
 * Browser transport for complete §04 bundles. A share holds the bundle in a
 * fragment, never in a request to a storage service. The worker decodes
 * untrusted links. The archive reader then validates the rebuilt ZIP like any
 * other bundle. The format is described in ../share-links.md.
 * Load this module only when opening or making a share.
 */
import type { QRCode } from 'qrcode';
import { MAX_SHARE_URL_LENGTH, SHARE_ARCHIVE_LIMITS, type ShareMember, type ShareView } from './share-format.ts';

export { MAX_SHARE_URL_LENGTH, SHARE_ARCHIVE_LIMITS };
export const MAX_SHARE_QR_VERSION = 40;

const WORKER_TIMEOUT_MS = 60_000;

export interface ShareTokens {
  link: string;
  /** Null when the payload exceeds the largest QR code. */
  qr: string | null;
}

export interface ShareLinks {
  url: string;
  /** The same share with a decimal token, or null when no QR code can hold it. */
  qrUrl: string | null;
  tooLong: boolean;
}

/** A shared bundle as a STORE ZIP, and the member and binding it opens. */
export interface OpenedShare {
  archive: Uint8Array;
  model: string | null;
  target: string | null;
  /** The link token for the same payload, also when a QR token was opened. */
  link: string;
}

export type ShareWorkerTask = { operation: 'encode'; view: ShareView; members: ShareMember[] }
  | { operation: 'decode'; token: string };
export type ShareWorkerRequest = ShareWorkerTask & { id: number };
export type ShareWorkerReply = {
  id: number; tokens?: ShareTokens | null; link?: string; view?: ShareView; members?: ShareMember[]; error?: string;
};

interface ArchiveEngine {
  readArchive(bytes: Uint8Array, options: { id: string }): Promise<any>;
  writeArchive(files: Record<string, Uint8Array>, options: object): Promise<Uint8Array>;
}

/** The page's engine, bundle store and resolver, which open shares. */
interface ShareHost {
  FlatPPLEngine: ArchiveEngine;
  FlatPPLWebBundles: any;
  FlatPPLWebResolver: any;
}

/** Members relative to the bundle root, as the archive reader sees them. */
async function archiveMembers(zip: Uint8Array, engine: ArchiveEngine): Promise<ShareMember[]> {
  const archive = await engine.readArchive(zip, { id: 'share.flatppl.zip' });
  try {
    const members: ShareMember[] = [];
    for (const [name, entry] of Object.entries(archive.entries) as [string, any][]) {
      members.push(entry.directory ? { name, kind: 'directory', bytes: new Uint8Array() }
        : entry.symlink ? { name, kind: 'symlink', bytes: new TextEncoder().encode(await archive.readLink(name)) }
        : { name, kind: 'file', bytes: await archive.read(name) });
    }
    return members;
  } finally { await archive.close(); }
}

/** A STORE ZIP of the members, as an opened share rebuilds it. */
function membersArchive(members: ShareMember[], engine: ArchiveEngine): Promise<Uint8Array> {
  const files: Record<string, Uint8Array> = Object.create(null);
  const symlinks: string[] = [];
  for (const member of members) {
    files[member.kind === 'directory' ? member.name + '/' : member.name] = member.bytes;
    if (member.kind === 'symlink') symlinks.push(member.name);
  }
  return engine.writeArchive(files, { compression: 'store', symlinks, limits: SHARE_ARCHIVE_LIMITS });
}

/**
 * Start the codec worker on first use. Dispose on dialog/preview teardown to
 * release WASM memory. Timeouts terminate the worker, including a decoder
 * stuck on hostile input.
 */
export function createShareCodec(workerUrl?: string | URL, host: ShareHost = globalThis as any) {
  const engine = host.FlatPPLEngine;
  let worker: Worker | undefined, disposed = false, nextId = 0;
  const pending = new Map<number, {
    resolve: (reply: ShareWorkerReply) => void;
    reject: (reason: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }>();

  function stop(error: Error) {
    worker?.terminate();
    worker = undefined;
    for (const job of pending.values()) {
      clearTimeout(job.timer);
      job.reject(error);
    }
    pending.clear();
  }

  function start() {
    if (disposed) throw new Error('The share codec is closed.');
    if (worker) return worker;
    worker = new Worker(workerUrl ?? new URL('share-worker.js', document.baseURI), { type: 'module' });
    worker.addEventListener('message', (event: MessageEvent<ShareWorkerReply>) => {
      const reply = event.data;
      const job = pending.get(reply.id);
      if (!job) return;
      pending.delete(reply.id);
      clearTimeout(job.timer);
      if (reply.error) job.reject(new Error(reply.error));
      else job.resolve(reply);
    });
    worker.addEventListener('error', () => stop(new Error('The share codec could not load. Try again or download the ZIP bundle.')));
    worker.addEventListener('messageerror', () => stop(new Error('The share codec could not read its response.')));
    return worker;
  }

  function run(task: ShareWorkerTask): Promise<ShareWorkerReply> {
    return new Promise((resolve, reject) => {
      const active = start();
      const id = nextId++;
      const timer = setTimeout(() => stop(new Error('The share codec timed out. Download the ZIP bundle instead.')), WORKER_TIMEOUT_MS);
      pending.set(id, { resolve, reject, timer });
      try { active.postMessage({ id, ...task } satisfies ShareWorkerRequest); }
      catch (error) { stop(error instanceof Error ? error : new Error(String(error))); }
    });
  }

  const invalidReply = () => new Error('The share codec returned an invalid response.');

  /** Open the archive with `SHARE_ARCHIVE_LIMITS`. */
  async function decode(token: string): Promise<OpenedShare> {
    if (token.length > MAX_SHARE_URL_LENGTH) throw new Error('This share link is too long. Open the ZIP bundle instead.');
    const { link, view, members } = await run({ operation: 'decode', token });
    if (typeof link !== 'string' || !Array.isArray(members) || typeof view?.model !== 'string'
        || typeof view.target !== 'string') throw invalidReply();
    return { archive: await membersArchive(members, engine), model: view.model || null, target: view.target || null, link };
  }

  /**
   * Open a link as a visitor's page does, in an unlisted preview: mount it
   * under the share limits, select its view, resolve and register its graph.
   */
  async function opens(token: string): Promise<boolean> {
    const bundles = host.FlatPPLWebBundles;
    let path: string | undefined;
    try {
      const shared = await decode(token);
      path = (await bundles.open(shared.archive, 'share-check', () => false,
        { limits: bundles.LINK_LIMITS, listed: false })).path;
      const selection = shared.model && await bundles.select(bundles.folder(path), shared.model) || path;
      await bundles.registerGraph(await host.FlatPPLWebResolver.resolveBundle(selection, undefined, { strict: true }));
      return true;
    } catch (_) {
      return false;
    } finally {
      if (path) await bundles.discard(path);
    }
  }

  return {
    /**
     * `model` is a member path within the bundle. Null means the bundle does
     * not open as a share. ZIP download remains available.
     */
    async encode(zip: Uint8Array, selection: { model?: string; target?: string } = {}): Promise<ShareTokens | null> {
      const view = { model: selection.model === 'main.flatppl' ? '' : selection.model ?? '', target: selection.target ?? '' };
      const { tokens } = await run({ operation: 'encode', view, members: await archiveMembers(zip, engine) });
      if (tokens !== null && typeof tokens?.link !== 'string') throw invalidReply();
      return tokens && await opens(tokens.link) ? tokens : null;
    },
    decode,
    dispose() { disposed = true; stop(new Error('The share codec is closed.')); },
  };
}

/**
 * The token holds the bundle and the view, so the fragment holds nothing else.
 * A QR token starts with its marker and needs no `share=`.
 */
export function makeShareLinks(tokens: ShareTokens, baseUrl: string): ShareLinks {
  const base = new URL(baseUrl);
  base.search = '';
  base.hash = '';
  if (!['http:', 'https:'].includes(base.protocol)) throw new Error('Share links require an HTTP or HTTPS playground URL.');
  const url = base.href + '#share=' + tokens.link;
  return {
    url, qrUrl: tokens.qr === null ? null : base.href + '#' + tokens.qr,
    tooLong: url.length > MAX_SHARE_URL_LENGTH,
  };
}

/**
 * The QR library splits the text into optimal segments: numeric for the
 * token. Scheme and host ignore case, so upper case fits them into the
 * alphanumeric mode. Use medium error correction and allow every standard QR
 * size. The QR library and its renderers enter the page only when this runs.
 */
export async function createShareQr(links: ShareLinks) {
  if (links.tooLong || links.qrUrl === null) return null;
  const imported = await import('qrcode');
  const qr = imported.default ?? imported;
  const { origin } = new URL(links.qrUrl);
  const text = origin.toUpperCase() + links.qrUrl.slice(origin.length);
  let matrix: QRCode;
  try { matrix = qr.create(text, { errorCorrectionLevel: 'M' }); }
  catch (error) {
    if (error instanceof Error && error.message.includes('amount of data is too big')) return null;
    throw error;
  }
  if (matrix.version > MAX_SHARE_QR_VERSION) return null;
  const options = {
    errorCorrectionLevel: 'M' as const, version: matrix.version,
    maskPattern: matrix.maskPattern, margin: 4, scale: 8,
  };
  const [svg, png] = await Promise.all([
    qr.toString(text, { ...options, type: 'svg' }),
    qr.toDataURL(text, { ...options, type: 'image/png' }),
  ]);
  return { url: text, svg, png, size: matrix.modules.size, version: matrix.version };
}
