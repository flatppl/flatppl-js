/**
 * Browser transport for complete §04 bundles. A share holds Brotli-compressed
 * STORE ZIP bytes in a fragment, never in a request to a storage service.
 * The archive reader owns member validation. This layer bounds the outer codec.
 * Load this module only when opening or making a share.
 */
import type { QRCode, QRCodeSegment } from 'qrcode';

export const MAX_SHARE_URL_LENGTH = 64 * 1024;
export const MAX_SHARE_ARCHIVE_BYTES = 8 * 1024 * 1024;
export const MAX_SHARE_COMPRESSED_BYTES = 1024 * 1024;
export const MAX_SHARE_QR_VERSION = 40;

const BASE32 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
const WORKER_TIMEOUT_MS = 60_000;

export interface SharePayload {
  base64Token: string;
  base32Token: string;
  archiveBytes: number;
  compressedBytes: number;
}

export interface ShareLinks {
  url: string;
  qrUrl: string;
  tooLong: boolean;
  urlLength: number;
  qrPrefix: string;
  qrToken: string;
  qrSuffix: string;
}

export type ShareWorkerRequest = { id: number; operation: 'encode' | 'decode'; bytes: Uint8Array };
export type ShareWorkerReply = { id: number; bytes?: Uint8Array | null; error?: string };

/** Encode RFC 4648 base64url without padding or large argument lists. */
function base64url(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/** Base32 costs more URL characters but fewer QR bits in alphanumeric mode. */
function base32(bytes: Uint8Array): string {
  let result = '', bits = 0, value = 0;
  for (const byte of bytes) {
    value = (value << 8) | byte;
    bits += 8;
    while (bits >= 5) {
      bits -= 5;
      result += BASE32[(value >>> bits) & 31];
    }
  }
  if (bits) result += BASE32[(value << (5 - bits)) & 31];
  return result;
}

/** Require a single canonical spelling, including zero unused trailing bits. */
function tokenBytes(token: string): Uint8Array {
  if (token.startsWith('br1.')) {
    const body = token.slice(4);
    if (!body || body.length > Math.ceil(MAX_SHARE_COMPRESSED_BYTES * 4 / 3)
        || !/^[A-Za-z0-9_-]+$/.test(body) || body.length % 4 === 1) {
      throw new Error('Invalid or oversized share link. Open the ZIP bundle instead.');
    }
    const binary = atob(body.replace(/-/g, '+').replace(/_/g, '/'));
    const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
    if (bytes.length > MAX_SHARE_COMPRESSED_BYTES || base64url(bytes) !== body) {
      throw new Error('Invalid share encoding.');
    }
    return bytes;
  }
  if (token.startsWith('BR1.')) {
    const body = token.slice(4);
    if (!body || body.length > Math.ceil(MAX_SHARE_COMPRESSED_BYTES * 8 / 5)
        || !/^[A-Z2-7]+$/.test(body)) {
      throw new Error('Invalid or oversized QR share. Open the ZIP bundle instead.');
    }
    const bytes = new Uint8Array(Math.floor(body.length * 5 / 8));
    let value = 0, bits = 0, offset = 0;
    for (const char of body) {
      value = (value << 5) | BASE32.indexOf(char);
      bits += 5;
      if (bits >= 8) {
        bits -= 8;
        bytes[offset++] = (value >>> bits) & 255;
      }
    }
    if (bytes.length > MAX_SHARE_COMPRESSED_BYTES || base32(bytes) !== body) {
      throw new Error('Invalid share encoding.');
    }
    return bytes;
  }
  throw new Error('This share link uses an unsupported codec version.');
}

/**
 * Start the codec worker on first use. Keep caller-owned snapshots intact by
 * transferring a copy. Dispose on dialog/preview teardown to release WASM memory.
 * Timeouts terminate the worker, including a decoder stuck on hostile input.
 */
export function createShareCodec(workerUrl?: string | URL) {
  let worker: Worker | undefined, disposed = false, nextId = 0;
  const pending = new Map<number, {
    resolve: (value: Uint8Array | null) => void;
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
      else if (reply.bytes === null || reply.bytes instanceof Uint8Array) job.resolve(reply.bytes);
      else job.reject(new Error('The share codec returned an invalid response.'));
    });
    worker.addEventListener('error', () => stop(new Error('The share codec could not load. Try again or download the ZIP bundle.')));
    worker.addEventListener('messageerror', () => stop(new Error('The share codec could not read its response.')));
    return worker;
  }

  function run(operation: ShareWorkerRequest['operation'], bytes: Uint8Array): Promise<Uint8Array | null> {
    return new Promise((resolve, reject) => {
      const active = start();
      const id = nextId++;
      const timer = setTimeout(() => stop(new Error('The share codec timed out. Download the ZIP bundle instead.')), WORKER_TIMEOUT_MS);
      pending.set(id, { resolve, reject, timer });
      const owned = bytes.slice();
      try { active.postMessage({ id, operation, bytes: owned } satisfies ShareWorkerRequest, [owned.buffer]); }
      catch (error) { stop(error instanceof Error ? error : new Error(String(error))); }
    });
  }

  return {
    /** Null means the bundle exceeds the transport budget. ZIP download remains available. */
    async encode(zip: Uint8Array): Promise<SharePayload | null> {
      if (zip.byteLength > MAX_SHARE_ARCHIVE_BYTES) return null;
      const bytes = await run('encode', zip);
      return bytes === null ? null : {
        base64Token: 'br1.' + base64url(bytes),
        base32Token: 'BR1.' + base32(bytes),
        archiveBytes: zip.byteLength,
        compressedBytes: bytes.byteLength,
      };
    },
    async decode(token: string): Promise<Uint8Array> {
      const bytes = await run('decode', tokenBytes(token));
      if (bytes === null) throw new Error('The shared bundle exceeds the size limit. Open the ZIP bundle instead.');
      return bytes;
    },
    dispose() { disposed = true; stop(new Error('The share codec is closed.')); },
  };
}

/** Keep all model data and view state in the fragment, including QR links. */
export function makeShareLinks(payload: SharePayload, baseUrl: string, selection: { model?: string; target?: string } = {}): ShareLinks {
  const base = new URL(baseUrl);
  base.search = '';
  base.hash = '';
  if (!['http:', 'https:'].includes(base.protocol)) throw new Error('Share links require an HTTP or HTTPS playground URL.');
  const qrPrefix = base.href + '#share=';
  const suffix = new URLSearchParams();
  if (selection.model) suffix.set('model', selection.model);
  if (selection.target) suffix.set('target', selection.target);
  const qrSuffix = suffix.size ? '&' + suffix : '';
  const url = qrPrefix + payload.base64Token + qrSuffix;
  return {
    url, qrUrl: qrPrefix + payload.base32Token + qrSuffix,
    tooLong: url.length > MAX_SHARE_URL_LENGTH, urlLength: url.length,
    qrPrefix, qrToken: payload.base32Token, qrSuffix,
  };
}

/**
 * Compare actual QR matrices, not URL lengths. Use medium error correction and
 * allow every standard QR size. Both candidates retain the same state.
 * The QR library and its renderers enter the page only when this function runs.
 */
export async function createShareQr(links: ShareLinks) {
  if (links.tooLong) return null;
  const imported = await import('qrcode');
  const qr = imported.default ?? imported;
  const utf8 = new TextEncoder();
  const candidates: { url: string; encoding: 'base64url' | 'base32'; segments: QRCodeSegment[] }[] = [
    { url: links.url, encoding: 'base64url', segments: [{ data: utf8.encode(links.url), mode: 'byte' }] },
    { url: links.qrUrl, encoding: 'base32', segments: [
      { data: utf8.encode(links.qrPrefix), mode: 'byte' },
      { data: links.qrToken, mode: 'alphanumeric' },
      ...(links.qrSuffix ? [{ data: utf8.encode(links.qrSuffix), mode: 'byte' as const }] : []),
    ] },
  ];
  let best: (typeof candidates[number] & { matrix: QRCode }) | undefined;
  for (const candidate of candidates) {
    // A version-40 M symbol holds at most 3391 alphanumeric or 2331 byte
    // characters. Avoid constructing a huge matrix just to find overflow.
    if (candidate.url.length > 3391) continue;
    let matrix: QRCode;
    try { matrix = qr.create(candidate.segments, { errorCorrectionLevel: 'M' }); }
    catch (error) {
      if (error instanceof Error && error.message.includes('amount of data is too big')) continue;
      throw error;
    }
    if (matrix.version <= MAX_SHARE_QR_VERSION && (!best || matrix.modules.size < best.matrix.modules.size)) {
      best = { ...candidate, matrix };
    }
  }
  if (!best) return null;
  const options = {
    errorCorrectionLevel: 'M' as const, version: best.matrix.version,
    maskPattern: best.matrix.maskPattern, margin: 4, scale: 8,
  };
  const [svg, png] = await Promise.all([
    qr.toString(best.segments, { ...options, type: 'svg' }),
    qr.toDataURL(best.segments, { ...options, type: 'image/png' }),
  ]);
  return { url: best.url, encoding: best.encoding, svg, png, size: best.matrix.modules.size, version: best.matrix.version };
}
