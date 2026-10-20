/**
 * Share-link wire format, specified in ../share-links.md: token text, payload
 * versions and the member container. Pure functions for the share worker.
 */

export const MAX_SHARE_URL_LENGTH = 64 * 1024;
/** A version byte and a Zstandard frame without its magic number. */
export const MAX_SHARE_PAYLOAD_BYTES = MAX_SHARE_URL_LENGTH * 3 / 4;
export const MAX_SHARE_CONTENT_BYTES = 8 * 1024 * 1024;
/** Numeric capacity of a version-40 QR code with medium error correction. */
export const MAX_SHARE_DIGITS = 5596;
/** Marks a QR token, whose digits would otherwise also read as base64url. */
const QR_MARKER = '~';
/**
 * Envelope for rebuilding a ZIP from the versioned wire container. The
 * browser's LINK_LIMITS impose tighter member, nesting and expansion limits
 * when mounting it, including the outgoing link check.
 */
export const SHARE_ARCHIVE_LIMITS = Object.freeze({
  maxArchiveBytes: 16 * 1024 * 1024,
  maxMemberBytes: 16 * 1024 * 1024,
  maxTotalBytes: 32 * 1024 * 1024,
  maxEntries: 1000,
  maxDepth: 8,
  maxPathBytes: 1024,
});

export const ZSTD_MAGIC = Uint8Array.of(0x28, 0xb5, 0x2f, 0xfd);
const KINDS = ['file', 'directory', 'symlink'] as const;

export interface ShareMember {
  name: string;
  kind: typeof KINDS[number];
  bytes: Uint8Array;
}

/** The member and binding a share opens. Empty means `main.flatppl` and no target. */
export interface ShareView {
  model: string;
  target: string;
}

const invalid = () => new Error('Invalid share link.');

/** The bundle is well-formed but exceeds what a share may hold. */
export class ShareLimitError extends Error {
  constructor() { super('The shared bundle exceeds the share-link limits. Open the ZIP bundle instead.'); }
}

/** RFC 4648 base64url without padding. */
function base64url(bytes: Uint8Array): string {
  let binary = '';
  for (let i = 0; i < bytes.length; i += 8192) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  }
  return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

function hexBytes(hex: string): Uint8Array {
  if (hex.length % 2) hex = '0' + hex;
  const bytes = new Uint8Array(hex.length / 2);
  for (let i = 0; i < bytes.length; i++) bytes[i] = parseInt(hex.slice(2 * i, 2 * i + 2), 16);
  return bytes;
}

export function linkToken(payload: Uint8Array): string {
  return base64url(payload);
}

/**
 * The marker and the payload as one decimal integer, for a QR numeric segment.
 * The nonzero version byte keeps leading zero bytes. Null when too long.
 */
export function qrToken(payload: Uint8Array): string | null {
  const digits = BigInt('0x' + Array.from(payload, byte => byte.toString(16).padStart(2, '0')).join('')).toString();
  return digits.length <= MAX_SHARE_DIGITS ? QR_MARKER + digits : null;
}

/** Accept only the single canonical spelling of either token form. */
export function tokenPayload(token: string): Uint8Array {
  if (token.startsWith(QR_MARKER)) {
    const digits = token.slice(QR_MARKER.length);
    if (digits.length > MAX_SHARE_DIGITS || !/^[1-9][0-9]*$/.test(digits)) throw invalid();
    return hexBytes(BigInt(digits).toString(16));
  }
  if (!token || token.length > Math.ceil(MAX_SHARE_PAYLOAD_BYTES * 4 / 3)
      || !/^[A-Za-z0-9_-]+$/.test(token) || token.length % 4 === 1) throw invalid();
  const binary = atob(token.replace(/-/g, '+').replace(/_/g, '/'));
  const bytes = Uint8Array.from(binary, char => char.charCodeAt(0));
  if (base64url(bytes) !== token) throw invalid();
  return bytes;
}

/** LEB128, at most four bytes, without redundant trailing groups. */
function writeLength(out: number[], value: number) {
  do {
    out.push(value > 0x7f ? (value & 0x7f) | 0x80 : value);
    value >>>= 7;
  } while (value);
}

/**
 * The view comes first, then `main.flatppl` without a name. Each further member
 * is a kind byte, a name and its bytes, both length-prefixed, in ascending name
 * order.
 */
export function encodeContainer(view: ShareView, members: ShareMember[]): Uint8Array {
  const main = members.filter(m => m.name === 'main.flatppl');
  if (main.length !== 1 || main[0].kind !== 'file') throw new Error('A bundle needs one main.flatppl file.');
  const rest = members.filter(m => m.name !== 'main.flatppl').sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
  const parts: Uint8Array[] = [];
  const utf8 = new TextEncoder();
  for (const field of [utf8.encode(view.model), utf8.encode(view.target), main[0].bytes]) {
    const length: number[] = [];
    writeLength(length, field.length);
    parts.push(Uint8Array.from(length), field);
  }
  for (const [i, member] of rest.entries()) {
    if (i && member.name === rest[i - 1].name) throw new Error(`Duplicate bundle member: '${member.name}'`);
    const name = utf8.encode(member.name);
    const header = [KINDS.indexOf(member.kind)];
    writeLength(header, name.length);
    const length: number[] = [];
    writeLength(length, member.bytes.length);
    parts.push(Uint8Array.from(header), name, Uint8Array.from(length), member.bytes);
  }
  const out = new Uint8Array(parts.reduce((n, part) => n + part.length, 0));
  let offset = 0;
  for (const part of parts) { out.set(part, offset); offset += part.length; }
  return out;
}

/**
 * Parse the whole container or nothing. The rebuilt ZIP passes the archive
 * reader, which validates names and links as for any other bundle. Lengths
 * count UTF-8 bytes, and the ZIP gives a directory name a trailing slash.
 */
export function decodeContainer(bytes: Uint8Array,
  limits: { maxEntries: number; maxPathBytes: number } = SHARE_ARCHIVE_LIMITS): { view: ShareView; members: ShareMember[] } {
  let offset = 0;
  const utf8 = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
  function length(): number {
    let value = 0;
    for (let shift = 0; shift < 28; shift += 7) {
      if (offset >= bytes.length) throw invalid();
      const byte = bytes[offset++];
      value |= (byte & 0x7f) << shift;
      if (!(byte & 0x80)) {
        if (!byte && shift) throw invalid();
        return value;
      }
    }
    throw invalid();
  }
  function take(n: number): Uint8Array {
    if (n > bytes.length - offset) throw invalid();
    return bytes.slice(offset, offset += n);
  }
  function text(raw: Uint8Array): string {
    try { return utf8.decode(raw); } catch (_) { throw invalid(); }
  }
  function field(): string {
    const raw = take(length());
    if (raw.length > limits.maxPathBytes) throw new ShareLimitError();
    return text(raw);
  }
  const view = { model: field(), target: field() };
  if (view.model === 'main.flatppl' || view.model.endsWith('/')) throw invalid();
  const members: ShareMember[] = [{ name: 'main.flatppl', kind: 'file', bytes: take(length()) }];
  let previous = '';
  while (offset < bytes.length) {
    if (members.length >= limits.maxEntries) throw new ShareLimitError();
    const kind = KINDS[bytes[offset++]];
    const raw = take(length());
    const name = text(raw);
    if (!kind || !raw.length || name <= previous || name === 'main.flatppl' || name.endsWith('/')) throw invalid();
    if (raw.length + (kind === 'directory' ? 1 : 0) > limits.maxPathBytes) throw new ShareLimitError();
    const data = take(length());
    if (kind === 'directory' && data.length) throw invalid();
    if (kind === 'symlink') {
      if (!data.length) throw invalid();
      if (data.length > limits.maxPathBytes) throw new ShareLimitError();
      text(data);
    }
    members.push({ name, kind, bytes: data });
    previous = name;
  }
  return { view, members };
}

/** Restore the frame magic number that payloads omit. */
export function payloadFrame(payload: Uint8Array): { version: number; frame: Uint8Array } {
  if (payload.length < 2) throw invalid();
  const frame = new Uint8Array(ZSTD_MAGIC.length + payload.length - 1);
  frame.set(ZSTD_MAGIC);
  frame.set(payload.subarray(1), ZSTD_MAGIC.length);
  return { version: payload[0], frame };
}

export function framePayload(version: number, frame: Uint8Array): Uint8Array {
  if (!ZSTD_MAGIC.every((byte, i) => frame[i] === byte)) throw new Error('Not a Zstandard frame.');
  const payload = new Uint8Array(frame.length - ZSTD_MAGIC.length + 1);
  payload[0] = version;
  payload.set(frame.subarray(ZSTD_MAGIC.length), 1);
  return payload;
}
