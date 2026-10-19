// Host-owned archive state is asynchronous. BundleGraph is structured-clone-safe.
export type LoadDependency = {
  source: string | null;
  kind: 'modules' | 'data';
  loc: any;
  argumentLoc: any;
};
export type ArchiveLimits = {
  maxArchiveBytes: number;
  maxMemberBytes: number;
  maxTotalBytes: number;
  maxEntries: number;
  maxDepth: number;
  maxPathBytes: number;
};
export type ArchiveEntry = { size: number; directory: boolean; symlink: boolean };
export type ArchiveInfo = {
  root: string;
  entries: Record<string, ArchiveEntry>;
  parent?: string;
  directory?: string;
};
export type Archive = ArchiveInfo & {
  id: string;
  entryPath: string;
  resolve(path: string): Promise<string>;
  stat(path: string): Promise<ArchiveEntry | null>;
  read(path: string): Promise<Uint8Array>;
  readLink(path: string): Promise<string>;
  subdirectory(path: string): Promise<Archive>;
  close(): Promise<void>;
};
export type BundleResolutions = Record<string, {
  modules: Record<string, string>;
  data: Record<string, string>;
}>;
export type BundleGraph = {
  primaryPath: string;
  primarySource: string;
  sources: Record<string, string>;
  files: Record<string, Uint8Array>;
  resolutions: BundleResolutions;
  archives: Record<string, ArchiveInfo>;
};
export type ReadBytes = (path: string) => Promise<Uint8Array | null | undefined>;
