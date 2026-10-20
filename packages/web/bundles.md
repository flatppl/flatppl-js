# Portable bundles and sharing

A FlatPPL bundle is a ZIP containing `main.flatppl`. It can also contain modules,
data, and other files. A single enclosing directory is accepted. For example:

```text
model.flatppl.zip
├── main.flatppl
├── modules/prior.flatppl
└── data/observations.json
```

Inside `main.flatppl`, load these files by relative path:

```flatppl
prior = load_module("modules/prior.flatppl")
observations = load_data("data/observations.json", cartpow(reals, 2))
```

Imports and data stay inside their bundle. Missing members do not fall through
to the network. Nested ZIPs retain separate roots. Symbolic links must remain
inside the root. Entry names cannot be absolute or contain `..` segments.
Relative imports such as `modules/../data/observations.json` are valid within the root.

Use **Tools → Download with dependencies** to collect a model's reachable imports and data.
Export adjusts import paths without changing the original files. Missing
dependencies make export fail. Undeclared external input values are not captured.
Opening an existing ZIP preserves all its files, including binary data.
A standalone model downloads as one `.flatppl` file. A ZIP containing only a
regular `main.flatppl` also opens as one file, without a bundle folder.

**Share** is available in the web playground. It takes one snapshot of any model
and its edited dependencies, including custom models. The link
holds the compressed bundle in the fragment. The browser keeps that fragment
while navigating members. Reload restores the shared snapshot, not later edits.
Save or download edits before closing the tab.

Opening a link never saves a user file. Standalone shares open as
`shared-model.flatppl`; the link format does not retain the original filename.
Save asks for a filename or, for a bundle, a folder name. Archives use IndexedDB
even when shown as one file, retaining their import limits and confinement.
A storage failure leaves the preview available for
download. Existing loose files remain accessible. Two bundles count as equal
only when their member paths, contents, and link metadata match.
The header's Delete or Discard action removes the entire opened bundle, after confirmation.

Links open a source preview. **Run preview** enables analysis, imports, plots,
and other document views. Inspect unfamiliar files first: running a model can
use substantial time and memory. Saving a preview does not run it.
The Share dialog lists every file in the exported snapshot. Mounted bundles
include unused members too.

The QR code holds the same content as decimal digits, which QR codes store
more compactly than other text. It uses medium error correction and up to
version 40. The browser exports an SVG suitable for print. Large models should
use a downloadable ZIP instead. [Share-link format](share-links.md) describes
the encoding.

The browser accepts STORE and Deflate ZIPs. Defaults limit a compressed archive
to 64 MiB, one member to 128 MiB, total expanded data to 256 MiB, 10,000 entries,
and eight nested archives. Decoders check actual output sizes and CRCs.
Paths and symbolic-link targets have a 65,535-byte limit, matching ZIP's filename field.

Share links have a 64 KiB URL limit. URL-opened bundles have stricter browser
limits: 9 MiB per archive (including ZIP headers), 8 MiB in total expanded members, 2 MiB per member,
256 entries, four nested archives, 1,024 bytes per path, and 32 folder components.
Local ZIP uploads retain the general limits above. Both STORE and Deflate work.
The Math view limits data conversion to 8 MiB before constructing its JSON request.

Document images require consent before loading, including images in Markdown
hovers and the Math view. Image approval lasts for the current page session.
HTML opens in a sandbox with network loads blocked. **Load external resources**
allows its images, styles, fonts, media, and frames to contact servers, including
redirect destinations. Scripts and access to playground storage stay blocked.

You can host a ZIP yourself and open its URL through a `#model=` link. The host
must allow browser access through CORS. Otherwise download the ZIP and open it
locally. The playground provides no storage service, proxy, or short-link service.
