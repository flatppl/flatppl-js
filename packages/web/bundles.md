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

Use **Tools → Download bundle** to collect a model's reachable imports and data.
Export adjusts import paths without changing the original files. Missing
dependencies make export fail. Undeclared external input values are not captured.
Opening an existing ZIP preserves all its files, including binary data.

**Share** is available in the web playground. It takes one snapshot of any model
and its edited dependencies, including custom models. The link
holds a Brotli-compressed ZIP in the fragment. The browser keeps that fragment
while navigating members. Reload restores the shared snapshot, not later edits.
Save or download edits before closing the tab.

Opening a link never saves a user file. Save asks for one folder name. Saved
bundles use IndexedDB. A storage failure leaves the preview available for
download. Existing loose files remain accessible. Two bundles count as equal
only when their member paths, contents, and link metadata match.

Links open a source preview. **Run preview** enables analysis, imports, plots,
and other document views. Inspect unfamiliar files first: running a model can
use substantial time and memory. Saving a preview does not run it.
The Share dialog lists every file in the exported snapshot. Mounted bundles
include unused members too.

The QR image can use Base32 to fit more content than a byte-encoded URL. The
browser selects the smaller QR matrix and exports an SVG suitable for print.
QR codes support version 40 with medium error correction.
Large models should use a downloadable ZIP instead.

The browser accepts STORE and Deflate ZIPs. Defaults limit a compressed archive
to 64 MiB, one member to 128 MiB, total expanded data to 256 MiB, 10,000 entries,
and eight nested archives. Decoders check actual output sizes and CRCs.
Paths and symbolic-link targets have a 65,535-byte limit, matching ZIP's filename field.

Share links have a 64 KiB URL limit. URL-opened bundles have stricter browser
limits: 8 MiB per archive and in total expanded members, 2 MiB per member,
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
