"""Regenerate the `load_data` fixtures (spec §07 formats).

Arrow IPC comes from polars, an independent writer. Safetensors is packed by
hand from the format definition: an 8-byte little-endian header length, a JSON
header of `{key: {dtype, shape, data_offsets}}`, then the data buffer.

Usage: python generate.py  (writes next to this file; needs numpy, polars and
pyarrow)
"""

import json
import struct
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.ipc as ipc

HERE = Path(__file__).parent

# One table in every format: a real, an integer and a boolean column, an extra
# column a selection may omit, and three rows of which tests read two.
events = {
    "a": [1.5, -2.0, 3.25],
    "b": [2, 0, 7],
    "flag": [True, False, True],
    "unused": [9.0, 9.0, 9.0],
}

frame = pl.DataFrame(events, schema={"a": pl.Float64, "b": pl.Int32, "flag": pl.Boolean, "unused": pl.Float32})
frame.write_ipc(HERE / "events.arrow")
frame.write_ipc_stream(HERE / "events.arrows")
(HERE / "events.json").write_text(json.dumps([dict(zip(events, row)) for row in zip(*events.values())]) + "\n")
(HERE / "events_columns.json").write_text(json.dumps(events) + "\n")
cells = lambda v: str(v).lower() if isinstance(v, bool) else str(v)
rows = [",".join(events)] + [",".join(cells(v) for v in row) for row in zip(*events.values())]
(HERE / "events.csv").write_text("\n".join(rows) + "\n")
rows = ["  ".join(events)] + ["\t".join(cells(v) for v in row) for row in zip(*events.values())]
(HERE / "events.wsv").write_text("# whitespace-separated\n" + "\n".join(rows) + "\n")
(HERE / "column.csv").write_text("x\n0.5\n-1\n2.75\n")
# Untyped cells: an integral value is an integer, `true`/`false` read as 1/0.
(HERE / "untyped.csv").write_text("i,flag,x\n3,true,3.5\n3.0,false,1\n1e3,TRUE,2\n")
(HERE / "untyped.json").write_text('{"i": [3, 3.0, 1e3], "x": [3.5, 1, 2]}\n')

# Arrow columns no value set reads, beside a readable one: a typed null, a
# Null-typed column, strings, a timestamp and a fixed-size list.
pl.DataFrame({
    "a": pl.Series([1.5, 2.5], dtype=pl.Float64),
    "n": pl.Series([1.0, None], dtype=pl.Float64),
    "z": pl.Series([None, None], dtype=pl.Null),
    "s": pl.Series(["1.5", "2"], dtype=pl.String),
    "t": pl.Series([0, 1], dtype=pl.Datetime("us")),
    "l": pl.Series([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=pl.Array(pl.Float64, 3)),
}).write_ipc(HERE / "mixed.arrow")

DTYPES = {np.float64: "F64", np.float32: "F32", np.float16: "F16", np.int64: "I64",
          np.int32: "I32", np.int16: "I16", np.int8: "I8", np.uint64: "U64", np.uint32: "U32",
          np.uint16: "U16", np.uint8: "U8", np.bool_: "BOOL"}


def safetensors(path, tensors, dtype_override=None):
    header, blobs, offset = {"__metadata__": {"format": "pt"}}, [], 0
    for key, array in tensors.items():
        blob = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<")).tobytes()
        dtype = (dtype_override or {}).get(key) or DTYPES[array.dtype.type]
        header[key] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(blob)]}
        blobs.append(blob)
        offset += len(blob)
    text = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + b"".join(blobs))


safetensors(HERE / "net.safetensors", {
    "W": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.5]], dtype=np.float32),
    "b": np.array([0.25, -0.5, 8.0], dtype=np.float64),
    "enc.layer.weight": np.array([3, -4], dtype=np.int32),
    ".enc.layer.mask.": np.array([True, False]),
    "enc.layer.half": np.array([1.5, -0.125], dtype=np.float16),
    "count": np.array(5, dtype=np.int64),
})
one = np.array([1.0], dtype=np.float32)
for name, tensors in {
    "empty_segment": {"enc..weight": one},
    "empty_key": {"...": one},
    "duplicate_path": {"enc.weight": one, ".enc.weight": one},
    "prefix_and_leaf": {"enc": one, "enc.weight": one},
}.items():
    safetensors(HERE / f"{name}.safetensors", tensors)
safetensors(HERE / "fp8.safetensors", {"x": np.array([1], dtype=np.uint8)}, {"x": "F8_E4M3"})

# Every dtype §07 maps, one tensor each. BF16 has no numpy type, so its bits are
# packed by hand: 0x3FC0 is 1.5 and 0xC040 is -3.0.
safetensors(HERE / "dtypes.safetensors", {
    "f64": np.array([1.5, -2.0], dtype=np.float64),
    "f32": np.array([1.5, -2.0], dtype=np.float32),
    "f16": np.array([1.5, -2.0], dtype=np.float16),
    "bf16": np.array([0x3FC0, 0xC040], dtype=np.uint16),
    "i64": np.array([1, -2], dtype=np.int64),
    "i32": np.array([1, -2], dtype=np.int32),
    "i16": np.array([1, -2], dtype=np.int16),
    "i8": np.array([1, -2], dtype=np.int8),
    "u64": np.array([1, 2], dtype=np.uint64),
    "u32": np.array([1, 2], dtype=np.uint32),
    "u16": np.array([1, 2], dtype=np.uint16),
    "u8": np.array([1, 2], dtype=np.uint8),
    "bool": np.array([True, False]),
}, {"bf16": "BF16"})
safetensors(HERE / "u64.safetensors", {"x": np.array([2**63], dtype=np.uint64)})

# Every Arrow layout a reader must step over, before the typed columns it reads,
# in two record batches. A wrong buffer count shifts every later column.
exotic = {
    "utf8": pa.array(["a", None], pa.utf8()),
    "large_utf8": pa.array(["a", "b"], pa.large_utf8()),
    "view": pa.array(["a long string past the inline view", "b"], pa.string_view()),
    "binary": pa.array([b"a", b"b"], pa.binary()),
    "list": pa.array([[1.0], [2.0, 3.0]], pa.list_(pa.float64())),
    "large_list": pa.array([[1.0], [2.0]], pa.large_list(pa.float64())),
    "list_view": pa.array([[1.0], [2.0]], pa.list_view(pa.float64())),
    "fixed_list": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float64(), 2)),
    "struct": pa.array([{"u": 1.0}, {"u": 2.0}], pa.struct([("u", pa.float64())])),
    "dict": pa.array(["x", "y"]).dictionary_encode(),
    "sparse": pa.UnionArray.from_sparse(pa.array([0, 1], pa.int8()),
                                         [pa.array([1.0, 2.0]), pa.array([3, 4], pa.int32())]),
    "dense": pa.UnionArray.from_dense(pa.array([0, 1], pa.int8()), pa.array([0, 0], pa.int32()),
                                       [pa.array([1.0]), pa.array([3], pa.int32())]),
    "ree": pa.RunEndEncodedArray.from_arrays(pa.array([2], pa.int32()), pa.array([7.0])),
    "time": pa.array([0, 1], pa.timestamp("us")),
    "null": pa.nulls(2),
}
typed = {
    "i8": pa.array([1, -2], pa.int8()), "i16": pa.array([1, -2], pa.int16()),
    "i32": pa.array([1, -2], pa.int32()), "i64": pa.array([1, -2], pa.int64()),
    "u8": pa.array([1, 2], pa.uint8()), "u16": pa.array([1, 2], pa.uint16()),
    "u32": pa.array([1, 2], pa.uint32()), "u64": pa.array([1, 2], pa.uint64()),
    "f16": pa.array([np.float16(1.5), np.float16(-2.0)], pa.float16()),
    "f32": pa.array([1.5, -2.0], pa.float32()), "f64": pa.array([1.5, None], pa.float64()),
    "bool": pa.array([True, False], pa.bool_()),
    # Dictionary-encoded numbers: keys out of value order, and a null key.
    "dict_f64": pa.array([1.5, -2.0]).dictionary_encode(),
    "dict_i16": pa.DictionaryArray.from_arrays(pa.array([1, 0], pa.int8()),
                                               pa.array([-2, 1], pa.int16())),
    "dict_null": pa.DictionaryArray.from_arrays(pa.array([0, None], pa.int8()),
                                                pa.array([1.5])),
}
batch = pa.record_batch({**exotic, **typed})
table = pa.Table.from_batches([batch, batch])
with ipc.new_file(HERE / "types.arrow", table.schema) as writer:
    writer.write_table(table)
with ipc.new_stream(HERE / "types.arrows", table.schema) as writer:
    writer.write_table(table)
with ipc.new_file(HERE / "compressed.arrow", table.schema,
                  options=ipc.IpcWriteOptions(compression="lz4")) as writer:
    writer.write_table(table)
# Long LZ4 matches and literal runs, which take extended length bytes.
runs = np.concatenate([np.full(1000, 1.5), np.arange(100) * 0.25])
with ipc.new_file(HERE / "lz4_runs.arrow", pa.schema([("x", pa.float64())]),
                  options=ipc.IpcWriteOptions(compression="lz4")) as writer:
    writer.write_table(pa.table({"x": runs}))
with ipc.new_file(HERE / "zstd.arrow", table.schema,
                  options=ipc.IpcWriteOptions(compression="zstd")) as writer:
    writer.write_table(table)
# A dictionary key outside its dictionary is a malformed file.
bad_key = pa.DictionaryArray.from_arrays(pa.array([0, 5], pa.int8()), pa.array([1.5]), safe=False)
with ipc.new_stream(HERE / "bad_key.arrows", pa.schema([("x", bad_key.type)])) as writer:
    writer.write_batch(pa.record_batch({"x": bad_key}))
# A record batch whose body length steps back onto its own message.
sink = pa.BufferOutputStream()
with ipc.new_stream(sink, pa.schema([("a", pa.float64()), ("b", pa.float64())])) as writer:
    writer.write_batch(pa.record_batch({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}))
stream = bytearray(sink.getvalue().to_pybytes())
at = 8 + struct.unpack_from("<i", stream, 4)[0]  # past the schema message
length = struct.unpack_from("<i", stream, at + 4)[0]
metadata = stream[at + 8:at + 8 + length]
assert metadata.count(struct.pack("<q", 48)) == 1  # the body length
stream[at + 8:at + 8 + length] = metadata.replace(struct.pack("<q", 48), struct.pack("<q", -(8 + length)))
(HERE / "negative_body.arrows").write_bytes(bytes(stream))


def patched(path, data, old, new, region=slice(None)):
    """Write `data` with the one occurrence of `old` in `region` replaced by `new`."""
    data = bytearray(data)
    part = data[region]
    assert part.count(old) == 1, path
    data[region] = part.replace(old, new)
    (HERE / path).write_bytes(bytes(data))


# Column `b`'s data buffer at a negative offset.
patched("negative_offset.arrows", sink.getvalue().to_pybytes(),
        struct.pack("<4q", 24, 0, 24, 24), struct.pack("<4q", 24, 0, -24, 24))
# A Null column, and its batch, claiming 2^40 rows.
sink = pa.BufferOutputStream()
with ipc.new_stream(sink, pa.schema([("n", pa.null())])) as writer:
    writer.write_batch(pa.record_batch({"n": pa.nulls(3)}))
rows = bytearray(sink.getvalue().to_pybytes())
at = 8 + struct.unpack_from("<i", rows, 4)[0]
end = at + 8 + struct.unpack_from("<i", rows, at + 4)[0]
# The batch length and the node's length and null count.
assert rows[at:end].count(struct.pack("<q", 3)) == 3
rows[at:end] = rows[at:end].replace(struct.pack("<q", 3), struct.pack("<q", 2**40))
(HERE / "huge_rows.arrows").write_bytes(bytes(rows))
# A file whose record batch block points past the end.
sink = pa.BufferOutputStream()
with ipc.new_file(sink, pa.schema([("a", pa.float64())])) as writer:
    writer.write_batch(pa.record_batch({"a": [1.5]}))
data = sink.getvalue().to_pybytes()
batch_at = 8 + 8 + struct.unpack_from("<i", data, 12)[0]
footer = len(data) - 10 - struct.unpack_from("<i", data, len(data) - 10)[0]
patched("bad_block.arrow", data, struct.pack("<q", batch_at), struct.pack("<q", 10**9),
        slice(footer, len(data) - 10))
# The out-of-range column must not spoil the readable one beside it.
with ipc.new_file(HERE / "u64.arrow", pa.schema([("a", pa.float64()), ("x", pa.uint64())])) as writer:
    writer.write_table(pa.table({"a": pa.array([1.5], pa.float64()),
                                 "x": pa.array([2**63], pa.uint64())}))
