#!/usr/bin/env python3
"""Read a duello many-body trajectory (positions and quaternions).

Layout, little-endian: magic b"DUELLO-TRAJ\\0", u32 version (1), u32 header
length, JSON header, then fixed records of u32 copy, u64 sweep, f64 box, and
per body f64 x, y, z, qw, qx, qy, qz. A trailing partial record (stopped run)
is ignored. All bodies share the header's `reference` beads.

Usage as a script prints a one-line summary; import `read` for analysis.
With numpy, `read_numpy` maps all frames in one go.
"""

import json
import struct
import sys

MAGIC = b"DUELLO-TRAJ\0"
VERSION = 1


def _header(data):
    if data[:12] != MAGIC:
        raise ValueError("not a duello trajectory")
    version, length = struct.unpack_from("<II", data, 12)
    if version != VERSION:
        raise ValueError(f"unsupported trajectory version {version}")
    header = json.loads(data[20 : 20 + length])
    return header, 20 + length


def read(path):
    """Header dict and a list of frames (copy, sweep, box, [7-tuples per body])."""
    with open(path, "rb") as f:
        data = f.read()
    header, offset = _header(data)
    n = header["n_bodies"]
    size = 20 + 56 * n
    frames = []
    while offset + size <= len(data):
        copy, sweep, box = struct.unpack_from("<IQd", data, offset)
        values = struct.unpack_from(f"<{7 * n}d", data, offset + 20)
        poses = [values[7 * i : 7 * i + 7] for i in range(n)]
        frames.append((copy, sweep, box, poses))
        offset += size
    return header, frames


def read_numpy(path):
    """Header and a structured array with fields copy, sweep, box, pose (n×7)."""
    import numpy as np

    with open(path, "rb") as f:
        data = f.read()
    header, offset = _header(data)
    n = header["n_bodies"]
    dtype = np.dtype(
        [("copy", "<u4"), ("sweep", "<u8"), ("box", "<f8"), ("pose", "<f8", (n, 7))]
    )
    count = (len(data) - offset) // dtype.itemsize
    return header, np.frombuffer(data, dtype=dtype, count=count, offset=offset)


if __name__ == "__main__":
    header, frames = read(sys.argv[1])
    summary = f"frames={len(frames)} bodies={header['n_bodies']}"
    if frames:  # a run stopped before its first frame has none
        copy, sweep, box, poses = frames[0]
        first = ",".join(repr(v) for v in poses[0])
        summary += f" copy={copy} sweep={sweep} box={box!r} first={first}"
    print(summary)
