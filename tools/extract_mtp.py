"""Extract one model's MTP block (blk.<n_layers>.*) from a GGUF into a standalone GGUF.

Strata's --mtp-gguf wants a file whose every tensor is that block; this writes exactly that: the parent's
metadata is copied verbatim, only the tensor list is replaced.

  python extract_mtp.py <parent.gguf> <out.gguf> [--block 40]
"""
from __future__ import annotations

import argparse
import struct
import sys

# bytes per block, and elements per block; F32/F16 are per element
BLOCK = {
    0: (1, 4), 1: (1, 2),
    2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 84), 11: (256, 110), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2),
}


def u32(b, o): return struct.unpack_from('<I', b, o)[0]
def u64(b, o): return struct.unpack_from('<Q', b, o)[0]


def read_str(b, o):
    n = u64(b, o)
    o += 8
    return b[o:o + n].decode('utf-8', 'replace'), o + n


def skip_kv(b, o):
    """Returns the offset just past one KV entry."""
    _, o = read_str(b, o)
    t = u32(b, o); o += 4
    if t in (0, 1, 7): return o + 1
    if t in (2, 3): return o + 2
    if t in (4, 5, 6): return o + 4
    if t in (10, 11, 12): return o + 8
    if t == 8:
        _, o = read_str(b, o); return o
    if t == 9:
        et = u32(b, o); n = u64(b, o + 4); o += 12
        sz = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}.get(et)
        if et == 8:
            for _ in range(n):
                _, o = read_str(b, o)
            return o
        return o + n * sz
    raise ValueError('kv type %d' % t)


def put_str(s):
    e = s.encode('utf-8')
    return struct.pack('<Q', len(e)) + e


def tensor_bytes(ty, dims):
    n = 1
    for d in dims:
        n *= d
    per_block, nbytes = BLOCK[ty]
    assert n % per_block == 0, (ty, dims)
    return n // per_block * nbytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('parent')
    ap.add_argument('out')
    ap.add_argument('--block', type=int, default=40)
    a = ap.parse_args()

    with open(a.parent, 'rb') as f:
        head = f.read(64 << 20)          # header + tensor infos must fit
        assert head[:4] == b'GGUF', 'not a gguf'
        ver = u32(head, 4)
        ntensors = u64(head, 8)
        nkv = u64(head, 16)
        o = 24
        kv_start = o
        for _ in range(nkv):
            o = skip_kv(head, o)
        kv_end = o
        infos = []
        for _ in range(ntensors):
            name, o = read_str(head, o)
            nd = u32(head, o); o += 4
            dims = [u64(head, o + 8 * i) for i in range(nd)]
            o += 8 * nd
            ty = u32(head, o); o += 4
            off = u64(head, o); o += 8
            infos.append((name, dims, ty, off))
        infos_end = o
        data_start = (infos_end + 31) // 32 * 32
        f.seek(0)
        f.seek(data_start)

        pre = 'blk.%d.' % a.block
        picked = [t for t in infos if t[0].startswith(pre)]
        if not picked:
            print('no tensors with prefix', pre)
            return 2
        print('extracting %d tensors (%s*)' % (len(picked), pre))
        blobs = []
        cur = 0
        for name, dims, ty, off in picked:
            nb = tensor_bytes(ty, dims)
            f.seek(data_start + off)
            b = f.read(nb)
            assert len(b) == nb, (name, nb)
            aligned = (cur + 31) // 32 * 32
            if aligned != cur:
                blobs.append(b'\0' * (aligned - cur))
                cur = aligned
            blobs.append(b)
            print('  %-42s type %2d %s %d bytes' % (name, ty, dims, nb))
            infos_new = None
            cur += nb

    # rebuild: header with the new count, the parent's KV bytes verbatim, our tensor infos, then the data
    out_header = bytearray()
    out_header += b'GGUF' + struct.pack('<I', ver) + struct.pack('<Q', len(picked)) + struct.pack('<Q', nkv)
    out_header += head[kv_start:kv_end]
    offsets = []
    cur = 0
    for name, dims, ty, _ in picked:
        cur = (cur + 31) // 32 * 32
        offsets.append(cur)
        cur += tensor_bytes(ty, dims)
    for (name, dims, ty, _), off in zip(picked, offsets):
        out_header += put_str(name) + struct.pack('<I', len(dims))
        for d in dims:
            out_header += struct.pack('<Q', d)
        out_header += struct.pack('<I', ty) + struct.pack('<Q', off)
    pad = (32 - len(out_header) % 32) % 32
    out_header += b'\0' * pad
    with open(a.out, 'wb') as g:
        g.write(out_header)
        for b in blobs:
            g.write(b)
    print('wrote', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
