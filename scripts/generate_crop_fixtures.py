#!/usr/bin/env python3
"""Generate conformance-window HEIF fixtures and HM reference hashes.

Usage: python3 scripts/generate_crop_fixtures.py /path/to/HM /tmp/crop-fixtures
The grid reference is assembled independently from HM's cropped tile output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess

from generate_422_fixtures import box, full, u16, u32, wrap

CASES = [
    ('crop420', 1, 8, [2, 4, 4, 6]),
    ('crop422', 2, 10, [2, 4, 3, 5]),
    ('crop444', 3, 12, [1, 3, 3, 5]),
]


def reference_entry(name, width, height, cw, ch, depth, chroma, raw, **details):
    samples = list(raw) if depth == 8 else struct.unpack('<' + 'H' * (len(raw) // 2), raw)
    assert len(samples) == width * height + 2 * cw * ch
    value = 0xcbf29ce484222325
    for sample in samples:
        value = ((value ^ sample) * 0x100000001b3) & ((1 << 64) - 1)
    return dict(name=name, width=width, height=height, cb_width=cw, cb_height=ch,
                depth=depth, chroma=chroma, fnv=f'{value:016x}',
                sha256=hashlib.sha256(raw).hexdigest(), **details)


def crop_grid(tile_file, destination):
    # Reuse the hvcC and coded tile data from our own one-item fixture.
    tile = tile_file.read_bytes()
    at = tile.index(b'hvcC')
    hvcc = tile[at - 4:at - 4 + int.from_bytes(tile[at - 4:at], 'big')]
    at = tile.index(b'mdat')
    coded = tile[at + 4:]
    # Two 58x56 conformance-cropped tiles, with the rightmost four columns
    # outside the 112x56 grid. Rotate, then crop to 40x84 at (10,17).
    grid = bytes([0, 0, 0, 1]) + u16(112) + u16(56)
    ipco = box('ipco', hvcc + full('ispe', u32(58) + u32(56))
               + full('ispe', u32(112) + u32(56)) + box('irot', b'\1')
               + box('clap', b''.join(u32(v) for v in [40, 1, 84, 1, 2, 1, 3, 1])))
    ipma = full('ipma', u32(3) + u16(1) + bytes([2, 0x81, 0x82])
                + u16(2) + bytes([2, 0x81, 0x82])
                + u16(3) + bytes([3, 0x83, 0x84, 0x85]))
    iprp = box('iprp', ipco + ipma)
    entries = b''.join(full('infe', u16(i) + u16(0) + kind + b'\0', 2)
                       for i, kind in [(1, b'hvc1'), (2, b'hvc1'), (3, b'grid')])
    iinf = full('iinf', u16(3) + entries)
    iref = full('iref', box('dimg', u16(3) + u16(2) + u16(1) + u16(2)))
    ftyp = box('ftyp', b'heix' + u32(0) + b'mif1heix')

    def meta(offset):
        locations = bytearray()
        for i, payload in [(1, coded), (2, coded), (3, grid)]:
            locations += u16(i) + u16(0) + u16(1) + u32(offset) + u32(len(payload))
            offset += len(payload)
        iloc = full('iloc', bytes([0x44, 0]) + u16(3) + locations)
        hdlr = full('hdlr', u32(0) + b'pict' + bytes(12) + b'\0')
        return full('meta', hdlr + full('pitm', u16(3)) + iinf + iloc + iprp + iref)

    destination.write_bytes(ftyp + meta(len(ftyp) + len(meta(0)) + 8)
                            + box('mdat', coded + coded + grid))


def grid_reference(raw):
    values = struct.unpack('<' + 'H' * (len(raw) // 2), raw)
    result = []
    offset = 0
    for w, h, gw, left, top, ow, oh in [(58, 56, 112, 10, 17, 40, 84),
                                        (29, 56, 56, 10, 8, 40, 42),
                                        (29, 56, 56, 10, 8, 40, 42)]:
        tile = [values[offset + y * w:offset + (y + 1) * w] for y in range(h)]
        offset += w * h
        mosaic = [(row + row)[:gw] for row in tile]
        # A CCW rotation maps output (x,y) to input (gw-1-y,x).
        for y in range(top, top + oh):
            result.extend(mosaic[x][gw - 1 - y] for x in range(left, left + ow))
    return struct.pack('<' + 'H' * len(result), *result)


def generate(hm, out):
    out.mkdir(parents=True, exist_ok=True)
    encoder = next(hm.glob('bin/**/TAppEncoder'))
    decoder = next(hm.glob('bin/**/TAppDecoder'))
    manifest = []
    for name, chroma, depth, crop in CASES:
        sub_w, sub_h = {1: (2, 2), 2: (2, 1), 3: (1, 1)}[chroma]
        data = bytearray()
        for component, w, h in [(0, 64, 64), (1, 64 // sub_w, 64 // sub_h), (2, 64 // sub_w, 64 // sub_h)]:
            for y in range(h):
                for x in range(w):
                    sample = ((x * 7 + y * 11 + ((x // 8 + y // 8) % 2) * 39 + component * 47) % 256) << (depth - 8)
                    data += bytes([sample]) if depth == 8 else struct.pack('<H', sample)
        source = out / f'{name}.yuv'
        source.write_bytes(data)
        stream = out / f'{name}.hevc'
        reference = out / f'{name}-hm.yuv'
        command = [str(encoder), '-c', str(hm / 'cfg/encoder_intra_main.cfg'),
                   '--Profile=main-RExt', '--Level=3', '--InputChromaFormat=' + str({1: 420, 2: 422, 3: 444}[chroma]),
                   f'--InputBitDepth={depth}', f'--InternalBitDepth={depth}',
                   '--SourceWidth=64', '--SourceHeight=64', '--FrameRate=1', '--FramesToBeEncoded=1', '--QP=27',
                   '--ConformanceWindowMode=3', *[f'--ConfWin{side}={value}' for side, value in zip(['Left', 'Right', 'Top', 'Bottom'], crop)],
                   f'--InputFile={source}', f'--BitstreamFile={stream}', f'--ReconFile={out / (name + "-recon.yuv")}']
        with (out / f'{name}-encode.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        with (out / f'{name}-decode.log').open('w') as log:
            subprocess.run([str(decoder), '-b', str(stream), '-o', str(reference)], cwd=out, stdout=log, stderr=log, check=True)
        width, height = 64 - crop[0] - crop[1], 64 - crop[2] - crop[3]
        wrap(stream, out / f'{name}.heic', width, height, depth, chroma)
        manifest.append(reference_entry(name, width, height, width // sub_w, height // sub_h,
                                        depth, chroma, reference.read_bytes(), coded_width=64, coded_height=64,
                                        crop=crop, encoder_command=command))
    crop_grid(out / 'crop422.heic', out / 'grid422.heic')
    raw = grid_reference((out / 'crop422-hm.yuv').read_bytes())
    (out / 'grid422-reference.yuv').write_bytes(raw)
    manifest.append(reference_entry('grid422', 40, 84, 40, 42, 10, 2, raw,
                                    description='Two conformance-cropped tiles, grid trim, CCW rotation, offset clean aperture'))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps([{key: value for key, value in row.items() if key != 'encoder_command'} for row in manifest], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hm', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    generate(args.hm.resolve(), args.out.resolve())
