#!/usr/bin/env python3
"""Generate original monochrome/lossless/alpha fixtures and independent HM hashes.
Usage: python3 scripts/generate_alpha_fixtures.py /path/to/HM /tmp/alpha-fixtures
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from generate_422_fixtures import box, full, u16, u32, wrap

CASES = [
    ('mono8', 0, 8, False),
    ('mono10', 0, 10, True),
    ('mono12', 0, 12, False),
    ('lossless420', 1, 10, True),
    ('lossless422', 2, 12, True),
    ('lossless444', 3, 8, True),
]


def fnv(samples):
    value = 0xcbf29ce484222325
    for sample in samples:
        value = ((value ^ sample) * 0x100000001b3) & ((1 << 64) - 1)
    return f'{value:016x}'


def parts(path):
    data = path.read_bytes()
    offset = data.index(b'hvcC') - 4
    hvcc = data[offset:offset + int.from_bytes(data[offset:offset + 4], 'big')]
    return hvcc, data[data.index(b'mdat') + 4:]


def alpha_container(colour, alpha, destination, *, grid=False, prem=False):
    hvcc_c, coded_c = parts(colour)
    hvcc_a, coded_a = parts(alpha)
    # Plain images are 64x64. Grids duplicate each original tile horizontally;
    # both colour and alpha independently rotate CCW, mirror and crop to 56x100.
    properties = [hvcc_c, full('ispe', u32(64) + u32(64)), hvcc_a,
                  full('ispe', u32(64) + u32(64)),
                  full('auxC', b'urn:mpeg:hevc:2015:auxid:1\0')]
    items = [(1, b'hvc1', coded_c, [1, 2]), (2, b'hvc1', coded_a, [3, 4, 5])]
    refs = [(b'auxl', 2, [1])]
    primary, auxiliary = 1, 2
    if grid:
        properties += [full('ispe', u32(128) + u32(64)), box('irot', b'\1'),
                       box('clap', b''.join(u32(v) for v in [56, 1, 100, 1, 0, 1, 0xfffffffC, 1])), box('imir', b'\1')]
        descriptor = bytes([0, 0, 0, 1]) + u16(128) + u16(64)
        items = [(1, b'hvc1', coded_c, [1, 2]), (2, b'hvc1', coded_c, [1, 2]),
                 (3, b'hvc1', coded_a, [3, 4]), (4, b'hvc1', coded_a, [3, 4]),
                 (5, b'grid', descriptor, [6, 7, 9, 8]), (6, b'grid', descriptor, [6, 7, 9, 8, 5])]
        primary, auxiliary = 5, 6
        refs = [(b'dimg', 5, [1, 2]), (b'dimg', 6, [3, 4]), (b'auxl', 6, [5])]
    if prem:
        refs += [(b'prem', primary, [auxiliary])]
    associations = b''.join(u16(i) + bytes([len(props), *[0x80 | p for p in props]]) for i, _, _, props in items)
    iprp = box('iprp', box('ipco', b''.join(properties)) + full('ipma', u32(len(items)) + associations))
    iinf = full('iinf', u16(len(items)) + b''.join(full('infe', u16(i) + u16(0) + kind + b'\0', 2) for i, kind, _, _ in items))
    iref = full('iref', b''.join(box(kind.decode(), u16(source) + u16(len(targets)) + b''.join(u16(t) for t in targets)) for kind, source, targets in refs))
    ftyp = box('ftyp', b'heix' + u32(0) + b'mif1heix')

    def meta(offset):
        locations = b''
        for i, _, payload, _ in items:
            locations += u16(i) + u16(0) + u16(1) + u32(offset) + u32(len(payload))
            offset += len(payload)
        iloc = full('iloc', bytes([0x44, 0]) + u16(len(items)) + locations)
        return full('meta', full('hdlr', u32(0) + b'pict' + bytes(12) + b'\0') + full('pitm', u16(primary)) + iinf + iloc + iprp + iref)
    destination.write_bytes(ftyp + meta(len(ftyp) + len(meta(0)) + 8) + box('mdat', b''.join(payload for _, _, payload, _ in items)))


def generate(hm, out):
    out.mkdir(parents=True, exist_ok=True)
    encoder = next(hm.glob('bin/**/TAppEncoder'))
    decoder = next(hm.glob('bin/**/TAppDecoder'))
    manifest = []
    references = {}
    for name, chroma, depth, lossless in CASES:
        sub_w, sub_h = {0: (1, 1), 1: (2, 2), 2: (2, 1), 3: (1, 1)}[chroma]
        samples = []
        for component in range(1 if chroma == 0 else 3):
            width, height = (64, 64) if component == 0 else (64 // sub_w, 64 // sub_h)
            for y in range(height):
                for x in range(width):
                    # Smooth regions next to edges, plus low-bit precision and
                    # both endpoints in the alpha plane. Entirely original.
                    sample = ((x * 7 + y * 11 + component * 47 + ((x // 16 + y // 16) % 2) * 37) % 256) << (depth - 8)
                    sample += (x + y) % (1 << (depth - 8))
                    if (x + y) % 17 == 0: sample = 0
                    if (x + y) % 19 == 0: sample = (1 << depth) - 1
                    samples.append(sample)
        source = out / f'{name}.yuv'
        source.write_bytes(bytes(samples) if depth == 8 else struct.pack('<' + 'H' * len(samples), *samples))
        stream = out / f'{name}.hevc'
        recon = out / f'{name}-recon.yuv'
        reference = out / f'{name}-hm.yuv'
        command = [str(encoder), '-c', str(hm / 'cfg/encoder_intra_main.cfg'), '--Profile=main-RExt', '--Level=3',
                   f'--InputChromaFormat={ {0:400, 1:420, 2:422, 3:444}[chroma]}',
                   f'--InputBitDepth={depth}', f'--InternalBitDepth={depth}', '--SourceWidth=64', '--SourceHeight=64',
                   '--FrameRate=1', '--FramesToBeEncoded=1', '--QP=27', '--TransquantBypassEnable=1',
                   f'--CUTransquantBypassFlagForce={int(lossless)}', '--ImplicitResidualDPCM=0', '--ExplicitResidualDPCM=0',
                   '--WaveFrontSynchro=1', f'--InputFile={source}', f'--BitstreamFile={stream}', f'--ReconFile={recon}']
        with (out / f'{name}-encode.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        with (out / f'{name}-decode.log').open('w') as log:
            subprocess.run([str(decoder), '-b', str(stream), '-o', str(reference)], cwd=out, stdout=log, stderr=log, check=True)
        raw = reference.read_bytes()
        assert raw == recon.read_bytes()
        if lossless: assert raw == source.read_bytes()
        values = list(raw) if depth == 8 else list(struct.unpack('<' + 'H' * (len(raw) // 2), raw))
        references[name] = values
        wrap(stream, out / f'{name}.heic', 64, 64, depth, chroma)
        manifest.append(dict(name=name, width=64, height=64, depth=depth, chroma=chroma, lossless=lossless,
                             fnv=fnv(values), sha256=hashlib.sha256(raw).hexdigest(),
                             encoder_command=[arg.replace(f'{hm}/', 'HM/').replace(f'{out}/', '') for arg in command]))
    for name, alpha, grid, prem in [('rgba10', 'mono10', False, False), ('rgba8alpha', 'mono8', False, False), ('rgba-grid', 'mono10', True, False), ('rgba-prem', 'mono10', False, True)]:
        alpha_container(out / 'lossless420.heic', out / f'{alpha}.heic', out / f'{name}.heic', grid=grid, prem=prem)
        values = references[alpha]
        if grid:
            rows = [values[y * 64:(y + 1) * 64] * 2 for y in range(64)]
            values = [rows[63 - x][127 - y] for y in range(10, 110) for x in range(4, 60)]
        manifest.append(dict(name=name, width=56 if grid else 64, height=100 if grid else 64,
                             alpha_depth=8 if alpha == 'mono8' else 10, alpha_fnv=fnv(values), premultiplied=prem))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps([{k:v for k,v in row.items() if k != 'encoder_command'} for row in manifest], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hm', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    generate(args.hm.resolve(), args.out.resolve())
