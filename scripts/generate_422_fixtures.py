#!/usr/bin/env python3
"""Generate deterministic 4:2:2 HEIF fixtures with HM (no runtime dependency).

Usage: python3 scripts/generate_422_fixtures.py /path/to/hm-reference /tmp/422-fixtures
Writes raw input, Annex-B, HEIF, independent decoder YUV and manifest.json.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import subprocess


def u16(value):
    return struct.pack('>H', value)


def u32(value):
    return struct.pack('>I', value)


def box(kind, body):
    return u32(len(body) + 8) + kind.encode('ascii') + body


def full(kind, body, version=0):
    return box(kind, bytes([version, 0, 0, 0]) + body)


def wrap(source, destination, width, height, depth, chroma=2, color=None):
    """Wrap the encoder's single-layer stream in a one-item HEIF container."""
    nalus = [nal.rstrip(b'\0') for nal in re.split(b'\x00\x00\x00?\x01', source.read_bytes())[1:] if nal]
    kind = lambda nal: (nal[0] >> 1) & 63
    params = [nal for nal in nalus if kind(nal) in (32, 33, 34)]
    data = b''.join(u32(len(nal)) + nal for nal in nalus if kind(nal) not in (32, 33, 34))
    sps = next(nal for nal in params if kind(nal) == 33)
    rbsp = sps[2:].replace(b'\x00\x00\x03', b'\x00\x00')
    # These fixtures have one temporal layer: copy the SPS's general PTL.
    hvcc = bytearray(b'\x01' + rbsp[1:13] + bytes.fromhex('f000fcfdf8f80000'))
    hvcc[16] = 0xfc | chroma
    hvcc[17] = hvcc[18] = 0xf8 + depth - 8
    arrays = b''.join(bytes([0x80 | kind(nal)]) + u16(1) + u16(len(nal)) + nal for nal in params)
    hvcc = bytes(hvcc) + bytes([3, len(params)]) + arrays  # four-byte lengths
    properties = box('hvcC', hvcc) + full('ispe', u32(width) + u32(height))
    associations = [0x81, 0x82]
    if color is not None:
        primaries, transfer, matrix, full_range = color
        properties += box('colr', b'nclx' + u16(primaries) + u16(transfer) + u16(matrix) + bytes([0x80 if full_range else 0]))
        associations.append(0x83)
    ipco = box('ipco', properties)
    iprp = box('iprp', ipco + full('ipma', u32(1) + u16(1) + bytes([len(associations), *associations])))
    iinf = full('iinf', u16(1) + full('infe', u16(1) + u16(0) + b'hvc1\0', 2))
    ftyp = box('ftyp', b'heix' + u32(0) + b'mif1heix')

    def meta(offset):
        iloc = full('iloc', bytes([0x44, 0]) + u16(1) + u16(1) + u16(0) + u16(1) + u32(offset) + u32(len(data)))
        hdlr = full('hdlr', u32(0) + b'pict' + bytes(12) + b'\0')
        return full('meta', hdlr + full('pitm', u16(1)) + iinf + iloc + iprp)

    destination.write_bytes(ftyp + meta(len(ftyp) + len(meta(0)) + 8) + box('mdat', data))


CASES = [
    ('pattern8', 64, 64, 8, 27, []),
    ('pattern10', 96, 80, 10, 18, [
        '--WaveFrontSynchro=1', '--MaxCuDQPDepth=1', '--MaxDeltaQP=2', '--ScalingList=1',
    ]),
    ('smooth10', 128, 128, 10, 44, ['--CbQpOffset=12', '--CrQpOffset=-12']),
    ('flat10', 64, 64, 10, 32, ['--QuadtreeTUMaxDepthIntra=1']),
    ('skip10', 64, 64, 10, 0, ['--LoopFilterDisable=1', '--SAO=0']),
    ('pattern12', 64, 64, 12, 24, []),
    ('scaled12', 64, 64, 12, 24, ['--SaoLumaOffsetBitShift=2', '--SaoChromaOffsetBitShift=1']),
    ('slices10', 96, 96, 10, 18, [
        '--MaxCUWidth=32', '--MaxCUHeight=32', '--MaxPartitionDepth=3',
        '--SliceMode=1', '--SliceArgument=3', '--LoopFilterDisable=1', '--SAO=0',
        '--MaxCuDQPDepth=1', '--MaxDeltaQP=2',
    ]),
]


def pattern(name, width, height, depth):
    raw = bytearray()
    for component, plane_width in [(0, width), (1, width // 2), (2, width // 2)]:
        for y in range(height):
            for x in range(plane_width):
                if name.startswith('flat'):
                    value = 400 + component * 100
                elif name.startswith('smooth'):
                    value = 120 + ((x * 3 + y * 2 + component * 23) % 180) * 3
                else:
                    value = ((x * 7 + y * 11 + ((x // 8 + y // 8) % 2) * 39 + component * 47) % 256) << (depth - 8)
                raw.extend(bytes([value]) if depth == 8 else struct.pack('<H', value))
    return raw


def generate(hm, out):
    out.mkdir(parents=True, exist_ok=True)
    encoder = next(hm.glob('bin/**/TAppEncoder'))
    decoder = next(hm.glob('bin/**/TAppDecoder'))
    manifest = []
    for name, width, height, depth, qp, extra in CASES:
        source = out / (name + '.yuv')
        source.write_bytes(pattern(name, width, height, depth))
        stream = out / (name + '.hevc')
        recon = out / (name + '-recon.yuv')
        reference = out / (name + '-hm.yuv')
        heif = out / (name + '.heic')
        command = [
            str(encoder), '-c', str(hm / 'cfg/encoder_intra_main.cfg'),
            '--Profile=main-RExt', '--Level=3', '--InputChromaFormat=422',
            f'--InputBitDepth={depth}', f'--InternalBitDepth={depth}',
            f'--SourceWidth={width}', f'--SourceHeight={height}',
            '--FrameRate=1', '--FramesToBeEncoded=1', f'--QP={qp}',
            f'--InputFile={source}', f'--BitstreamFile={stream}', f'--ReconFile={recon}',
        ] + extra
        with (out / (name + '-encode.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=log, check=True)
        with (out / (name + '-decode.log')).open('w') as log:
            subprocess.run([str(decoder), '-b', str(stream), '-o', str(reference)],
                           cwd=out, stdout=log, stderr=log, check=True)
        expected = reference.read_bytes()
        assert expected == recon.read_bytes(), name + ': HM encoder/decoder mismatch'
        wrap(stream, heif, width, height, depth)
        samples = expected if depth == 8 else struct.unpack('<' + 'H' * (len(expected) // 2), expected)
        hash_value = 0xcbf29ce484222325
        for value in samples:
            hash_value = ((hash_value ^ value) * 0x100000001b3) & ((1 << 64) - 1)
        manifest.append(dict(
            name=name, width=width, height=height, depth=depth, qp=qp, extra=extra,
            samples=len(samples), fnv=f'{hash_value:016x}',
            sha256=hashlib.sha256(expected).hexdigest(), encoder_command=command,
        ))
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hm', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    generate(args.hm.resolve(), args.out.resolve())
