#!/usr/bin/env python3
"""Generate RGB conversion oracles from HM planar output using scalar float64.

Usage: python3 scripts/generate_rgb_fixtures.py /path/to/HM /tmp/rgb-fixtures
The reference converter is independent of the Rust converter and uses no
third-party decoder's colour conversion implementation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from generate_422_fixtures import wrap


def convert(raw, matrix, full_range, bits):
    width, height, cw, ch = 58, 54, 29, 27
    values = struct.unpack('<' + 'H' * (len(raw) // 2), raw)
    luma = values[:width * height]
    cb = values[width * height:width * height + cw * ch]
    cr = values[width * height + cw * ch:]
    kr, kb = {6: (0.299, 0.114), 9: (0.2627, 0.0593)}[matrix]
    kg = 1 - kr - kb
    result = []
    for y in range(height):
        for x in range(width):
            # VUI location type 1 is centred in both directions.
            fx, fy = max(0, min(cw - 1, (x - .5) / 2)), max(0, min(ch - 1, (y - .5) / 2))
            x0, y0 = math.floor(fx), math.floor(fy)
            x1, y1 = min(cw - 1, x0 + 1), min(ch - 1, y0 + 1)
            sample = lambda plane: sum(plane[iy * cw + ix] * wx * wy for iy, wy in [(y0, 1 - (fy - y0)), (y1, fy - y0)] for ix, wx in [(x0, 1 - (fx - x0)), (x1, fx - x0)])
            v = luma[y * width + x]
            u, w = sample(cb), sample(cr)
            if full_range:
                v, u, w = v / 1023, (u - 512) / 1023, (w - 512) / 1023
            else:
                v, u, w = (v - 64) / 876, (u - 512) / 896, (w - 512) / 896
            blue, red = v + 2 * (1 - kb) * u, v + 2 * (1 - kr) * w
            green = (v - kr * red - kb * blue) / kg
            result.extend(math.floor(max(0, min(1, value)) * ((1 << bits) - 1) + .5) for value in [red, green, blue])
    return bytes(result) if bits == 8 else struct.pack('<' + 'H' * len(result), *result)


def generate(hm, out):
    out.mkdir(parents=True, exist_ok=True)
    encoder = next(hm.glob('bin/**/TAppEncoder'))
    decoder = next(hm.glob('bin/**/TAppDecoder'))
    raw = bytearray()
    for c, w, h in [(0, 64, 64), (1, 32, 32), (2, 32, 32)]:
        for y in range(h):
            for x in range(w):
                raw += struct.pack('<H', ((x * 7 + y * 11 + ((x // 8 + y // 8) % 2) * 39 + c * 47) % 256) * 4)
    source = out / 'input.yuv'
    source.write_bytes(raw)
    stream = out / 'color.hevc'
    reference = out / 'color-hm.yuv'
    command = [str(encoder), '-c', str(hm / 'cfg/encoder_intra_main.cfg'),
               '--Profile=main-RExt', '--Level=3', '--InputChromaFormat=420', '--InputBitDepth=10', '--InternalBitDepth=10',
               '--SourceWidth=64', '--SourceHeight=64', '--FrameRate=1', '--FramesToBeEncoded=1', '--QP=27',
               '--ConformanceWindowMode=3', '--ConfWinLeft=2', '--ConfWinRight=4', '--ConfWinTop=4', '--ConfWinBottom=6',
               '--VuiParametersPresent=1', '--TargetBitrate=1000000', '--VideoSignalTypePresent=1', '--VideoFullRange=1', '--ColourDescriptionPresent=1',
               '--ColourPrimaries=9', '--TransferCharacteristics=16', '--MatrixCoefficients=9',
               '--ChromaLocInfoPresent=1', '--ChromaSampleLocTypeTopField=1', '--ChromaSampleLocTypeBottomField=1',
               f'--InputFile={source}', f'--BitstreamFile={stream}', f'--ReconFile={out / "recon.yuv"}']
    with (out / 'encode.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=log, check=True)
    with (out / 'decode.log').open('w') as log:
        subprocess.run([str(decoder), '-b', str(stream), '-o', str(reference)], cwd=out, stdout=log, stderr=log, check=True)
    rows = []
    for name, color in [('vui2020', None), ('nclx601', (1, 13, 6, False))]:
        wrap(stream, out / f'{name}.heic', 58, 54, 10, 1, color)
        effective = (9, 16, 9, True) if color is None else color
        entry = dict(name=name, width=58, height=54, color=list(effective), chroma_location=1, sha256={})
        for bits in [8, 16]:
            converted = convert(reference.read_bytes(), effective[2], effective[3], bits)
            (out / f'{name}-rgb{bits}.bin').write_bytes(converted)
            entry['sha256'][str(bits)] = hashlib.sha256(converted).hexdigest()
        rows.append(entry)
    (out / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('hm', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    generate(args.hm.resolve(), args.out.resolve())
