#!/usr/bin/env python3
"""Offline Audio8 single-file export. No Hub requests."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from audio8.artifact import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Local official snapshot or float single-file bundle')
    parser.add_argument('output')
    parser.add_argument('--profile', choices=['bf16', 'int8', 'int8-all'], default='int8')
    parser.add_argument('--codec-dtype', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    print(json.dumps(export(args.source, args.output, profile=args.profile,
                            codec_dtype=args.codec_dtype, dry_run=args.dry_run), indent=2))


if __name__ == '__main__':
    main()
