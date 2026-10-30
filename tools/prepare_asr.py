#!/usr/bin/env python3
"""Explicitly download/export the pinned multilingual Whisper Base ASR bundle."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio8.asr import export_asr, REPO, REVISION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', help='Local official Whisper Base snapshot directory')
    parser.add_argument('output', help='New models/audio8_asr/whisper-base.safetensors')
    parser.add_argument('--download', action='store_true', help='Explicitly fetch the pinned ASR files into source')
    args = parser.parse_args()
    if args.download:
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REVISION, local_dir=args.source,
                          allow_patterns=['model.safetensors', 'config.json', 'generation_config.json',
                                          'tokenizer.json', 'preprocessor_config.json', 'README.md'])
    print(json.dumps(export_asr(args.source, args.output), indent=2))


if __name__ == '__main__':
    main()
