#!/usr/bin/env python3
"""Explicitly download the pinned official Audio8 snapshot for offline export."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audio8.artifact import REPO, REVISION
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    args = parser.parse_args()
    print(snapshot_download(REPO, revision=REVISION, local_dir=args.directory,
                            allow_patterns=['model.safetensors', 'codec.pth', 'config.json',
                                            'tokenizer.json', 'LICENSE', 'NOTICE']))


if __name__ == '__main__':
    main()
