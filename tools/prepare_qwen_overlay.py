"""Prepare an isolated Transformers 5.11.0 overlay; never edit site-packages."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True,
                        help='New directory; existing destinations are refused.')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / 'tools/qwen_upstream_sha256.json').read_text(encoding='utf-8'))
    version = importlib.metadata.version('transformers')
    if version != '5.11.0':
        raise SystemExit(f'Requires transformers==5.11.0 in a separate environment; found {version}')
    spec = importlib.util.find_spec('transformers')
    if not spec or not spec.origin:
        raise SystemExit('Cannot locate the installed Transformers package.')
    source = Path(spec.origin).resolve().parent
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit(f'Destination already exists; use a fresh directory: {destination}')
    if destination == source or source in destination.parents:
        raise SystemExit('Destination must be outside the installed Transformers package.')
    for name, digest in expected.items():
        path = source / 'models/qwen2_5_vl' / name
        # Git/checkouts may use CRLF; normalize only line endings for validation.
        actual = hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        if actual != digest:
            raise SystemExit(f'Unexpected upstream source: {name}. Refusing to patch it.')
    destination.mkdir(parents=True)
    shutil.copytree(source, destination / 'transformers',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in expected:
        shutil.copy2(root / 'qwen2_5_vl' / name,
                     destination / 'transformers/models/qwen2_5_vl' / name)
    manifest = {'transformers_version': version, 'base_source': str(source),
                'patched_files': sorted(expected), 'base_hashes': expected,
                'status': 'experimental overlay, not benchmark-validated'}
    (destination / 'overlay_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Created isolated overlay: {destination}')
    print('Set PYTHONPATH to this directory and use a fresh Python process.')
    print('Original installation unchanged. See docs/REPRODUCIBILITY.md before inference.')


if __name__ == '__main__':
    main()
