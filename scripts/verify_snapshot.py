"""Verify the byte identity of the frozen source and bundled data snapshot."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'docs/source-manifest.json').read_text(encoding='utf-8'))
failures = []
for name, expected in manifest['files'].items():
    path = root / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        failures.append(name)
if failures:
    raise SystemExit('Snapshot mismatch: ' + ', '.join(failures))
print(f"Verified {len(manifest['files'])} source/data files.")
