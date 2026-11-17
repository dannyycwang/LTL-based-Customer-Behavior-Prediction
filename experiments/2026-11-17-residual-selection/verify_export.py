from pathlib import Path
import hashlib, json
root = Path(__file__).resolve().parent
manifest = json.loads((root / "SHA256SUMS.json").read_text(encoding="utf-8"))
for name, expected in manifest.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
print(f"Verified {len(manifest)} files")
