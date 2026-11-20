"""Build a portable source/results archive without virtualenv or Git metadata.

Run from the repository root after reviewing reports and running experiments.
This only writes local files. It does not publish or submit anything.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def included_files():
    roots = ["src", "tests", "reports", "results", "scripts", "instructions"]
    files = []
    for name in roots:
        files += [p for p in (ROOT / name).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    files += [ROOT / name for name in ["readme.md", "README_ASSIGNMENT.md", "requirements.txt", "requirements-lock.txt", "pyproject.toml", ".gitignore", ".gitattributes"]]
    return sorted(set(p for p in files if p.exists()))


def main():
    try:
        upstream = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        upstream = "1056d711e166747cc485725e31f2535967b030dd"
    manifest = ROOT / "results/source.json"
    hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in included_files() if p != manifest}
    provenance = {"upstream_repository": "https://github.com/InnoRL/IAEML-assignment",
                  "upstream_revision": "1056d711e166747cc485725e31f2535967b030dd",
                  "local_head_at_packaging": upstream,
                  "author": "Nikita Shankin", "date": "2026-11-20",
                  "personal_reflection_complete": not any("**Nikita:" in (ROOT / "reports" / name).read_text(encoding="utf-8") for name in ("assignment1.md", "assignment2.md")),
                  "published_or_submitted": False,
                  "sha256": hashes}
    manifest.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    output = ROOT / "output/IAEML_Nikita_Shankin.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in included_files():
            archive.write(path, path.relative_to(ROOT))
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any(".venv/" in n or ".git/" in n or "__pycache__" in n for n in names)
        print(json.dumps({"archive": str(output), "files": len(names), "bytes": output.stat().st_size,
                          "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
