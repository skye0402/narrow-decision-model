"""Content manifests usable in a container without Git or ML dependencies."""
import hashlib
from pathlib import Path, PurePosixPath
import re

SOURCE_VERSION = "training-source-1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_file(root, name):
    root = Path(root).resolve()
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError("Manifest path must stay inside its artifact")
    path = root / name
    if not path.resolve().is_relative_to(root) or any(p.is_symlink() for p in [path, *path.parents] if p != root and p.is_relative_to(root)):
        raise ValueError("Artifact symlinks are not supported")
    if not path.is_file():
        raise ValueError("Missing artifact file")
    return path


def verify_files(root, hashes, *, required=(), exact=False, exclude=()):
    if not isinstance(hashes, dict) or not hashes or not set(required) <= set(hashes):
        raise ValueError("Incomplete file manifest")
    for name, expected in hashes.items():
        if not isinstance(name, str) or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("Invalid file manifest entry")
        if file_sha256(checked_file(root, name)) != expected:
            raise ValueError("Artifact content digest mismatch")
    if exact:
        actual = {str(path.relative_to(root)) for path in Path(root).rglob("*") if path.is_file() and str(path.relative_to(root)) not in exclude}
        if actual != set(hashes):
            raise ValueError("Unexpected files in artifact")


def verify_source(root, manifest):
    if manifest.get("version") != SOURCE_VERSION or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("source_commit", "")):
        raise ValueError("Unsupported source manifest")
    verify_files(root, manifest["files_sha256"], required=("prism_model/neural.py", "prism_model/training.py",
        "scripts/train_model.py", "requirements/training.txt", "uv.lock"))
    return manifest
