#!/usr/bin/env python3
"""Download, verify, and convert the pinned rBGR LUAD release."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
COMMIT = "95d8a6710950920ca0e2eb5001896b8d0c47d4df"
FILES = {
    "Y_LUAD.rds": (
        "TCGA_Data/Proteomics_Data_Y/Y_LUAD.rds",
        "dbd239cf184ca4f3c9a71cdb28329f19d641f780dc1af8d2a09eb711fcf5a58d",
    ),
    "X_LUAD.rds": (
        "TCGA_Data/mRNA_Immune_X/X_LUAD.rds",
        "b6521c7cc78993747af442a8be0a639b71db0446e48d2b9289e93504411455ee",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the pinned, participant-joined LUAD RPPA table."
    )
    parser.add_argument(
        "--source-repo",
        type=Path,
        help="optional local checkout of bayesrx/rBGR at the pinned commit",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data" / "derived" / "luad_rppa.csv",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def acquire(name: str, relative: str, expected: str, raw_dir: Path, source: Path | None) -> Path:
    destination = raw_dir / name
    if source is not None:
        origin = source / relative
        if not origin.exists():
            raise FileNotFoundError(f"Missing {origin}")
        shutil.copy2(origin, destination)
    elif not destination.exists() or sha256(destination) != expected:
        url = f"https://raw.githubusercontent.com/bayesrx/rBGR/{COMMIT}/{relative}"
        partial = destination.with_suffix(destination.suffix + ".part")
        try:
            urllib.request.urlretrieve(url, partial)
            partial.replace(destination)
        finally:
            if partial.exists():
                partial.unlink()
    observed = sha256(destination)
    if observed != expected:
        raise RuntimeError(
            f"Checksum mismatch for {name}: expected {expected}, observed {observed}"
        )
    return destination


def main() -> None:
    args = parse_args()
    raw_dir = ROOT / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    acquired = {
        name: acquire(name, relative, checksum, raw_dir, args.source_repo)
        for name, (relative, checksum) in FILES.items()
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "Rscript",
            str(ROOT / "convert_data.R"),
            str(acquired["Y_LUAD.rds"]),
            str(acquired["X_LUAD.rds"]),
            str(args.output),
        ],
        check=True,
    )
    print(f"Verified rBGR commit {COMMIT}")
    print(f"Derived CSV: {args.output}")


if __name__ == "__main__":
    main()
