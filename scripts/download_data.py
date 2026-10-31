"""Download the competition data from Kaggle into the project's raw data layer.

The script is idempotent: files that are already present with the expected size are
skipped, so it is safe to re-run. Every run writes a ``_manifest.json`` next to the
data (file list, sizes, SHA256 checksums, download timestamp) so you can always tell
exactly which data a model was trained on.

Usage:
    uv run python scripts/download_data.py            # download + extract
    uv run python scripts/download_data.py --check    # only verify credentials and access
    uv run python scripts/download_data.py --force    # re-download everything
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

# --- project-specific settings ------------------------------------------------------
COMPETITION = "ieee-fraud-detection"
INCLUDE = ["*.csv"]  # fnmatch patterns matched against competition file names
TARGET_DIR = Path(__file__).resolve().parents[1] / "data" / "01_raw"
# -------------------------------------------------------------------------------------

COMPETITION_URL = f"https://www.kaggle.com/competitions/{COMPETITION}"
MANIFEST = TARGET_DIR / "_manifest.json"
STAGING = TARGET_DIR / ".download"


class DownloadError(RuntimeError):
    pass


def kaggle_cli() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise DownloadError("Kaggle CLI not found. Run `uv sync` first (it is a dev dependency).")
    return exe


def check_credentials() -> str:
    """Return a description of the credential source, or raise with setup instructions."""
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN environment variable"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME / KAGGLE_KEY environment variables"
    for name in ("access_token", "kaggle.json"):
        if (config_dir / name).is_file():
            return str(config_dir / name)
    raise DownloadError(
        "No Kaggle credentials found. Create an API token at https://www.kaggle.com/settings "
        f"and save it to {config_dir / 'access_token'} (or set KAGGLE_API_TOKEN)."
    )


def run_kaggle(*args: str) -> str:
    result = subprocess.run([kaggle_cli(), *args], capture_output=True, text=True, check=False)
    output = result.stdout + result.stderr
    forbidden = "403 Client Error" in output or "Forbidden" in output
    unauthorized = "401 Client Error" in output or "Unauthorized" in output
    if result.returncode != 0 or forbidden or unauthorized:
        if forbidden:
            raise DownloadError(
                f"Access denied (403). Accept the competition rules first: {COMPETITION_URL}/rules"
            )
        if unauthorized:
            raise DownloadError("Unauthorized (401). Your Kaggle token is invalid or expired.")
        raise DownloadError(f"`kaggle {' '.join(args)}` failed:\n{output}")
    return result.stdout


def list_competition_files() -> list[tuple[str, int]]:
    """Return (name, size) for every competition file matching INCLUDE."""
    files: list[tuple[str, int]] = []
    page_token = ""
    while True:
        args = ["competitions", "files", COMPETITION, "--csv", "--page-size", "200"]
        if page_token:
            args += ["--page-token", page_token]
        lines = run_kaggle(*args).splitlines()
        page_token = ""
        for line in lines:
            if line.startswith("Next Page Token = "):
                page_token = line.split("=", 1)[1].strip()
        header = next(i for i, line in enumerate(lines) if line.startswith("name,size"))
        for row in csv.DictReader(lines[header:]):
            if any(fnmatch.fnmatch(row["name"], pattern) for pattern in INCLUDE):
                files.append((row["name"], int(row["size"])))
        if not page_token:
            return files


def download_file(name: str, size: int, force: bool) -> str:
    dest = TARGET_DIR / name
    if dest.is_file() and dest.stat().st_size == size and not force:
        return "skipped"

    STAGING.mkdir(parents=True, exist_ok=True)
    run_kaggle("competitions", "download", COMPETITION, "-f", name, "-p", str(STAGING), "-q", "-o")

    basename = Path(name).name
    plain, zipped = STAGING / basename, STAGING / f"{basename}.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if zipped.is_file():  # Kaggle compresses large files on the fly
        with (
            zipfile.ZipFile(zipped) as zf,
            zf.open(zf.namelist()[0]) as src,
            dest.open("wb") as out,
        ):
            shutil.copyfileobj(src, out)
        zipped.unlink()
    elif plain.is_file():
        shutil.move(plain, dest)
    else:
        raise DownloadError(f"Download of {name} produced no file in {STAGING}")

    if dest.stat().st_size != size:
        raise DownloadError(f"{name}: expected {size} bytes, got {dest.stat().st_size}")
    return "downloaded"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(files: list[tuple[str, int]]) -> None:
    manifest = {
        "competition": COMPETITION,
        "source": COMPETITION_URL,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [
            {"path": name, "size": size, "sha256": sha256(TARGET_DIR / name)}
            for name, size in files
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=f"Download Kaggle data for {COMPETITION}")
    parser.add_argument("--check", action="store_true", help="only verify credentials and access")
    parser.add_argument("--force", action="store_true", help="re-download files that exist")
    args = parser.parse_args()

    try:
        print(f"Credentials: {check_credentials()}")
        files = list_competition_files()
        total_gb = sum(size for _, size in files) / 1e9
        print(f"Competition '{COMPETITION}': {len(files)} files to fetch ({total_gb:.2f} GB)")
        if args.check:
            print("Access OK.")
            return 0

        for i, (name, size) in enumerate(files, 1):
            status = download_file(name, size, args.force)
            print(f"[{i:>3}/{len(files)}] {status:<10} {name} ({size / 1e6:,.1f} MB)")

        print("Computing checksums...")
        write_manifest(files)
        # Removed only now: on Windows the folder can stay locked briefly after the last download.
        shutil.rmtree(STAGING, ignore_errors=True)
        print(f"Done. Data in {TARGET_DIR}, manifest in {MANIFEST.name}")
        return 0
    except DownloadError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
