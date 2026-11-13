"""Download the AI2D dataset (Kembhavi et al., ECCV 2016) into external/ai2d/ (git-ignored).

    python scripts/get_ai2d.py --accept-license

AI2D is distributed by the Allen Institute for AI under a NON-COMMERCIAL licence (the
license.txt inside the archive): academic / non-profit research and evaluation use only, no
redistribution to third parties and no public release of modified data. This script therefore
never copies AI2D content into the repository, and you must pass --accept-license.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "external" / "ai2d"
ZIP_URL = "https://ai2-public-datasets.s3.amazonaws.com/diagrams/ai2d-all.zip"
TEST_IDS_URL = "https://s3-us-east-2.amazonaws.com/prior-datasets/ai2d_test_ids.csv"


def fetch(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        print("already downloaded:", path.relative_to(ROOT))
        return
    print("downloading", url)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:  # noqa: S310 (fixed https URLs)
        total = int(r.headers.get("content-length") or 0)
        done = 0
        while chunk := r.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.0f} / {total / 1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accept-license", action="store_true", help="confirm you accept the AI2D non-commercial licence")
    args = ap.parse_args(argv)
    if not args.accept_license:
        print(__doc__)
        print("Re-run with --accept-license to download.")
        return 2
    DEST.mkdir(parents=True, exist_ok=True)
    zpath = DEST / "ai2d-all.zip"
    fetch(ZIP_URL, zpath)
    fetch(TEST_IDS_URL, DEST / "ai2d_test_ids.csv")
    if not (DEST / "ai2d" / "annotations").is_dir():
        print("extracting…")
        with zipfile.ZipFile(zpath) as z:
            for m in z.namelist():
                # safety: only extract the expected tree, no absolute paths / parent traversal
                if not m.startswith("ai2d/") or ".." in Path(m).parts:
                    continue
                z.extract(m, DEST)
    print((DEST / "ai2d" / "license.txt").read_text())
    n = len(list((DEST / "ai2d" / "images").glob("*.png")))
    print(f"AI2D ready in {DEST.relative_to(ROOT)}: {n} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())
