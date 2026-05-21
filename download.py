"""Download IBTrACS North Atlantic best-track CSV from NOAA NCEI.

Usage:
    python download.py

Idempotent — re-runs are no-ops if file already exists.
"""
from pathlib import Path
import sys
import requests

URL = (
    "https://www.ncei.noaa.gov/data/"
    "international-best-track-archive-for-climate-stewardship-ibtracs/"
    "v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
)
OUT = Path(__file__).parent / "data" / "raw" / "ibtracs.NA.list.v04r01.csv"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    # Probe Content-Length so we can compare against any existing file.
    head = requests.head(URL, timeout=30, allow_redirects=True)
    head.raise_for_status()
    expected = int(head.headers.get("Content-Length", 0))
    if OUT.exists() and expected > 0 and OUT.stat().st_size == expected:
        print(f"already present + size matches: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return 0
    if OUT.exists():
        actual = OUT.stat().st_size
        if expected > 0:
            print(f"existing file size {actual} != expected {expected}; re-downloading")
        else:
            print(f"could not verify size against server; re-downloading to be safe")
    print(f"downloading {URL}")
    tmp = OUT.with_suffix(OUT.suffix + ".part")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", expected))
        written = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {written / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="")
        print()
    # Atomic rename only after a complete, size-verified download.
    if total and tmp.stat().st_size != total:
        tmp.unlink()
        raise RuntimeError(
            f"incomplete download: got {tmp.stat().st_size} bytes, expected {total}"
        )
    tmp.replace(OUT)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
