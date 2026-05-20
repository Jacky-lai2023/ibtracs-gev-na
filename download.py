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
    if OUT.exists() and OUT.stat().st_size > 10_000_000:
        print(f"already present: {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    with requests.get(URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with OUT.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 16):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {written / 1e6:6.1f} / {total / 1e6:.1f} MB ({pct:5.1f}%)", end="")
        print()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
