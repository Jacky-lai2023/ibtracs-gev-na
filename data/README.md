# Data

Raw IBTrACS North Atlantic best-track CSV is **not committed** to this repo
(54 MB, exceeds GitHub's soft limit for sane diffs).

To fetch it:

```bash
python ../download.py
```

The script writes `data/raw/ibtracs.NA.list.v04r01.csv` and is idempotent.

## Source

- Dataset: **IBTrACS v04r01**, North Atlantic basin
- Provider: NOAA National Centers for Environmental Information (NCEI)
- URL: `https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/`
- Citation: Knapp, K. R., et al. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society*, 91(3), 363–376.

## Field used

Only `USA_WIND` (NHC-calibrated 1-minute sustained maximum wind, in knots) is used.
This is the standard field for North Atlantic best-track analysis and avoids the
1-min vs 10-min averaging mismatch between agencies.
