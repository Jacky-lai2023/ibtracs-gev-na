# ibtracs-gev-na

> **Generalized Extreme Value fit to IBTrACS North Atlantic annual-maximum 1-minute sustained wind, with bootstrap uncertainty quantification.**

**Headline result (1980–2024, n = 45 seasons):**

| Return period | 1-min sustained wind (kt) | 95% bootstrap CI |
|---:|---:|---:|
| 10-yr  | 154 | [148, 165] |
| 50-yr  | 163 | [158, 175] |
| **100-yr** | **165** | **[159, 183]** |
| 500-yr | 168 | [160, 197] |

GEV parameters: ξ = −0.470, μ = 123.2 kt, σ = 22.2 kt. Shape parameter ξ < 0 implies a Weibull-type (bounded upper tail) regime — physically consistent with a thermodynamic ceiling on maximum sustained wind.

---

## Quickstart

```bash
# Python 3.14, managed via uv
uv sync                  # or: pip install -r requirements.txt
python download.py       # fetch IBTrACS NA basin CSV (~57 MB) from NOAA NCEI
python fit_gev.py        # fit + bootstrap + render plots + print report
```

Outputs land in `figures/` (`gev_fit.png`, `return_level.png`) plus a printed report in stdout.

## Usage

The whole pipeline is `fit_gev.py:main`. Each step is a pure function and can be reused:

```python
from fit_gev import load_ibtracs, clean, annual_maxima, fit_gev, return_level, bootstrap

df = load_ibtracs(Path("data/raw/ibtracs.NA.list.v04r01.csv"))
am = annual_maxima(clean(df))           # 45 floats, indexed by SEASON
params = fit_gev(am)                    # {"xi", "mu", "sigma"}
rl_100 = return_level(params, T=100)    # 165.0
boot = bootstrap(am, B=1000, periods=(100,), seed=20260521)
```

## Method

1. **Data**. IBTrACS v04r01, North Atlantic basin file. Field `USA_WIND` (NHC-calibrated 1-min sustained wind, knots) is used exclusively — this avoids the 1-min vs 10-min averaging mismatch across agency-reported fields. Filter to `BASIN == "NA"`, `1980 ≤ SEASON ≤ 2024`, `USA_WIND > 0`. The 1980 cutoff reflects the era of consistent geostationary-satellite TC monitoring; pre-1980 best-track intensity estimates have larger and time-varying observational uncertainty.

2. **Block maxima**. Annual maximum `USA_WIND` per season. n = 45.

3. **GEV fit**. MLE via `scipy.stats.genextreme.fit`. Note scipy uses `c = −ξ`.

4. **Return levels**. *T*-year return level = GEV(1 − 1/*T*) quantile.

5. **Uncertainty**. Non-parametric bootstrap (B = 1000, fixed seed) over the 45-vector of annual maxima. Each resample is independently MLE-fit; 95% CI is the empirical 2.5%/97.5% quantile of resampled ξ and return levels.

## Results

The plots are reproduced from `figures/`:

![GEV fit](figures/gev_fit.png)

![Return level curve](figures/return_level.png)

Empirical Weibull plotting positions ((n + 1) / (n + 1 − rank)) overlay the parametric curve closely across the observed range, supporting the GEV-family assumption.

## Limitations

- **Small-sample MLE**. With n = 45, the MLE estimate of ξ is unstable: bootstrap 95% CI is [−4.5, −0.1]. Return-level CIs are tighter (joint reparameterisation absorbs ξ noise into μ, σ), but a production system should use L-moments (Hosking 1990) or a Bayesian fit with an informative prior on ξ.
- **Stationarity assumption**. GEV(μ, σ, ξ) is assumed time-invariant. ENSO modulation and any anthropogenic trend are not modelled. A natural next step is a non-stationary GEV with year, ENSO index, or SST anomaly as covariate(s) on μ.
- **All NA tracks, not landfalls**. The annual maximum is taken over the entire North Atlantic basin, not over land or coastal observations only. Landfall-conditioned return levels would require additional filtering of the `LANDFALL` field and would change ξ.
- **No declustering**. Block maxima are taken per calendar season; no explicit treatment of dependence between sequential storms within a season.

## Source

Knapp, K. R., M. C. Kruk, D. H. Levinson, H. J. Diamond, and C. J. Neumann, 2010: The International Best Track Archive for Climate Stewardship (IBTrACS): Unifying tropical cyclone best track data. *Bulletin of the American Meteorological Society*, 91, 363–376. <https://doi.org/10.1175/2009BAMS2755.1>

## Repository layout

```
├── fit_gev.py            # main pipeline (one entrypoint, pure functions)
├── download.py           # idempotent IBTrACS fetcher
├── data/
│   ├── README.md
│   └── raw/              # 57 MB CSV, gitignored
├── figures/              # rendered plots, tracked
├── requirements.txt
├── pyproject.toml        # uv-managed
└── README.md
```
