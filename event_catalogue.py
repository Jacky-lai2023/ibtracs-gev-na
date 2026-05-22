"""Stochastic event catalogue from Bayesian GEV posterior + named-storm return periods.

Two outputs:
1. 10,000-year synthetic annual-maximum catalogue via posterior predictive sampling
   (one draw of (mu, sigma, xi) -> one GEV sample, repeated). This propagates
   parameter uncertainty into the catalogue itself -- closer to how production
   cat models marginalise over event-set parameters.
2. For 17 iconic NA basin storms 1980-2024 (Allen, Andrew, Katrina, Wilma, Irma,
   ...), the posterior distribution of empirical return periods. Each storm's
   per-year exceedance probability is computed analytically from each posterior
   sample; median and 95% credible interval are reported.

Prerequisite: run bayesian_gev.py first (produces posterior_samples.npz).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genextreme

from fit_gev import DATA_PATH, FIG_DIR, annual_maxima, clean, load_ibtracs

POSTERIOR_PATH = Path(__file__).parent / "posterior_samples.npz"
CATALOGUE_YEARS = 10000
SEED = 20260521

# Iconic NA basin TC events 1980-2024 — used for return-period narrative.
# (Name, Season) pairs disambiguate name reuse.
ICONIC_STORMS = [
    ("Allen", 1980), ("Gilbert", 1988), ("Andrew", 1992), ("Mitch", 1998),
    ("Isabel", 2003), ("Ivan", 2004), ("Katrina", 2005), ("Rita", 2005),
    ("Wilma", 2005), ("Dean", 2007), ("Irma", 2017), ("Harvey", 2017),
    ("Maria", 2017), ("Michael", 2018), ("Dorian", 2019), ("Ian", 2022),
    ("Sandy", 2012),
]


def extract_iconic_peaks(df_clean: pd.DataFrame) -> pd.DataFrame:
    df_clean = df_clean.copy()
    df_clean["NAME_U"] = df_clean["NAME"].astype(str).str.strip().str.upper()
    rows = []
    missing = []
    for name, year in ICONIC_STORMS:
        sub = df_clean[(df_clean["NAME_U"] == name.upper()) & (df_clean["SEASON"] == year)]
        if len(sub) == 0:
            missing.append(f"{name} {year}")
            continue
        rows.append({"name": name, "year": year, "peak_kt": float(sub["USA_WIND"].max())})
    if missing:
        warnings.warn(
            f"ICONIC_STORMS not found in IBTrACS: {', '.join(missing)}",
            RuntimeWarning, stacklevel=2,
        )
    if not rows:
        raise RuntimeError("no iconic storms matched IBTrACS — check NAME column format")
    return pd.DataFrame(rows).sort_values("peak_kt", ascending=False).reset_index(drop=True)


def sample_catalogue(samples: dict, years: int, seed: int) -> np.ndarray:
    """Posterior predictive: cycle through posterior draws to generate `years` synthetic maxima."""
    rng = np.random.default_rng(seed)
    n_post = len(samples["mu"])
    idx = rng.integers(0, n_post, size=years)
    mu = samples["mu"][idx]
    sigma = samples["sigma"][idx]
    xi = samples["xi"][idx]
    # Our posterior xi follows the EVT convention (xi<0 -> Weibull / bounded above).
    # scipy.stats.genextreme uses c = -xi_EVT, so pass -xi.
    cat = genextreme.rvs(-xi, loc=mu, scale=sigma, random_state=rng)
    if not np.isfinite(cat).all():
        n_bad = int((~np.isfinite(cat)).sum())
        warnings.warn(
            f"posterior-predictive catalogue has {n_bad} non-finite samples "
            f"(out of {years}); these will be dropped from downstream stats",
            RuntimeWarning, stacklevel=2,
        )
        cat = cat[np.isfinite(cat)]
    return cat


def posterior_return_period(intensity: float, samples: dict) -> np.ndarray:
    """Per posterior draw: 1 / P(annual_max >= intensity); inf where intensity exceeds support."""
    mu = samples["mu"]
    sigma = samples["sigma"]
    xi = samples["xi"]
    z = (intensity - mu) / sigma
    arg = 1.0 + xi * z
    valid = arg > 1e-9
    safe_arg = np.where(valid, arg, 1e-9)
    cdf = np.exp(-safe_arg ** (-1.0 / xi))
    cdf = np.where(valid, cdf, np.where(xi < 0, 1.0, 0.0))
    p_exceed = 1.0 - cdf
    return np.where(p_exceed > 1e-9, 1.0 / p_exceed, np.inf)


SAFFIR_SIMPSON_BANDS = [
    # (label, lower_edge, upper_edge)  -- half-open [lower, upper)
    # Edges are midpoints between adjacent NHC integer thresholds so that
    # continuous synthetic samples are assigned unambiguously while integer
    # observed kt values fall into the obviously-correct band.
    # NHC Saffir-Simpson integer thresholds (1-min sustained, kt):
    #   TS 34-63 / Cat 1 64-82 / Cat 2 83-95 / Cat 3 96-112 / Cat 4 113-136 / Cat 5 >=137
    ("TS 34-63", 33.5, 63.5),
    ("Cat 1 64-82", 63.5, 82.5),
    ("Cat 2 83-95", 82.5, 95.5),
    ("Cat 3 96-112", 95.5, 112.5),
    ("Cat 4 113-136", 112.5, 136.5),
    ("Cat 5 >=137", 136.5, np.inf),
]


def saffir_simpson_ppc(observed: np.ndarray, synthetic: np.ndarray) -> pd.DataFrame:
    """Posterior predictive check: observed vs synthetic distribution by Saffir-Simpson band.

    Strong agreement (< 2 percentage-point error per band) is evidence that the
    Bayesian GEV correctly reproduces the data-generating distribution at the
    discretisation most relevant to insurance applications.

    Bin edges sit at half-integer midpoints between adjacent NHC thresholds, so
    integer-kt observed values land in their natural band and continuous
    synthetic values are assigned without gaps or overlap.
    """
    rows = []
    n_obs = len(observed)
    n_syn = len(synthetic)
    for name, lo, hi in SAFFIR_SIMPSON_BANDS:
        obs_pct = 100 * ((observed >= lo) & (observed < hi)).sum() / n_obs
        syn_pct = 100 * ((synthetic >= lo) & (synthetic < hi)).sum() / n_syn
        rows.append({"band": name, "observed_%": obs_pct, "synthetic_%": syn_pct,
                     "abs_diff_pp": abs(obs_pct - syn_pct)})
    return pd.DataFrame(rows)


def summarise_rp(rp_draws: np.ndarray, storm_label: str | None = None,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Median + (1-alpha) CrI of return periods, treating posterior draws beyond
    the GEV upper support as RP=+inf rather than silently discarding them.

    Quantiles are computed on the joint (finite, +inf) distribution: if the
    target quantile `q` falls into the inf mass at the top, the answer is +inf;
    otherwise we rescale q into the finite-mass portion. Without this, the
    upper CrI is biased low whenever `inf_frac > alpha/2` — a real concern for
    storms near the posterior upper support (e.g. Allen 1980 at 165 kt).
    """
    finite = rp_draws[np.isfinite(rp_draws)]
    n_total = len(rp_draws)
    if len(finite) == 0:
        msg = "all posterior draws give infinite return period"
        if storm_label:
            msg = f"{storm_label}: {msg} (intensity beyond GEV support for every draw)"
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
        return float("inf"), float("inf"), float("inf")
    inf_frac = 1.0 - len(finite) / n_total
    if inf_frac > 0.5 and storm_label:
        warnings.warn(
            f"{storm_label}: {inf_frac:.1%} of posterior draws give infinite return period",
            RuntimeWarning, stacklevel=2,
        )

    def quantile_with_inf(q: float) -> float:
        # If the target quantile sits in the inf tail at the top of the joint
        # distribution, the answer is inf; otherwise rescale q into the finite
        # portion (which occupies the first (1 - inf_frac) of total mass).
        if q >= 1.0 - inf_frac:
            return float("inf")
        return float(np.quantile(finite, q / (1.0 - inf_frac)))

    return quantile_with_inf(0.5), quantile_with_inf(alpha / 2), quantile_with_inf(1.0 - alpha / 2)


def plot_catalogue(catalogue: np.ndarray, storms: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(catalogue, bins=60, density=True, color="#5B8FB9", alpha=0.65,
            edgecolor="white", label=f"Posterior predictive catalogue\n({len(catalogue):,} synthetic years)")

    # Show one line per *distinct* peak intensity, listing co-incident storms.
    # Top 4 intensity levels span 150 - 165 kt.
    grouped = storms.groupby("peak_kt", sort=False).agg(
        names=("name", lambda s: " / ".join(f"{n} {y}" for n, y in zip(s, storms.loc[s.index, "year"])))
    ).sort_index(ascending=False)
    palette = ["#1F3A5F", "#C44536", "#7A4E2D", "#3F6E3B", "#8B5E83", "#B07A2A"]
    ymax = ax.get_ylim()[1]
    for i, (peak_kt, names_row) in enumerate(grouped.head(4).iterrows()):
        col = palette[i % len(palette)]
        ax.axvline(peak_kt, color=col, lw=1.6, ls="--", alpha=0.85)
        ax.text(peak_kt + 1.0, ymax * (0.95 - 0.10 * i),
                f"{int(peak_kt)} kt: {names_row['names']}",
                color=col, fontsize=8.5, va="top")

    ax.set_xlabel("Annual maximum 1-min sustained wind (kt)")
    ax.set_ylabel("Density")
    ax.set_title("10,000-year synthetic catalogue — posterior predictive — top historic storms overlaid")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    if not POSTERIOR_PATH.exists():
        print(f"missing {POSTERIOR_PATH} -- run: python bayesian_gev.py first")
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # allow_pickle=False: posterior_samples.npz contains only numeric arrays;
    # disabling the unsafe deserialiser is defensive against tampered files.
    _ALLOW_PICKLE = False  # noqa: redefined here for explicit intent
    with np.load(POSTERIOR_PATH, allow_pickle=_ALLOW_PICKLE) as samples_raw:
        samples = {k: np.asarray(samples_raw[k]) for k in ("mu", "sigma", "xi")}
        rhats = {k: float(samples_raw[f"_rhat_{k}"]) for k in ("mu", "sigma", "xi")}
    print(f"loaded {len(samples['mu']):,} posterior samples "
          f"(R-hat: mu={rhats['mu']:.3f}, sigma={rhats['sigma']:.3f}, xi={rhats['xi']:.3f})")
    if max(rhats.values()) > 1.01:
        warnings.warn(
            f"posterior_samples.npz reports R-hat > 1.01: {rhats}. "
            f"Catalogue is being built on possibly-unmixed chains.",
            RuntimeWarning, stacklevel=2,
        )

    # 1. Synthetic catalogue
    cat = sample_catalogue(samples, CATALOGUE_YEARS, SEED)
    print(f"synthetic catalogue: n={len(cat):,}, min={cat.min():.1f}, max={cat.max():.1f}, "
          f"median={np.median(cat):.1f}")

    # 2. Iconic storm peaks
    df = clean(load_ibtracs(DATA_PATH))
    storms = extract_iconic_peaks(df)
    print(f"\niconic storms extracted from IBTrACS: n={len(storms)}")

    # 3. Saffir-Simpson posterior predictive check
    am = annual_maxima(df).values
    ppc = saffir_simpson_ppc(am, cat)
    print(f"\nSaffir-Simpson posterior predictive check (observed n={len(am)} vs synthetic n={len(cat):,}):")
    print(f"  {'band':>14} | {'observed %':>10} | {'synthetic %':>11} | {'abs diff (pp)':>14}")
    for _, r in ppc.iterrows():
        print(f"  {r['band']:>14} | {r['observed_%']:>10.2f} | {r['synthetic_%']:>11.2f} | {r['abs_diff_pp']:>14.2f}")
    max_dev = ppc["abs_diff_pp"].max()
    verdict = "EXCELLENT (<2 pp)" if max_dev < 2 else "ACCEPTABLE (<3 pp)" if max_dev < 3 else "REVIEW NEEDED"
    print(f"  → Max |observed - synthetic| across bands: {max_dev:.2f} pp  ({verdict})")

    # 4. Return period per storm, posterior + empirical
    rows = []
    for _, s in storms.iterrows():
        rp_post = posterior_return_period(s["peak_kt"], samples)
        med, lo, hi = summarise_rp(rp_post, storm_label=f"{s['name']} ({int(s['year'])})")
        emp_count = int((cat >= s["peak_kt"]).sum())
        # Use len(cat), not CATALOGUE_YEARS — if sample_catalogue dropped any
        # non-finite samples, len(cat) is the true denominator.
        rp_empirical = len(cat) / emp_count if emp_count > 0 else float("inf")
        rows.append({
            "Storm": f"{s['name']} ({int(s['year'])})",
            "Peak USA_WIND (kt)": int(s["peak_kt"]),
            "RP median (yr)": f"{med:.0f}",
            "RP 95% CrI (yr)": f"[{lo:.0f}, {hi:.0f}]",
            "Synthetic empirical RP (yr)": f"{rp_empirical:.0f}",
        })
    table = pd.DataFrame(rows)

    print()
    print("=" * 72)
    print("Historic NA basin storms 1980–2024: posterior return-period attribution")
    print("=" * 72)
    print(table.to_string(index=False))
    print("=" * 72)

    # Save table as markdown for README inclusion
    md_path = Path(__file__).parent / "figures" / "named_storms.md"
    with md_path.open("w") as f:
        f.write("| Storm | Peak USA_WIND (kt) | RP median (yr) | RP 95% CrI (yr) | Synthetic empirical RP (yr) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['Storm']} | {r['Peak USA_WIND (kt)']} | "
                    f"{r['RP median (yr)']} | {r['RP 95% CrI (yr)']} | "
                    f"{r['Synthetic empirical RP (yr)']} |\n")
    print(f"\nmarkdown table written to {md_path}")

    plot_catalogue(cat, storms, FIG_DIR / "catalogue_overlay.png")
    print(f"figure written to {FIG_DIR}/catalogue_overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
