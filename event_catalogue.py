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
    for name, year in ICONIC_STORMS:
        sub = df_clean[(df_clean["NAME_U"] == name.upper()) & (df_clean["SEASON"] == year)]
        if len(sub) == 0:
            continue
        rows.append({"name": name, "year": year, "peak_kt": float(sub["USA_WIND"].max())})
    return pd.DataFrame(rows).sort_values("peak_kt", ascending=False).reset_index(drop=True)


def sample_catalogue(samples: dict, years: int, seed: int) -> np.ndarray:
    """Posterior predictive: cycle through posterior draws to generate `years` synthetic maxima."""
    rng = np.random.default_rng(seed)
    n_post = len(samples["mu"])
    idx = rng.integers(0, n_post, size=years)
    mu = samples["mu"][idx]
    sigma = samples["sigma"][idx]
    xi = samples["xi"][idx]
    # scipy uses c = +xi (where xi here is EVT convention) — but our convention
    # matches scipy's c directly (we sampled in this convention); see bayesian_gev.
    # genextreme.rvs uses c such that pdf has (1 + c*z)^(-1/c-1). EVT xi = -c.
    # So we pass c = -xi to scipy.
    return genextreme.rvs(-xi, loc=mu, scale=sigma, random_state=rng)


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


def summarise_rp(rp_draws: np.ndarray) -> tuple[float, float, float]:
    finite = rp_draws[np.isfinite(rp_draws)]
    if len(finite) == 0:
        return float("inf"), float("inf"), float("inf")
    med = float(np.median(finite))
    lo = float(np.quantile(finite, 0.025))
    hi = float(np.quantile(finite, 0.975))
    return med, lo, hi


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
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def main() -> int:
    if not POSTERIOR_PATH.exists():
        print(f"missing {POSTERIOR_PATH} -- run: python bayesian_gev.py first")
        return 1

    samples_raw = np.load(POSTERIOR_PATH)
    samples = {k: samples_raw[k] for k in ("mu", "sigma", "xi")}
    print(f"loaded {len(samples['mu']):,} posterior samples")

    # 1. Synthetic catalogue
    cat = sample_catalogue(samples, CATALOGUE_YEARS, SEED)
    print(f"synthetic catalogue: n={len(cat):,}, min={cat.min():.1f}, max={cat.max():.1f}, "
          f"median={np.median(cat):.1f}")

    # 2. Iconic storm peaks
    df = clean(load_ibtracs(DATA_PATH))
    storms = extract_iconic_peaks(df)
    print(f"\niconic storms extracted from IBTrACS: n={len(storms)}")

    # 3. Return period per storm, posterior + empirical
    rows = []
    for _, s in storms.iterrows():
        rp_post = posterior_return_period(s["peak_kt"], samples)
        med, lo, hi = summarise_rp(rp_post)
        emp_count = int((cat >= s["peak_kt"]).sum())
        rp_empirical = CATALOGUE_YEARS / emp_count if emp_count > 0 else float("inf")
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
