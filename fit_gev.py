"""Fit GEV to IBTrACS North Atlantic annual-maximum 1-min sustained wind.

Pipeline: raw CSV -> filter (basin/season/wind validity) -> annual maxima
         -> genextreme MLE -> return levels -> non-parametric bootstrap CI.

Run:
    python download.py     # fetch raw CSV (idempotent)
    python fit_gev.py      # fit + print report + render plots
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import genextreme

# === Constants ===
DATA_PATH = Path(__file__).parent / "data" / "raw" / "ibtracs.NA.list.v04r01.csv"
FIG_DIR = Path(__file__).parent / "figures"
SEASON_MIN = 1980
SEASON_MAX = 2024
RETURN_PERIODS = (10, 25, 50, 100, 250, 500)
BOOTSTRAP_B = 1000
BOOTSTRAP_SEED = 20260521


def load_ibtracs(path: Path) -> pd.DataFrame:
    """Read IBTrACS CSV. Row 1 is a units row in v04r01 — skip it.

    keep_default_na=False is critical: pandas' default na_values includes the
    string "NA", which would silently null out every North Atlantic basin row.
    """
    return pd.read_csv(
        path,
        skiprows=[1],
        low_memory=False,
        keep_default_na=False,
        na_values=["", " "],
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["BASIN"] == "NA"].copy()
    out["SEASON"] = pd.to_numeric(out["SEASON"], errors="coerce")
    out["USA_WIND"] = pd.to_numeric(out["USA_WIND"], errors="coerce")
    out = out[(out["SEASON"] >= SEASON_MIN) & (out["SEASON"] <= SEASON_MAX)]
    out = out[out["USA_WIND"].notna() & (out["USA_WIND"] > 0)]
    return out


def annual_maxima(df: pd.DataFrame) -> pd.Series:
    am = df.groupby("SEASON")["USA_WIND"].max().sort_index()
    am.name = "annual_max_wind_kt"
    return am


def fit_gev(am: pd.Series) -> dict:
    """scipy parameterisation: pdf uses (c, loc, scale) with c = -xi."""
    c, loc, scale = genextreme.fit(am.values)
    return {"xi": -c, "mu": float(loc), "sigma": float(scale), "_scipy_c": c}


def fit_gev_lmoments(am: pd.Series) -> dict:
    """L-moments estimator (Hosking 1985, closed form).

    More stable than MLE for small samples (n < 50). Used as a sanity check
    on the MLE shape parameter, which is known to be unstable with small n.

    Reference: Hosking, J. R. M. (1990). L-moments: Analysis and Estimation
    of Distributions Using Linear Combinations of Order Statistics.
    Journal of the Royal Statistical Society B, 52(1), 105–124.
    """
    from math import gamma, log

    x = np.sort(am.values)
    n = len(x)
    # Sample L-moments via unbiased plotting-position estimators (Landwehr 1979)
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = ((i - 1) / (n - 1) * x).sum() / n
    b2 = ((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x).sum() / n
    lam1 = b0
    lam2 = 2 * b1 - b0
    lam3 = 6 * b2 - 6 * b1 + b0
    t3 = lam3 / lam2  # L-skewness

    # Hosking 1985 closed-form (valid for |t3| < 1)
    c = 2 / (3 + t3) - log(2) / log(3)
    k = 7.8590 * c + 2.9554 * c * c
    # In Hosking's k convention, ξ_EVT = -k (scipy uses c = +k)
    xi = -k
    sigma = (lam2 * k) / ((1 - 2 ** (-k)) * gamma(1 + k))
    mu = lam1 - sigma * (gamma(1 + k) - 1) / k
    return {"xi": xi, "mu": float(mu), "sigma": float(sigma), "_scipy_c": k}


def return_level(params: dict, T: float) -> float:
    """T-year return level: quantile at probability 1 - 1/T."""
    return float(genextreme.ppf(1 - 1 / T, params["_scipy_c"], params["mu"], params["sigma"]))


def bootstrap(am: pd.Series, B: int, periods: tuple[int, ...], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(am)
    vals = am.values
    xis = np.empty(B)
    rls = {T: np.empty(B) for T in periods}
    for b in range(B):
        resample = rng.choice(vals, size=n, replace=True)
        try:
            p = fit_gev(pd.Series(resample))
        except Exception:
            xis[b] = np.nan
            for T in periods:
                rls[T][b] = np.nan
            continue
        xis[b] = p["xi"]
        for T in periods:
            rls[T][b] = return_level(p, T)
    return {"xi": xis, **{f"rl_{T}": rls[T] for T in periods}}


def ci(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = arr[~np.isnan(arr)]
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def plot_fit(am: pd.Series, params: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.linspace(am.min() * 0.9, am.max() * 1.1, 400)
    pdf = genextreme.pdf(x, params["_scipy_c"], params["mu"], params["sigma"])
    ax.hist(am.values, bins=12, density=True, alpha=0.55, edgecolor="white",
            color="#5B8FB9", label=f"Annual maxima ({SEASON_MIN}–{SEASON_MAX}, n={len(am)})")
    ax.plot(x, pdf, color="#1F3A5F", lw=2.2,
            label=fr"GEV fit: $\xi$={params['xi']:.3f}, $\mu$={params['mu']:.1f}, $\sigma$={params['sigma']:.1f}")
    ax.set_xlabel("Annual maximum 1-min sustained wind (kt)")
    ax.set_ylabel("Density")
    ax.set_title("IBTrACS North Atlantic — GEV fit to annual maxima")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_return_level(am: pd.Series, params: dict, boot: dict, path: Path) -> None:
    import matplotlib.pyplot as plt
    Ts = np.logspace(np.log10(2), np.log10(500), 60)
    rl_point = [return_level(params, T) for T in Ts]
    # Bootstrap band per T (recompute on the fly using stored xis is expensive;
    # instead interpolate between the periods we stored)
    stored_T = sorted(int(k.split("_")[1]) for k in boot if k.startswith("rl_"))
    lo = [ci(boot[f"rl_{T}"])[0] for T in stored_T]
    hi = [ci(boot[f"rl_{T}"])[1] for T in stored_T]

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(Ts, rl_point, color="#1F3A5F", lw=2.2, label="MLE return level")
    ax.fill_between(stored_T, lo, hi, alpha=0.25, color="#5B8FB9",
                    label="Bootstrap 95% CI")
    # Empirical points (Weibull plotting positions)
    n = len(am)
    sorted_am = np.sort(am.values)
    ranks = np.arange(1, n + 1)
    emp_T = (n + 1) / (n + 1 - ranks)
    ax.scatter(emp_T, sorted_am, s=22, color="#C44536", zorder=5,
               label="Empirical (Weibull)")
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("1-min sustained wind (kt)")
    ax.set_title("Return level curve with bootstrap 95% CI")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def report(am: pd.Series, params: dict, boot: dict) -> str:
    xi_lo, xi_hi = ci(boot["xi"])
    lines = [
        "=" * 60,
        f"IBTrACS NA basin, annual-max 1-min sustained wind, {SEASON_MIN}–{SEASON_MAX}",
        f"n = {len(am)} seasons",
        "-" * 60,
        "GEV fit (MLE):",
        f"  xi    = {params['xi']:+.4f}   (95% CI [{xi_lo:+.3f}, {xi_hi:+.3f}])",
        f"  mu    = {params['mu']:.2f} kt",
        f"  sigma = {params['sigma']:.2f} kt",
        "-" * 60,
        "Return levels (point + bootstrap 95% CI):",
    ]
    for T in RETURN_PERIODS:
        rl = return_level(params, T)
        lo, hi = ci(boot[f"rl_{T}"])
        lines.append(f"  {T:4d}-yr : {rl:6.1f} kt  [{lo:5.1f}, {hi:5.1f}]")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    if not DATA_PATH.exists():
        print(f"missing {DATA_PATH} — run: python download.py")
        return 1
    FIG_DIR.mkdir(exist_ok=True)

    df = load_ibtracs(DATA_PATH)
    print(f"raw rows: {len(df):,}")
    df = clean(df)
    print(f"clean rows (NA, {SEASON_MIN}-{SEASON_MAX}, USA_WIND>0): {len(df):,}")
    am = annual_maxima(df)
    print(f"annual maxima ({len(am)} seasons): min={am.min():.0f} kt, "
          f"max={am.max():.0f} kt, median={am.median():.0f} kt")

    params = fit_gev(am)
    params_lm = fit_gev_lmoments(am)
    boot = bootstrap(am, B=BOOTSTRAP_B, periods=RETURN_PERIODS, seed=BOOTSTRAP_SEED)
    print()
    print(report(am, params, boot))
    print()
    print("L-moments (Hosking 1985) — small-sample-robust sanity check:")
    print(f"  xi    = {params_lm['xi']:+.4f}")
    print(f"  mu    = {params_lm['mu']:.2f} kt")
    print(f"  sigma = {params_lm['sigma']:.2f} kt")
    print(f"  RL_100 = {return_level(params_lm, 100):.1f} kt")

    plot_fit(am, params, FIG_DIR / "gev_fit.png")
    plot_return_level(am, params, boot, FIG_DIR / "return_level.png")
    print(f"\nfigures written to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
