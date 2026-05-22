"""Fit GEV to IBTrACS North Atlantic annual-maximum 1-min sustained wind.

Pipeline: raw CSV -> filter (basin/season/wind validity) -> annual maxima
         -> genextreme MLE -> return levels -> non-parametric bootstrap CI.

Run:
    python download.py     # fetch raw CSV (idempotent)
    python fit_gev.py      # fit + print report + render plots
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import cramervonmises, genextreme, kstest, norm

# === Constants ===
DATA_PATH = Path(__file__).parent / "data" / "raw" / "ibtracs.NA.list.v04r01.csv"
FIG_DIR = Path(__file__).parent / "figures"
SEASON_MIN = 1980
SEASON_MAX = 2024
RETURN_PERIODS = (2, 5, 10, 25, 50, 100, 250, 500)
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


def clean(df: pd.DataFrame, season_min: int = SEASON_MIN, season_max: int = SEASON_MAX) -> pd.DataFrame:
    out = df[df["BASIN"] == "NA"].copy()
    out["SEASON"] = pd.to_numeric(out["SEASON"], errors="coerce")
    out["USA_WIND"] = pd.to_numeric(out["USA_WIND"], errors="coerce")
    out = out[(out["SEASON"] >= season_min) & (out["SEASON"] <= season_max)]
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
    # Unbiased probability-weighted moment (PWM) estimators (Landwehr, Matalas
    # & Wallis 1979, WRR 15(5):1055-1064). L-moments are linear combinations of
    # these PWMs (Hosking 1990, eqs 2.3-2.5).
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = ((i - 1) / (n - 1) * x).sum() / n
    b2 = ((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x).sum() / n
    lam1 = b0
    lam2 = 2 * b1 - b0
    lam3 = 6 * b2 - 6 * b1 + b0
    t3 = lam3 / lam2  # L-skewness

    # Closed-form GEV L-moments estimator (Hosking, Wallis & Wood 1985,
    # Technometrics 27(3):251-261, eq 12; refined in Hosking 1990 §3 / Hosking
    # & Wallis 1997 Appendix A.8). Hosking k convention: k_Hosking = -xi_EVT.
    c = 2 / (3 + t3) - log(2) / log(3)
    k = 7.8590 * c + 2.9554 * c * c
    xi = -k  # convert to EVT convention (xi<0 -> Weibull / bounded above)
    sigma = (lam2 * k) / ((1 - 2 ** (-k)) * gamma(1 + k))
    # Location parameter mu = lam1 - sigma * (1 - Gamma(1+k)) / k.
    # (Earlier draft had the sign flipped on the gamma term; corrected against
    # the lmoments3 reference implementation and verified by E[X] = lam1.)
    mu = lam1 - sigma * (1 - gamma(1 + k)) / k
    # _scipy_c stores k = -xi for direct use with scipy.stats.genextreme,
    # which uses the c = -xi_EVT parameterisation.
    return {"xi": xi, "mu": float(mu), "sigma": float(sigma), "_scipy_c": k}


def return_level(params: dict, T: float) -> float:
    """T-year return level: quantile at probability 1 - 1/T."""
    return float(genextreme.ppf(1 - 1 / T, params["_scipy_c"], params["mu"], params["sigma"]))


_FIT_ERRORS = (RuntimeError, ValueError, FloatingPointError, OverflowError, np.linalg.LinAlgError)


def bootstrap(am: pd.Series, B: int, periods: tuple[int, ...], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(am)
    vals = am.values
    xis = np.empty(B)
    rls = {T: np.empty(B) for T in periods}
    n_failed = 0
    for b in range(B):
        resample = rng.choice(vals, size=n, replace=True)
        try:
            p = fit_gev(pd.Series(resample))
        except _FIT_ERRORS:
            xis[b] = np.nan
            for T in periods:
                rls[T][b] = np.nan
            n_failed += 1
            continue
        xis[b] = p["xi"]
        for T in periods:
            rls[T][b] = return_level(p, T)
    failure_rate = n_failed / B
    if failure_rate > 0.01:
        warnings.warn(
            f"bootstrap: {n_failed}/{B} ({failure_rate:.1%}) MLE fits failed; "
            f"surviving samples may be biased toward well-behaved resamples",
            RuntimeWarning, stacklevel=2,
        )
    return {"xi": xis, "_n_failed": n_failed, "_B": B,
            **{f"rl_{T}": rls[T] for T in periods}}


def ci(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    arr = arr[~np.isnan(arr)]
    return float(np.quantile(arr, alpha / 2)), float(np.quantile(arr, 1 - alpha / 2))


def mann_kendall_trend(values: np.ndarray) -> dict:
    """Mann-Kendall non-parametric trend test on a 1-D series.

    H0: no monotonic trend. Reject for p < alpha. Variance includes the
    tie correction from Hipel & McLeod (1994), which matters when data
    are quantized (e.g. USA_WIND reported in 5-kt increments).

    Returns dict(tau, S, z, p_value, n_tied_groups).
    """
    from collections import Counter
    n = len(values)
    i_idx, j_idx = np.triu_indices(n, k=1)
    s = int(np.sign(values[j_idx] - values[i_idx]).sum())
    # Tie correction: subtract sum_groups t(t-1)(2t+5) for each tie group of size t>1
    ties = Counter(values)
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in ties.values() if t > 1)
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = float(2 * norm.sf(abs(z)))  # sf avoids underflow for large |z|
    tau = s / (n * (n - 1) / 2)
    return {"tau": float(tau), "S": float(s), "z": float(z), "p_value": p,
            "n_tied_groups": sum(1 for t in ties.values() if t > 1)}


def goodness_of_fit(am: pd.Series, params: dict, B_null: int = 500, seed: int = BOOTSTRAP_SEED) -> dict:
    """KS + Lilliefors-corrected KS + Cramer-von Mises tests for GEV adequacy.

    Raw KS / CvM p-values from scipy use the known-parameter null distribution;
    when parameters are estimated from the same data, that null is wrong (too
    optimistic). Both p-values are corrected via the same parametric-bootstrap
    procedure: simulate from the fitted GEV, refit, compute the statistic, and
    use the empirical null as the reference distribution.
    """
    x = am.values
    c = params["_scipy_c"]
    loc = params["mu"]
    scale = params["sigma"]
    ks_stat, ks_p_raw = kstest(x, "genextreme", args=(c, loc, scale))
    cvm = cramervonmises(x, "genextreme", args=(c, loc, scale))
    rng = np.random.default_rng(seed)
    null_ks = np.full(B_null, np.nan)
    null_cvm = np.full(B_null, np.nan)
    for b in range(B_null):
        try:
            sim = genextreme.rvs(c, loc=loc, scale=scale, size=len(x), random_state=rng)
            c_s, loc_s, scale_s = genextreme.fit(sim)
            null_ks[b], _ = kstest(sim, "genextreme", args=(c_s, loc_s, scale_s))
            null_cvm[b] = cramervonmises(sim, "genextreme", args=(c_s, loc_s, scale_s)).statistic
        except _FIT_ERRORS:
            pass  # leave as nan; excluded below
    ks_valid = ~np.isnan(null_ks)
    cvm_valid = ~np.isnan(null_cvm)
    ks_p_lilli = float((null_ks[ks_valid] >= ks_stat).mean()) if ks_valid.any() else float("nan")
    cvm_p_boot = float((null_cvm[cvm_valid] >= cvm.statistic).mean()) if cvm_valid.any() else float("nan")
    return {
        "ks_stat": float(ks_stat),
        "ks_p_raw": float(ks_p_raw),
        "ks_p_lilliefors": ks_p_lilli,
        "cvm_stat": float(cvm.statistic),
        "cvm_p_raw": float(cvm.pvalue),
        "cvm_p_bootstrap": cvm_p_boot,
    }


def cutoff_sensitivity(df_raw: pd.DataFrame, cutoffs: tuple = (1970, 1980, 1990, 2000)) -> list:
    """Refit GEV with different SEASON cutoffs to test stability of the 1980 choice.

    Expects df_raw post-`load_ibtracs` (before `clean`); reapplies `clean()` per
    cutoff so filtering stays consistent with the main pipeline.
    """
    rows = []
    for start in cutoffs:
        sub = clean(df_raw, season_min=start, season_max=SEASON_MAX)
        am = sub.groupby("SEASON")["USA_WIND"].max()
        if len(am) < 10:
            continue
        c, loc, scale = genextreme.fit(am.values)
        rl100 = float(genextreme.ppf(0.99, c, loc, scale))
        rows.append({"cutoff": start, "n": int(len(am)), "xi": float(-c),
                     "mu": float(loc), "sigma": float(scale), "rl_100": rl100})
    return rows


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
    # Surface bootstrap failure rate inline (not just stderr warning).
    n_failed = boot.get("_n_failed", 0)
    B_total = boot.get("_B", len(boot["xi"]))
    if n_failed > 0:
        lines.append(f"  bootstrap failures: {n_failed}/{B_total} ({n_failed/B_total:.2%})")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> int:
    if not DATA_PATH.exists():
        print(f"missing {DATA_PATH} — run: python download.py")
        return 1
    FIG_DIR.mkdir(exist_ok=True)

    df_raw = load_ibtracs(DATA_PATH)
    print(f"raw rows: {len(df_raw):,}")
    df = clean(df_raw)
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
    print("L-moments (Hosking 1990) — small-sample-robust sanity check:")
    print(f"  xi    = {params_lm['xi']:+.4f}")
    print(f"  mu    = {params_lm['mu']:.2f} kt")
    print(f"  sigma = {params_lm['sigma']:.2f} kt")
    print(f"  RL_100 = {return_level(params_lm, 100):.1f} kt")
    print()
    print("Mann-Kendall trend test on annual maxima (H0: stationary):")
    mk = mann_kendall_trend(am.values)
    print(f"  Kendall tau = {mk['tau']:+.4f}, z = {mk['z']:+.3f}, p = {mk['p_value']:.4f}")
    verdict = "REJECT stationarity (trend significant)" if mk["p_value"] < 0.05 else "fail to reject (stationary plausible)"
    print(f"  → {verdict} at alpha=0.05")
    print()
    print("Goodness-of-fit (three tests; H0: data drawn from fitted GEV):")
    gof = goodness_of_fit(am, params)
    print(f"  Kolmogorov-Smirnov (raw, optimistic):  D = {gof['ks_stat']:.4f}, p = {gof['ks_p_raw']:.4f}")
    print(f"  Lilliefors-corrected KS (B=500):                              p = {gof['ks_p_lilliefors']:.4f}")
    print(f"  Cramer-von Mises (raw, optimistic):    W = {gof['cvm_stat']:.4f}, p = {gof['cvm_p_raw']:.4f}")
    print(f"  Cramer-von Mises (param. bootstrap):                          p = {gof['cvm_p_bootstrap']:.4f}")
    gof_pass = gof["ks_p_lilliefors"] > 0.05 and gof["cvm_p_bootstrap"] > 0.05
    print(f"  → GEV adequacy: {'supported' if gof_pass else 'REJECTED'} at alpha=0.05")
    print()
    print(f"Cutoff sensitivity (xi stability across SEASON start years):")
    print(f"  {'cutoff':>6} | {'n':>3} | {'xi':>9} | {'RL_100 (kt)':>12}")
    for r in cutoff_sensitivity(df_raw):
        marker = " ← current" if r["cutoff"] == SEASON_MIN else ""
        print(f"  {r['cutoff']:>6} | {r['n']:>3} | {r['xi']:>+9.4f} | {r['rl_100']:>12.2f}{marker}")

    plot_fit(am, params, FIG_DIR / "gev_fit.png")
    plot_return_level(am, params, boot, FIG_DIR / "return_level.png")
    print(f"\nfigures written to {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
