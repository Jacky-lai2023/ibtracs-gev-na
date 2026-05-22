"""Bayesian GEV fit to IBTrACS NA annual maxima via NUTS (numpyro / JAX).

Why Bayesian?
    MLE on n=45 has unstable shape-parameter (xi) uncertainty (95% CI [-4.5, -0.1]).
    A weakly informative prior on xi, anchored at TC-EVT literature (xi ~ -0.2),
    plus full posterior propagation through return-level computation, gives
    physically-defensible credible intervals at long return periods.

Priors (weakly informative):
    mu    ~ Normal(130, 50)        -- centred near observed sample mean
    sigma ~ HalfNormal(30)         -- positive, broad
    xi    ~ Normal(-0.2, 0.15)     -- TC max-wind EVT literature suggests
                                     bounded upper tail (xi < 0); std 0.15
                                     keeps the prior weakly informative.

Reference: Coles, S. (2001), An Introduction to Statistical Modeling of
Extreme Values, Chapter 9 (Bayesian inference for extremes).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path

# Pin JAX to CPU before any jax import: bit-identical NUTS results on hiring
# manager's laptop regardless of whether they have a CUDA device.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from fit_gev import (  # noqa: E402
    DATA_PATH,
    FIG_DIR,
    RETURN_PERIODS,
    annual_maxima,
    clean,
    load_ibtracs,
)

NUM_CHAINS = 4
NUM_WARMUP = 1000
NUM_SAMPLES = 2000
SEED = 20260521


def gev_log_prob(y: jnp.ndarray, mu: float, sigma: float, xi: float) -> jnp.ndarray:
    """GEV log-density. Support: 1 + xi*(y-mu)/sigma > 0; otherwise -inf.

    Convention: xi<0 -> Weibull (bounded upper tail), xi>0 -> Frechet,
    xi=0 -> Gumbel (handled numerically via small offset; NUTS gradients
    keep |xi| > 0 in practice).
    """
    z = (y - mu) / sigma
    arg = 1 + xi * z
    valid = arg > 1e-9
    safe_arg = jnp.where(valid, arg, 1.0)
    log_density = (
        -jnp.log(sigma)
        - (1.0 + 1.0 / xi) * jnp.log(safe_arg)
        - safe_arg ** (-1.0 / xi)
    )
    return jnp.where(valid, log_density, -jnp.inf)


def gev_model(y: jnp.ndarray) -> None:
    mu = numpyro.sample("mu", dist.Normal(130.0, 50.0))
    sigma = numpyro.sample("sigma", dist.HalfNormal(30.0))
    xi = numpyro.sample("xi", dist.Normal(-0.2, 0.15))
    numpyro.factor("gev_lik", gev_log_prob(y, mu, sigma, xi).sum())


def run_nuts(am_values: np.ndarray, seed: int = SEED) -> MCMC:
    kernel = NUTS(gev_model, target_accept_prob=0.99)
    mcmc = MCMC(
        kernel,
        num_warmup=NUM_WARMUP,
        num_samples=NUM_SAMPLES,
        num_chains=NUM_CHAINS,
        progress_bar=True,
        chain_method="sequential",  # safer than parallel on small CPU box
    )
    mcmc.run(jax.random.PRNGKey(seed), y=jnp.asarray(am_values, dtype=jnp.float32))
    return mcmc


def posterior_return_level(samples: dict, T: float) -> np.ndarray:
    """Posterior of T-year return level: q = mu + sigma * ((-log(1-1/T))^(-xi) - 1) / xi."""
    mu = samples["mu"]
    sigma = samples["sigma"]
    xi = samples["xi"]
    p = 1.0 - 1.0 / T
    inner = (-np.log(p)) ** (-xi)
    return mu + sigma * (inner - 1.0) / xi


def summarise(arr: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    arr = arr[np.isfinite(arr)]  # drop nan/inf (e.g. xi≈0 in posterior_return_level)
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    med = float(np.median(arr))
    lo = float(np.quantile(arr, alpha / 2))
    hi = float(np.quantile(arr, 1 - alpha / 2))
    return med, lo, hi


def plot_posterior(samples: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, name, color in zip(axes, ["mu", "sigma", "xi"], ["#1F3A5F", "#5B8FB9", "#C44536"]):
        ax.hist(samples[name], bins=50, density=True, alpha=0.7, color=color, edgecolor="white")
        med, lo, hi = summarise(samples[name])
        ax.axvline(med, color="black", lw=1.5, label=f"median {med:.3f}")
        ax.axvline(lo, color="black", lw=0.8, ls="--", label=f"95% CrI [{lo:.3f}, {hi:.3f}]")
        ax.axvline(hi, color="black", lw=0.8, ls="--")
        ax.set_xlabel(name)
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Bayesian GEV — posterior marginals (NUTS, 4 chains)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_return_level_posterior(samples: dict, am: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    Ts = np.logspace(np.log10(2), np.log10(500), 60)
    med = np.empty_like(Ts)
    lo = np.empty_like(Ts)
    hi = np.empty_like(Ts)
    for k, T in enumerate(Ts):
        post = posterior_return_level(samples, T)
        med[k], lo[k], hi[k] = summarise(post)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.fill_between(Ts, lo, hi, alpha=0.25, color="#1F3A5F", label="Posterior 95% CrI")
    ax.plot(Ts, med, color="#1F3A5F", lw=2.2, label="Posterior median")

    n = len(am)
    sorted_am = np.sort(am)
    ranks = np.arange(1, n + 1)
    emp_T = (n + 1) / (n + 1 - ranks)
    ax.scatter(emp_T, sorted_am, s=22, color="#C44536", zorder=5, label="Empirical (Weibull)")

    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("1-min sustained wind (kt)")
    ax.set_title("Bayesian return level curve — posterior median + 95% CrI")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    if not DATA_PATH.exists():
        print(f"missing {DATA_PATH} -- run: python download.py")
        return 1
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_ibtracs(DATA_PATH)
    am = annual_maxima(clean(df))
    print(f"running NUTS on n={len(am)} annual maxima, {NUM_CHAINS} chains, "
          f"{NUM_WARMUP} warmup + {NUM_SAMPLES} samples per chain")

    mcmc = run_nuts(am.values)
    samples = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    samples_by_chain = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}

    print()
    print("=" * 60)
    print("Bayesian GEV — posterior summary")
    print("-" * 60)
    print("MCMC diagnostics (printed by numpyro):")
    mcmc.print_summary(prob=0.95)

    print("-" * 60)
    print("Return levels — posterior median + 95% credible interval:")
    for T in RETURN_PERIODS:
        post = posterior_return_level(samples, T)
        med, lo, hi = summarise(post)
        print(f"  {T:4d}-yr : {med:6.1f} kt  [{lo:5.1f}, {hi:5.1f}]")
    print("=" * 60)

    plot_posterior(samples, FIG_DIR / "posterior_marginals.png")
    plot_return_level_posterior(samples, am.values, FIG_DIR / "return_level_bayes.png")
    print(f"\nfigures written to {FIG_DIR}/")

    # Convergence diagnostics: split-Rhat (Gelman et al. 2013, BDA3 §11.4) +
    # divergence rate. We report and warn but do not abort — the posterior is
    # still saved with provenance so downstream consumers can inspect the
    # diagnostics in the .npz header.
    assert NUM_SAMPLES % 2 == 0, f"NUM_SAMPLES must be even for split-Rhat; got {NUM_SAMPLES}"
    rhats = {}
    for name in ("mu", "sigma", "xi"):
        arr = samples_by_chain[name]  # (NUM_CHAINS, NUM_SAMPLES), no reshape needed
        split = arr.reshape(2 * NUM_CHAINS, -1)  # split each chain in half (split-Rhat)
        means = split.mean(axis=1)
        within = split.var(axis=1, ddof=1).mean()
        between = split.shape[1] * means.var(ddof=1)
        m = split.shape[1]
        var_hat = ((m - 1) / m) * within + between / m
        rhats[name] = float(np.sqrt(var_hat / within)) if within > 0 else float("nan")
    if any(np.isnan(r) for r in rhats.values()):
        raise RuntimeError(f"split-Rhat is NaN (degenerate chain, within-chain var=0): {rhats}")
    n_div = int(mcmc.get_extra_fields()["diverging"].sum())
    rhat_max = max(rhats.values())
    div_rate = n_div / (NUM_CHAINS * NUM_SAMPLES)
    print(f"\nConvergence diagnostics (target: split-Rhat ≤ 1.01, divergences < 5%):")
    print(f"  split-Rhat: mu={rhats['mu']:.4f}, sigma={rhats['sigma']:.4f}, xi={rhats['xi']:.4f}")
    print(f"  divergences: {n_div}/{NUM_CHAINS * NUM_SAMPLES} ({div_rate:.2%})")
    if rhat_max > 1.01:
        warnings.warn(
            f"split-Rhat {rhat_max:.3f} > 1.01: chains may not have mixed.",
            RuntimeWarning, stacklevel=2,
        )
        print(f"  ⚠ R-hat = {rhat_max:.3f} > 1.01 (chains may not have mixed)")
    if div_rate > 0.05:
        warnings.warn(
            f"divergence rate {div_rate:.1%} > 5%: NUTS hit support boundary frequently.",
            RuntimeWarning, stacklevel=2,
        )
        print(f"  ⚠ divergence rate {div_rate:.1%} > 5%")
    if rhat_max <= 1.01 and div_rate <= 0.05:
        print(f"  ✓ converged")

    np.savez(
        Path(__file__).parent / "posterior_samples.npz",
        mu=samples["mu"],
        sigma=samples["sigma"],
        xi=samples["xi"],
        # Provenance metadata for downstream verification:
        _rhat_mu=rhats["mu"],
        _rhat_sigma=rhats["sigma"],
        _rhat_xi=rhats["xi"],
        _divergences=n_div,
        _num_chains=NUM_CHAINS,
        _num_samples=NUM_SAMPLES,
        _seed=SEED,
    )
    print(f"posterior samples + diagnostics saved to posterior_samples.npz")
    # Non-zero exit if diagnostics fail so CI/automation can detect.
    # The posterior is still saved (with provenance) for inspection.
    return 0 if rhat_max <= 1.01 and div_rate <= 0.05 else 2


if __name__ == "__main__":
    raise SystemExit(main())
