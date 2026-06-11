"""
Stochastic Modeling of Sunset Afterglow Probability
===================================================
Monte Carlo simulation of atmospheric photon transport (Rayleigh scattering)
framed entirely as a probability problem:

    * Exponential distribution  -> photon free path (inverse-transform sampling)
    * Beer-Lambert law          -> conditional survival probability P(X > tau)
    * Law of Large Numbers      -> convergence of MC estimators
    * Central Limit Theorem     -> 1/sqrt(N) shrinkage of the estimator error
    * Random walk / Markov chain-> multiple-scattering collision decision tree
    * Weibull distribution      -> stochastic atmospheric turbidity
    * Joint distribution        -> correlated [P, T, H] -> afterglow threshold model

Running this module regenerates every figure in ./figures and the numeric
summary results.json that the report builder consumes.

Author: (course project)
"""

import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# --------------------------------------------------------------------------- #
#  Global configuration
# --------------------------------------------------------------------------- #
RNG = np.random.default_rng(271828)            # reproducible master stream
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})

# Reference Rayleigh optical depth of the whole atmosphere at the zenith.
# Literature value at 550 nm is ~0.097 (Bodhaine et al., 1999).
TAU_ZENITH_550 = 0.097
LAMBDA_REF = 550.0          # nm, reference wavelength
RED_NM = 700.0
BLUE_NM = 450.0

RESULTS = {}                # collected scalars -> results.json


# --------------------------------------------------------------------------- #
#  Physics  <->  probability translation
# --------------------------------------------------------------------------- #
def rayleigh_optical_depth(wavelength_nm, airmass):
    """Optical depth tau(lambda, m) for a slant path.

    Rayleigh cross-section scales as lambda^-4, so tau ~ lambda^-4.  ``airmass``
    is the relative slant path length (1 at zenith, ~38 at the horizon).
    tau is exactly the *rate x length* of the exponential free-path model, i.e.
    the expected number of scattering collisions along the path.
    """
    return airmass * TAU_ZENITH_550 * (LAMBDA_REF / wavelength_nm) ** 4


def sample_free_path(rate, size, rng):
    """Inverse-transform sampling of the Exponential distribution.

        X = -(1/rate) * ln(U),  U ~ Uniform(0,1)   ->   X ~ Exponential(rate)

    Here ``rate`` is the collision density lambda; the mean free path is 1/rate.
    """
    u = rng.random(size)
    return -np.log(u) / rate


# --------------------------------------------------------------------------- #
#  Figure 1 -- Exponential free path & inverse-transform validation
# --------------------------------------------------------------------------- #
def figure_exponential_validation():
    """Histogram of inverse-transform samples vs the analytic Exponential PDF."""
    n = 200_000
    # Two collision densities: dense (blue-like) and sparse (red-like) media.
    rates = {"Dense medium  (lambda = 2.0)": 2.0,
             "Sparse medium (lambda = 0.5)": 0.5}
    colors = {"Dense medium  (lambda = 2.0)": "#1f77b4",
              "Sparse medium (lambda = 0.5)": "#d62728"}

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ks_report = {}
    for label, rate in rates.items():
        x = sample_free_path(rate, n, RNG)
        ax.hist(x, bins=120, range=(0, 8), density=True, alpha=0.45,
                color=colors[label], label=f"{label} -- samples")
        grid = np.linspace(0, 8, 400)
        ax.plot(grid, rate * np.exp(-rate * grid), color=colors[label], lw=2.2)
        # empirical vs theoretical mean (should match 1/lambda by LLN)
        ks_report[rate] = (float(x.mean()), 1.0 / rate)

    ax.set_xlabel("free path  $x$  (units of mean free path at $\\lambda=1$)")
    ax.set_ylabel("probability density")
    ax.set_title("Fig. 1  Inverse-transform sampling reproduces the "
                 "Exponential PDF\n$X=-\\frac{1}{\\lambda}\\ln U$,  "
                 "$f(x)=\\lambda e^{-\\lambda x}$")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_exponential.png"))
    plt.close(fig)
    RESULTS["exp_mean_check"] = {str(k): v for k, v in ks_report.items()}


# --------------------------------------------------------------------------- #
#  Figure 2 -- Beer-Lambert survival: P(A | wavelength) = exp(-tau)
# --------------------------------------------------------------------------- #
def figure_beer_lambert():
    """Conditional survival probability of direct transmission for red vs blue.

    Event A = "photon traverses the slant path with NO collision".
    A occurs iff the first free path X exceeds the path length, and since the
    path length equals tau in mean-free-path units, P(A) = P(X > tau) = e^{-tau}.
    """
    airmasses = np.linspace(1, 40, 200)
    tau_red = rayleigh_optical_depth(RED_NM, airmasses)
    tau_blue = rayleigh_optical_depth(BLUE_NM, airmasses)
    surv_red = np.exp(-tau_red)
    surv_blue = np.exp(-tau_blue)

    # Monte-Carlo check at a handful of airmasses.
    n = 60_000
    mc_air = np.array([1, 5, 10, 20, 30, 38], dtype=float)
    mc_red, mc_blue = [], []
    for m in mc_air:
        tr = rayleigh_optical_depth(RED_NM, m)
        tb = rayleigh_optical_depth(BLUE_NM, m)
        xr = sample_free_path(1.0, n, RNG)     # unit-rate path in tau-units
        xb = sample_free_path(1.0, n, RNG)
        mc_red.append(np.mean(xr > tr))
        mc_blue.append(np.mean(xb > tb))

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(airmasses, surv_red, color="#d62728", lw=2,
            label="red 700 nm  (analytic $e^{-\\tau}$)")
    ax.plot(airmasses, surv_blue, color="#1f77b4", lw=2,
            label="blue 450 nm  (analytic $e^{-\\tau}$)")
    ax.scatter(mc_air, mc_red, color="#d62728", zorder=5, marker="o",
               edgecolor="k", s=45, label="red  Monte Carlo")
    ax.scatter(mc_air, mc_blue, color="#1f77b4", zorder=5, marker="s",
               edgecolor="k", s=45, label="blue Monte Carlo")
    ax.axvline(38, color="gray", ls="--", lw=1)
    ax.text(38, 0.9, " horizon\n airmass~38", fontsize=8, color="gray")
    ax.set_xlabel("airmass  $m$  (slant path length, 1 = zenith)")
    ax.set_ylabel("$P(\\mathrm{direct\\ transmission})$")
    ax.set_title("Fig. 2  Conditional survival probability $P(A\\mid\\lambda)"
                 "=e^{-\\tau(\\lambda,m)}$\nBlue light is extinguished far "
                 "faster than red -> the Sun reddens")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_beer_lambert.png"))
    plt.close(fig)

    # Numbers cited in the report (horizon airmass 38).
    m = 38.0
    RESULTS["horizon"] = {
        "airmass": m,
        "tau_red": float(rayleigh_optical_depth(RED_NM, m)),
        "tau_blue": float(rayleigh_optical_depth(BLUE_NM, m)),
        "surv_red": float(np.exp(-rayleigh_optical_depth(RED_NM, m))),
        "surv_blue": float(np.exp(-rayleigh_optical_depth(BLUE_NM, m))),
    }
    RESULTS["horizon"]["red_to_blue_ratio"] = (
        RESULTS["horizon"]["surv_red"] / RESULTS["horizon"]["surv_blue"])


# --------------------------------------------------------------------------- #
#  Figure 3 -- Law of Large Numbers & Central Limit Theorem
# --------------------------------------------------------------------------- #
def figure_lln():
    """Running MC estimator of P(A) converging to e^{-tau}, with CLT band."""
    m = 38.0
    tau_r = rayleigh_optical_depth(RED_NM, m)
    tau_b = rayleigh_optical_depth(BLUE_NM, m)
    p_r, p_b = np.exp(-tau_r), np.exp(-tau_b)

    n = 50_000
    xr = sample_free_path(1.0, n, RNG)
    xb = sample_free_path(1.0, n, RNG)
    hit_r = (xr > tau_r).astype(float)
    hit_b = (xb > tau_b).astype(float)
    k = np.arange(1, n + 1)
    run_r = np.cumsum(hit_r) / k
    run_b = np.cumsum(hit_b) / k

    fig = plt.figure(figsize=(7.4, 5.6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1.1], hspace=0.38)

    ax0 = fig.add_subplot(gs[0])
    ax0.axhline(p_r, color="#d62728", ls="--", lw=1.4,
                label=f"red limit $e^{{-\\tau}}$ = {p_r:.3f}")
    ax0.axhline(p_b, color="#1f77b4", ls="--", lw=1.4,
                label=f"blue limit $e^{{-\\tau}}$ = {p_b:.1e}")
    ax0.plot(k, run_r, color="#d62728", lw=1, alpha=0.9, label="red running mean")
    ax0.plot(k, run_b, color="#1f77b4", lw=1, alpha=0.9, label="blue running mean")
    # CLT +/- 2 sigma envelope for the red estimator
    se_r = np.sqrt(p_r * (1 - p_r) / k)
    ax0.fill_between(k, p_r - 2 * se_r, p_r + 2 * se_r, color="#d62728",
                     alpha=0.15, label="red  $\\pm2\\,$SE  (CLT)")
    ax0.set_xscale("log")
    ax0.set_xlabel("number of simulated photons  $N$")
    ax0.set_ylabel("estimated survival prob.")
    ax0.set_title("Fig. 3  Law of Large Numbers: MC estimators converge to "
                  "$e^{-\\tau}$")
    ax0.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")

    # Lower panel: |error| vs N on log-log with the 1/sqrt(N) reference line.
    ax1 = fig.add_subplot(gs[1])
    err_r = np.abs(run_r - p_r)
    ax1.plot(k, err_r, color="#d62728", lw=0.8, alpha=0.7, label="|error| (red)")
    ref = se_r * 1.0
    ax1.plot(k, ref, color="k", lw=1.6, ls="--",
             label="$\\sqrt{p(1-p)/N}\\ \\propto N^{-1/2}$ (CLT)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("number of simulated photons  $N$")
    ax1.set_ylabel("absolute error")
    ax1.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_lln.png"))
    plt.close(fig)

    RESULTS["lln"] = {
        "N": n,
        "red_final_estimate": float(run_r[-1]), "red_limit": float(p_r),
        "blue_final_estimate": float(run_b[-1]), "blue_limit": float(p_b),
        "red_final_abs_error": float(abs(run_r[-1] - p_r)),
        "blue_final_abs_error": float(abs(run_b[-1] - p_b)),
    }


# --------------------------------------------------------------------------- #
#  Figure 4 -- Multiple-scattering random walk (collision decision tree)
# --------------------------------------------------------------------------- #
def random_walk_slab(tau_total, n_photons, albedo, p_forward, rng):
    """1-D two-stream photon transport through a slab of optical thickness tau.

    Each photon is a random walk on the line [0, tau_total]:
      * step  s = -ln(U)          (Exponential free path, unit rate)
      * cross x >= tau  -> TRANSMITTED to the observer (success)
      * cross x <= 0    -> BACK-SCATTERED out to space (lost)
      * else collision: Bernoulli(albedo): absorbed (lost) with prob 1-albedo,
        otherwise scatter -- Bernoulli(p_forward) keeps/reverses direction.
    Returns counts of the three terminal outcomes.
    """
    x = np.zeros(n_photons)
    mu = np.ones(n_photons)                       # +1 toward observer
    alive = np.ones(n_photons, dtype=bool)
    transmitted = np.zeros(n_photons, dtype=bool)
    absorbed = np.zeros(n_photons, dtype=bool)
    backscattered = np.zeros(n_photons, dtype=bool)

    max_steps = 2000
    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.where(alive)[0]
        s = -np.log(rng.random(idx.size))         # exponential free path
        x[idx] += mu[idx] * s

        out_t = x[idx] >= tau_total               # reached observer
        out_b = x[idx] <= 0.0                      # escaped backward
        transmitted[idx[out_t]] = True
        backscattered[idx[out_b]] = True
        alive[idx[out_t | out_b]] = False

        # remaining photons collide inside the slab
        coll = idx[~(out_t | out_b)]
        if coll.size:
            scatter = rng.random(coll.size) < albedo
            absorbed[coll[~scatter]] = True
            alive[coll[~scatter]] = False
            sc = coll[scatter]
            # forward keeps direction, backward reverses it
            reverse = rng.random(sc.size) >= p_forward
            mu[sc[reverse]] *= -1.0
    return (int(transmitted.sum()), int(absorbed.sum()),
            int(backscattered.sum()), int(alive.sum()))


def figure_random_walk():
    """Outcome of the full collision decision tree for red vs blue photons."""
    m = 38.0
    n = 40_000
    albedo = 0.99            # single-scattering albedo (Rayleigh ~ no absorption)
    p_forward = 0.5          # Rayleigh phase fn ~ symmetric fwd/back in 1-D

    outcomes = {}
    for name, wl, col in [("red 700 nm", RED_NM, "#d62728"),
                          ("blue 450 nm", BLUE_NM, "#1f77b4")]:
        tau = rayleigh_optical_depth(wl, m)
        t, a, b, stuck = random_walk_slab(tau, n, albedo, p_forward, RNG)
        outcomes[name] = dict(tau=tau, transmitted=t, absorbed=a,
                              backscattered=b, stuck=stuck, color=col)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.4, 4.3),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    cats = ["transmitted", "back-scattered", "absorbed"]
    keys = ["transmitted", "backscattered", "absorbed"]
    width = 0.36
    xpos = np.arange(len(cats))
    for i, (name, d) in enumerate(outcomes.items()):
        vals = [d[k] / n for k in keys]
        axL.bar(xpos + (i - 0.5) * width, vals, width, color=d["color"],
                label=name, edgecolor="k", linewidth=0.5)
    axL.set_xticks(xpos)
    axL.set_xticklabels(cats)
    axL.set_ylabel("fraction of photons")
    axL.set_title("Fig. 4a  Terminal outcome of the random walk")
    axL.legend(frameon=False, fontsize=9)

    # Right panel: a few illustrative 1-D random-walk trajectories (blue).
    tau_b = outcomes["blue 450 nm"]["tau"]
    rng2 = np.random.default_rng(7)
    axR.axhline(0, color="k", lw=1)
    axR.axhline(tau_b, color="green", lw=1.5, ls="--")
    axR.text(0.5, tau_b * 1.01, "observer (transmitted)", color="green",
             fontsize=8, ha="left", va="bottom")
    axR.text(0.5, -0.3, "space (back-scattered)", color="gray", fontsize=8)
    for _ in range(12):
        x, mu, pos, t = 0.0, 1.0, [0.0], [0.0]
        for step in range(400):
            x += mu * (-np.log(rng2.random()))
            t.append(t[-1] + 1)
            x_clip = max(min(x, tau_b), 0.0)
            pos.append(x_clip)
            if x >= tau_b or x <= 0:
                break
            if rng2.random() >= albedo:
                break                      # absorbed
            if rng2.random() >= p_forward:
                mu *= -1
        axR.plot(t, pos, lw=0.9, alpha=0.8)
    axR.set_xlabel("collision step")
    axR.set_ylabel("optical depth into slab")
    axR.set_title("Fig. 4b  Sample blue-photon\nrandom walks")
    axR.set_ylim(-0.5, tau_b * 1.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_random_walk.png"))
    plt.close(fig)

    RESULTS["random_walk"] = {
        name: {k: d[k] for k in ("tau", "transmitted", "absorbed",
                                 "backscattered", "stuck")}
        for name, d in outcomes.items()}
    RESULTS["random_walk"]["n_photons"] = n
    RESULTS["random_walk"]["albedo"] = albedo


# --------------------------------------------------------------------------- #
#  CIE 1931 colour matching (for the spectral -> true colour optimisation)
# --------------------------------------------------------------------------- #
# CIE 1931 2-degree standard observer, 10 nm steps, 380..700 nm.
_CIE = np.array([
    [380, 0.0014, 0.0000, 0.0065], [390, 0.0042, 0.0001, 0.0201],
    [400, 0.0143, 0.0004, 0.0679], [410, 0.0435, 0.0012, 0.2074],
    [420, 0.1344, 0.0040, 0.6456], [430, 0.2839, 0.0116, 1.3856],
    [440, 0.3483, 0.0230, 1.7471], [450, 0.3362, 0.0380, 1.7721],
    [460, 0.2908, 0.0600, 1.6692], [470, 0.1954, 0.0910, 1.2876],
    [480, 0.0956, 0.1390, 0.8130], [490, 0.0320, 0.2080, 0.4652],
    [500, 0.0049, 0.3230, 0.2720], [510, 0.0093, 0.5030, 0.1582],
    [520, 0.0633, 0.7100, 0.0782], [530, 0.1655, 0.8620, 0.0422],
    [540, 0.2904, 0.9540, 0.0203], [550, 0.4334, 0.9950, 0.0087],
    [560, 0.5945, 0.9950, 0.0039], [570, 0.7621, 0.9520, 0.0021],
    [580, 0.9163, 0.8700, 0.0017], [590, 1.0263, 0.7570, 0.0011],
    [600, 1.0622, 0.6310, 0.0008], [610, 1.0026, 0.5030, 0.0003],
    [620, 0.8544, 0.3810, 0.0002], [630, 0.6424, 0.2650, 0.0000],
    [640, 0.4479, 0.1750, 0.0000], [650, 0.2835, 0.1070, 0.0000],
    [660, 0.1649, 0.0610, 0.0000], [670, 0.0874, 0.0320, 0.0000],
    [680, 0.0468, 0.0170, 0.0000], [690, 0.0227, 0.0082, 0.0000],
    [700, 0.0114, 0.0041, 0.0000],
])
_CIE_WL = _CIE[:, 0]
_CIE_XYZ = _CIE[:, 1:]


def _planck(wavelength_nm, temp=5778.0):
    """Planck blackbody spectral radiance (arbitrary units) -- the solar source."""
    h, c, kB = 6.626e-34, 2.998e8, 1.381e-23
    wl = wavelength_nm * 1e-9
    return (2 * h * c ** 2) / (wl ** 5) / (np.exp(h * c / (wl * kB * temp)) - 1)


def spectrum_to_srgb(spectrum):
    """Convert a spectral power distribution (on _CIE_WL) to a clipped sRGB triple."""
    X = np.trapezoid(spectrum * _CIE_XYZ[:, 0], _CIE_WL)
    Y = np.trapezoid(spectrum * _CIE_XYZ[:, 1], _CIE_WL)
    Z = np.trapezoid(spectrum * _CIE_XYZ[:, 2], _CIE_WL)
    s = X + Y + Z
    if s <= 0:
        return np.zeros(3)
    X, Y, Z = X / s, Y / s, Z / s          # chromaticity-normalise, keep hue
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]])
    rgb = M @ np.array([X, Y, Z])
    rgb = np.clip(rgb, 0, None)
    if rgb.max() > 0:
        rgb = rgb / rgb.max()              # normalise brightness for display
    # gamma encode (sRGB)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                   1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.clip(rgb, 0, 1)


# --------------------------------------------------------------------------- #
#  Figure 5 -- Full-spectrum transport -> rendered sunset colour  [optimisation]
# --------------------------------------------------------------------------- #
def figure_spectral_color():
    """Transmit the solar spectrum through increasing airmass and render colour."""
    source = _planck(_CIE_WL)
    airmasses = [0.0, 1.0, 5.0, 10.0, 20.0, 30.0, 38.0]

    fig = plt.figure(figsize=(7.6, 5.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.1, 1.0], hspace=0.42)

    ax = fig.add_subplot(gs[0])
    swatches = []
    for m in airmasses:
        tau = TAU_ZENITH_550 * (LAMBDA_REF / _CIE_WL) ** 4 * m
        transmitted = source * np.exp(-tau)
        ax.plot(_CIE_WL, transmitted / source.max(),
                label=f"m = {m:g}")
        swatches.append(spectrum_to_srgb(transmitted))
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("transmitted spectral power\n(relative to source peak)")
    ax.set_title("Fig. 5  Spectral transmission $I_0(\\lambda)e^{-\\tau(\\lambda,m)}$ "
                 "and the resulting sky colour")
    ax.legend(frameon=False, fontsize=8, ncol=2, title="airmass")
    # colour the wavelength axis background faintly
    ax.set_xlim(380, 700)

    # Lower panel: rendered colour swatches as airmass grows.
    axc = fig.add_subplot(gs[1])
    for i, (m, rgb) in enumerate(zip(airmasses, swatches)):
        axc.add_patch(plt.Rectangle((i, 0), 1, 1, color=rgb))
        axc.text(i + 0.5, -0.18, f"m={m:g}", ha="center", va="top", fontsize=8)
    axc.set_xlim(0, len(airmasses))
    axc.set_ylim(-0.25, 1)
    axc.axis("off")
    axc.set_title("Rendered colour of the direct solar beam "
                  "(zenith -> horizon)", fontsize=10)
    fig.savefig(os.path.join(FIG_DIR, "fig5_spectral_color.png"),
                bbox_inches="tight")
    plt.close(fig)

    RESULTS["sunset_color_rgb_horizon"] = [round(float(c), 3)
                                           for c in swatches[-1]]


# --------------------------------------------------------------------------- #
#  Figure 6 -- Joint [P,T,H] distribution + Weibull turbidity -> afterglow prob.
# --------------------------------------------------------------------------- #
def figure_joint_afterglow():
    """Sample a correlated weather vector and estimate P(afterglow).

    V = [P, T, H] is drawn from a multivariate normal (correlated: cold fronts
    raise pressure and lower temperature).  Atmospheric turbidity (aerosol
    loading) is an independent Weibull random variable.  The afterglow forms
    when the optical geometry reddens the beam *and* there is enough aerosol /
    cloud deck to catch the light, but not so much that it is fully blocked --
    a joint threshold (a "band") on a derived redness index.
    """
    n = 200_000
    # correlated [P (hPa), T (degC), H (%)]
    mean = np.array([1015.0, 15.0, 60.0])
    # correlation: P up <-> T down (-0.5); H up <-> T up a bit (+0.3)
    std = np.array([12.0, 8.0, 20.0])
    corr = np.array([[1.0, -0.5, -0.2],
                     [-0.5, 1.0, 0.3],
                     [-0.2, 0.3, 1.0]])
    cov = np.outer(std, std) * corr
    V = RNG.multivariate_normal(mean, cov, size=n)
    P, T, H = V[:, 0], V[:, 1], np.clip(V[:, 2], 0, 100)

    # Weibull turbidity tau_aer (aerosol optical depth): shape k=1.8, scale=0.12
    k_shape, scale = 1.8, 0.12
    tau_aer = scale * RNG.weibull(k_shape, size=n)

    # Air number density ~ P/T (ideal gas) modulates Rayleigh optical depth.
    Tk = T + 273.15
    density_factor = (P / 1013.25) * (288.15 / Tk)
    m_horizon = 38.0
    tau_red = rayleigh_optical_depth(RED_NM, m_horizon) * density_factor + 0.4 * tau_aer
    tau_blue = rayleigh_optical_depth(BLUE_NM, m_horizon) * density_factor + tau_aer

    # Redness index: how strongly blue is suppressed relative to red, weighted by
    # the surviving red flux that can illuminate a cloud deck.
    redness = (np.exp(-tau_red) - np.exp(-tau_blue)) * np.exp(-0.5 * tau_red)
    # Humidity supplies the cloud deck that reflects the reddened light.
    cloud = 1.0 / (1.0 + np.exp(-(H - 55) / 6.0))      # logistic in humidity
    afterglow_score = redness * cloud

    # Afterglow event: score in a productive band (enough redness + cloud, not
    # washed out by too much aerosol blocking everything).
    lo = np.quantile(afterglow_score, 0.80)
    event = afterglow_score >= lo
    p_after = float(event.mean())

    fig = plt.figure(figsize=(8.6, 6.2))
    gs = gridspec.GridSpec(2, 2, hspace=0.62, wspace=0.32)

    # (a) joint P-T scatter coloured by afterglow score
    ax0 = fig.add_subplot(gs[0, 0])
    sub = slice(0, 6000)
    sc = ax0.scatter(P[sub], T[sub], c=afterglow_score[sub], s=4,
                     cmap="magma", alpha=0.6)
    ax0.set_xlabel("pressure P (hPa)")
    ax0.set_ylabel("temperature T (degC)")
    ax0.set_title("Fig. 6a  Correlated joint $[P,T]$\ncoloured by afterglow score")
    plt.colorbar(sc, ax=ax0, shrink=0.85, label="score")

    # (b) Weibull turbidity histogram
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.hist(tau_aer, bins=80, density=True, color="#8c6d31", alpha=0.7)
    grid = np.linspace(0, tau_aer.max(), 300)
    pdf = (k_shape / scale) * (grid / scale) ** (k_shape - 1) * \
        np.exp(-(grid / scale) ** k_shape)
    ax1.plot(grid, pdf, "k", lw=2, label=f"Weibull(k={k_shape}, $\\lambda$={scale})")
    ax1.set_xlabel("aerosol optical depth  $\\tau_{aer}$")
    ax1.set_ylabel("density")
    ax1.set_title("Fig. 6b  Weibull turbidity")
    ax1.legend(frameon=False, fontsize=8)

    # (c) afterglow score distribution with threshold band
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(afterglow_score, bins=90, color="#c44e52", alpha=0.75)
    ax2.axvline(lo, color="k", ls="--", lw=1.5,
                label=f"threshold (P={p_after:.2f})")
    ax2.set_xlabel("afterglow score")
    ax2.set_ylabel("count")
    ax2.set_title("Fig. 6c  Afterglow score & threshold")
    ax2.legend(frameon=False, fontsize=8)

    # (d) afterglow probability vs humidity (conditional probability)
    ax3 = fig.add_subplot(gs[1, 1])
    hbins = np.linspace(10, 100, 19)
    centers, probs = [], []
    for a, b in zip(hbins[:-1], hbins[1:]):
        msk = (H >= a) & (H < b)
        if msk.sum() > 50:
            centers.append((a + b) / 2)
            probs.append(event[msk].mean())
    ax3.plot(centers, probs, "o-", color="#4c72b0")
    ax3.set_xlabel("relative humidity H (%)")
    ax3.set_ylabel("$P(\\mathrm{afterglow}\\mid H)$")
    ax3.set_title("Fig. 6d  Conditional probability\non humidity")
    fig.savefig(os.path.join(FIG_DIR, "fig6_joint.png"), bbox_inches="tight")
    plt.close(fig)

    # conditional probabilities for the report
    hi_H = event[H > 75].mean() if (H > 75).any() else float("nan")
    lo_H = event[H < 45].mean() if (H < 45).any() else float("nan")
    RESULTS["joint"] = {
        "p_afterglow": p_after,
        "weibull_shape": k_shape, "weibull_scale": scale,
        "p_afterglow_given_high_humidity": float(hi_H),
        "p_afterglow_given_low_humidity": float(lo_H),
        "corr_PT": -0.5,
    }


# --------------------------------------------------------------------------- #
def main():
    print("Generating figures ...")
    figure_exponential_validation(); print("  fig1 exponential  ok")
    figure_beer_lambert();           print("  fig2 beer-lambert ok")
    figure_lln();                    print("  fig3 LLN/CLT      ok")
    figure_random_walk();            print("  fig4 random walk  ok")
    figure_spectral_color();         print("  fig5 spectral col ok")
    figure_joint_afterglow();        print("  fig6 joint dist   ok")

    with open(os.path.join(ROOT, "results.json"), "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("\nKey results:")
    print(json.dumps(RESULTS, indent=2))


if __name__ == "__main__":
    main()
