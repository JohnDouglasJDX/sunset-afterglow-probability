"""Interactive educational lab for the direct-beam sunset model."""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.sunset_afterglow import (
    BLUE_NM,
    RED_NM,
    Atmosphere,
    direct_beam_spectrum,
    total_optical_depth,
)

st.set_page_config(page_title="Sunset probability lab", page_icon="🌅", layout="wide")
st.title("🌅 Sunset probability lab")
st.caption(
    "Explore how wavelength-dependent survival changes the direct solar beam. "
    "This is an educational radiative-transfer model, not a sunset forecast."
)

with st.sidebar:
    st.header("Atmosphere")
    elevation = st.slider("Solar elevation (degrees)", 0.0, 90.0, 2.0, 0.5)
    aerosol_depth = st.slider("Aerosol optical depth at 550 nm", 0.0, 0.5, 0.08, 0.01)
    angstrom = st.slider("Ångström exponent", 0.0, 2.5, 1.3, 0.1)
    photons = st.slider("Monte Carlo photons", 1_000, 200_000, 50_000, 1_000)
    seed = st.number_input("Experiment seed", 0, 10_000_000, 271828)

atmosphere = Atmosphere(aerosol_depth, angstrom)
beam = direct_beam_spectrum(elevation, atmosphere)
airmass = beam["airmass"]
tau_red = float(total_optical_depth(RED_NM, airmass, atmosphere))
tau_blue = float(total_optical_depth(BLUE_NM, airmass, atmosphere))
p_red, p_blue = np.exp(-tau_red), np.exp(-tau_blue)

rng = np.random.default_rng(seed)
red_estimate = rng.binomial(photons, p_red) / photons
blue_estimate = rng.binomial(photons, p_blue) / photons

c1, c2, c3, c4 = st.columns(4)
c1.metric("Relative airmass", f"{airmass:.2f}")
c2.metric("Red survival (700 nm)", f"{p_red:.4f}", f"MC {red_estimate:.4f}")
c3.metric("Blue survival (450 nm)", f"{p_blue:.4g}", f"MC {blue_estimate:.4g}")
c4.metric("Red/blue survival ratio", f"{p_red / max(p_blue, 1e-15):.0f}:1")

left, right = st.columns([2.2, 1])
with left:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(beam["wavelength_nm"], beam["source"], "--", color="#777777", label="source")
    ax.plot(
        beam["wavelength_nm"],
        beam["transmitted"],
        color="#c73e1d",
        linewidth=2.5,
        label="transmitted direct beam",
    )
    ax.set(xlabel="wavelength (nm)", ylabel="spectral power relative to source peak")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)

with right:
    rgb = tuple(int(round(255 * channel)) for channel in beam["rgb"])
    st.subheader("Display-normalized chromaticity")
    st.markdown(
        f'<div style="height:220px;border-radius:18px;background:rgb{rgb};'
        'border:1px solid #999"></div>',
        unsafe_allow_html=True,
    )
    st.code(f"sRGB ≈ {tuple(round(float(x), 3) for x in beam['rgb'])}")
    st.caption(
        "Brightness is intentionally normalized. Camera exposure, visual adaptation, "
        "ozone, clouds, and multiple-scattered skylight are not modeled."
    )

with st.expander("What the Monte Carlo is demonstrating"):
    st.markdown(
        r"""
For a photon with optical depth $\tau$, direct transmission is the Bernoulli event
$X>\tau$ where $X\sim\mathrm{Exponential}(1)$. Therefore
$P(\mathrm{transmission})=e^{-\tau}$. The Monte Carlo values above are sampled
binomial estimates of the same analytic survival probabilities. Their standard
error shrinks only as $N^{-1/2}$.

This lab predicts direct-beam attenuation under stated assumptions. Predicting
whether a real sunset will be judged spectacular requires calibrated observations.
"""
    )
