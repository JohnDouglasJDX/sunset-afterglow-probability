"""
Build the >=5 page PDF report from the simulation outputs (results.json + figures).
Display equations are rendered as images via matplotlib mathtext so the maths
looks typeset without needing a LaTeX engine installed.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(HERE, "figures")
EQDIR = os.path.join(HERE, "figures", "eq")
REPORT_DIR = os.path.join(HERE, "report")
os.makedirs(EQDIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

with open(os.path.join(HERE, "results.json")) as f:
    R = json.load(f)


# --------------------------------------------------------------------------- #
#  Equation rendering (LaTeX-ish via mathtext) -> PNG flowable
# --------------------------------------------------------------------------- #
def eq(latex, name, fontsize=17):
    """Render a single display equation to a PNG and return its path."""
    path = os.path.join(EQDIR, f"{name}.png")
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, f"${latex}$", fontsize=fontsize)
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.08,
                transparent=True)
    plt.close(fig)
    return path


def eq_flowable(latex, name, width=None):
    path = eq(latex, name)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    scale = 200.0 / 72.0          # dpi -> points
    pw, ph = w / scale, h / scale
    if width and pw > width:
        ph *= width / pw
        pw = width
    img = Image(path, width=pw, height=ph)
    img.hAlign = "CENTER"
    return img


# --------------------------------------------------------------------------- #
#  Styles
# --------------------------------------------------------------------------- #
styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.2,
                      leading=14.5, alignment=TA_JUSTIFY, spaceAfter=7)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14,
                    spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#7a1f1f"))
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11.5,
                    spaceBefore=8, spaceAfter=4, textColor=colors.HexColor("#33414f"))
title = ParagraphStyle("title", parent=styles["Title"], fontSize=18, leading=22)
subtitle = ParagraphStyle("subtitle", parent=styles["Normal"], fontSize=11,
                          alignment=TA_CENTER, textColor=colors.HexColor("#555555"))
caption = ParagraphStyle("caption", parent=styles["Normal"], fontSize=8.8,
                         alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                         spaceBefore=2, spaceAfter=10, leading=11)

S = []


def P(txt):
    S.append(Paragraph(txt, body))


def fig_block(fname, cap, width=5.6 * inch):
    path = os.path.join(FIG, fname)
    from PIL import Image as PILImage
    w, h = PILImage.open(path).size
    height = width * h / w
    img = Image(path, width=width, height=height)
    img.hAlign = "CENTER"
    S.append(Spacer(1, 4))
    S.append(img)
    S.append(Paragraph(cap, caption))


# --------------------------------------------------------------------------- #
#  Pull frequently used numbers
# --------------------------------------------------------------------------- #
h = R["horizon"]
lln = R["lln"]
rw = R["random_walk"]
jt = R["joint"]
rgb = R["sunset_color_rgb_horizon"]
rw_red = rw["red 700 nm"]
rw_blue = rw["blue 450 nm"]
nrw = rw["n_photons"]


# =========================================================================== #
#  TITLE
# =========================================================================== #
S.append(Spacer(1, 6))
S.append(Paragraph("Stochastic Modeling of Sunset Afterglow Probability", title))
S.append(Spacer(1, 2))
S.append(Paragraph("Atmospheric Particle Collisions and Rayleigh Scattering "
                   "as a Continuous-Time Stochastic Process", subtitle))
S.append(Spacer(1, 6))
S.append(Paragraph("Probability Theory and Stochastic Processes "
                   "&mdash; Course Project", subtitle))
S.append(Spacer(1, 2))
S.append(Paragraph("John Douglas", subtitle))
S.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#7a1f1f"),
                    spaceBefore=8, spaceAfter=10))

P("<b>Abstract.</b> This report recasts the formation of a sunset afterglow &mdash; the "
  "pink-to-crimson twilight sky &mdash; as a problem in probability rather than "
  "deterministic radiative transfer. A photon's path through the atmosphere is "
  "modeled as a continuous random variable; wavelength-dependent extinction "
  "becomes a conditional survival probability; and the appearance of the "
  "afterglow is a joint-threshold event on a correlated weather vector. A "
  "Monte&nbsp;Carlo simulator of "
  f"{nrw:,}&ndash;{200000:,} independent photon random walks reproduces the "
  "analytic Exponential and Beer&ndash;Lambert results, empirically verifies the "
  "Law of Large Numbers and the Central Limit Theorem error scaling, and &mdash; "
  "as an extension &mdash; performs full-spectrum transport with CIE colour "
  "rendering to estimate the <i>display-normalized chromaticity</i> of the setting "
  f"Sun (sRGB&nbsp;&approx;&nbsp;[{rgb[0]:.2f},&nbsp;{rgb[1]:.2f},&nbsp;{rgb[2]:.2f}], a "
  "vivid orange&ndash;red) as the line of sight sweeps from zenith to horizon.")

# =========================================================================== #
#  SECTION 1
# =========================================================================== #
S.append(Paragraph("1.&nbsp;&nbsp;Introduction &amp; Meteorological Background", h1))
P("At noon the Sun's light traverses roughly one &ldquo;airmass&rdquo; of "
  "atmosphere; at sunset the same light grazes the planet and crosses an "
  "optical path tens of times longer (airmass <i>m</i>&nbsp;&approx;&nbsp;38 at the "
  "horizon). Along that path, photons collide with air molecules and aerosols "
  "and are removed from the direct beam by <b>Rayleigh scattering</b>, whose "
  "strength grows as the inverse fourth power of wavelength. Short-wavelength "
  "blue light is therefore stripped out far more aggressively than "
  "long-wavelength red, and what survives to the observer &mdash; and reflects "
  "off any cloud deck &mdash; is the warm afterglow.")
P("The classical treatment of this phenomenon is deterministic: one integrates "
  "the radiative-transfer equation through a stratified medium. This project "
  "takes the complementary <b>statistical-mechanics</b> view advocated in the "
  "course. A single photon is an agent performing a random walk; the macroscopic "
  "colour of the sky is an <i>expectation</i> over an enormous ensemble of such "
  "agents. This reframing brings the core machinery of the course &mdash; "
  "continuous distributions, conditional probability, joint distributions, the "
  "Law of Large Numbers, and Monte&nbsp;Carlo estimation &mdash; to bear on a tangible, "
  "everyday physical system. Section&nbsp;2 builds the probabilistic model, "
  "Section&nbsp;3 specifies the simulator, Section&nbsp;4 analyses the generated "
  "data, and Section&nbsp;5 concludes.")

# =========================================================================== #
#  SECTION 2
# =========================================================================== #
S.append(Paragraph("2.&nbsp;&nbsp;Mathematical Derivations &amp; Probabilistic "
                   "Framework", h1))

S.append(Paragraph("2.1&nbsp;&nbsp;Free path as an Exponential random variable", h2))
P("Let scattering centres be distributed along the photon's path as a "
  "homogeneous Poisson process with rate (collision density) "
  "<i>&lambda;</i>&nbsp;=&nbsp;<i>g</i>(<i>P,T</i>), set by the local air "
  "density. The distance <i>X</i> to the <i>first</i> collision is then the "
  "waiting time of that Poisson process, i.e. an Exponential random variable:")
S.append(eq_flowable(r"f(x;\lambda)=\lambda e^{-\lambda x},\quad x\geq 0,"
                     r"\qquad \mathbb{E}[X]=\frac{1}{\lambda}=\ell_{\mathrm{mfp}}",
                     "exp_pdf", width=4.4 * inch))
P("with 1/<i>&lambda;</i> the mean free path. The survival "
  "function follows immediately, and it is exactly the probability that a photon "
  "crosses a path of length <i>L</i> with no collision:")
S.append(eq_flowable(r"P(X>L)=\int_L^{\infty}\lambda e^{-\lambda x}\,dx="
                     r"e^{-\lambda L}=e^{-\tau},\qquad \tau\equiv\lambda L",
                     "survival", width=4.4 * inch))
P("The dimensionless product <i>&tau;</i>&nbsp;=&nbsp;<i>&lambda;L</i> is the "
  "<b>optical depth</b> &mdash; the expected number of collisions along the path. "
  "The variable <i>X</i> is sampled by <b>inverse-transform sampling</b>: if "
  "<i>U</i>&nbsp;~&nbsp;Uniform(0,1), then because the CDF is "
  "<i>F</i>(x)&nbsp;=&nbsp;1&minus;<i>e</i><sup>&minus;&lambda;x</sup> "
  "(and 1&minus;<i>U</i> is itself Uniform(0,1)), the free path is sampled as")
S.append(eq_flowable(r"X=F^{-1}(U)=-\frac{1}{\lambda}\ln(1-U)\;=\;"
                     r"-\frac{1}{\lambda}\ln U\ \sim\ "
                     r"\mathrm{Exp}(\lambda).", "inverse", width=4.2 * inch))

S.append(Paragraph("2.2&nbsp;&nbsp;Wavelength filtering as conditional probability", h2))
P("The Rayleigh cross-section scales as "
  "<i>&sigma;</i>(<i>&lambda;</i>)&nbsp;&prop;&nbsp;<i>&lambda;</i><sup>&minus;4</sup>, "
  "so the collision rate &mdash; and hence the optical depth &mdash; inherits the "
  "same dependence. Writing the zenith optical depth at the reference wavelength "
  "as <i>&tau;</i><sub>0</sub> and the slant path as <i>m</i> airmasses,")
S.append(eq_flowable(r"\tau(\lambda,m)=m\,\tau_0\left(\frac{\lambda_0}"
                     r"{\lambda}\right)^{4},\qquad \lambda_0=550\,\mathrm{nm},"
                     r"\;\tau_0\approx0.097.", "tau_lambda", width=4.2 * inch))
P("Defining the event <i>A</i>&nbsp;=&nbsp;&ldquo;the photon reaches the observer "
  "along the direct beam,&rdquo; the conditional survival probabilities for red "
  "and blue light are <i>P</i>(<i>A</i>&thinsp;|&thinsp;<i>&lambda;</i>)&nbsp;=&nbsp;"
  "<i>e</i><sup>&minus;&tau;(&lambda;,m)</sup>. At the horizon "
  f"(<i>m</i>&nbsp;=&nbsp;38) the model gives <i>&tau;</i><sub>red</sub>&nbsp;=&nbsp;"
  f"{h['tau_red']:.2f} and <i>&tau;</i><sub>blue</sub>&nbsp;=&nbsp;{h['tau_blue']:.2f}, so")
S.append(eq_flowable(
    rf"P(A\mid700\,\mathrm{{nm}})=e^{{-{h['tau_red']:.2f}}}={h['surv_red']:.3f},"
    rf"\quad P(A\mid450\,\mathrm{{nm}})=e^{{-{h['tau_blue']:.2f}}}="
    rf"{h['surv_blue']:.1e}.", "cond", width=5.0 * inch))
P(f"Red light is therefore about <b>{h['red_to_blue_ratio']:.0f}&times;</b> more "
  "likely to survive the horizon path than blue &mdash; the quantitative origin "
  "of the reddened Sun. (Note this single-collision survival is the colour of the "
  "direct disk; the afterglow proper is the surviving red flux re-scattered toward "
  "the observer, modeled in &sect;2.3 and the random walk of &sect;3.)")

S.append(Paragraph("2.3&nbsp;&nbsp;Afterglow as a joint-threshold event", h2))
P("Whether a vivid afterglow actually forms depends on the state of the "
  "atmosphere, captured by a random vector "
  "<b>V</b>&nbsp;=&nbsp;[<i>P,T,H</i>] (pressure, temperature, humidity) drawn from "
  "a multivariate distribution. Here <b>V</b> is modeled as multivariate Normal with a "
  "physically motivated correlation (a passing cold front raises pressure while "
  "lowering temperature, so "
  f"&rho;<sub>PT</sub>&nbsp;=&nbsp;{jt['corr_PT']}). Air density "
  "<i>&rho;</i>&nbsp;&prop;&nbsp;<i>P/T</i> modulates the Rayleigh depth, while "
  "aerosol turbidity <i>&tau;</i><sub>aer</sub> &mdash; the haze that physically "
  "catches and reddens the light &mdash; is an independent <b>Weibull</b> "
  "variable used here as a flexible positive distribution for atmospheric loading:")
S.append(eq_flowable(r"f(t;k,\lambda)=\frac{k}{\lambda}\left(\frac{t}{\lambda}"
                     r"\right)^{k-1}e^{-(t/\lambda)^{k}},\quad "
                     rf"k={jt['weibull_shape']},\;\lambda={jt['weibull_scale']}.",
                     "weibull", width=4.4 * inch))
P("From these a <i>redness index</i> is built &mdash; the surplus of surviving red "
  "over blue, weighted by the red flux still available to illuminate a cloud deck "
  "&mdash; and gate it by a logistic function of humidity (the cloud/moisture that "
  "reflects the glow). The afterglow is declared when this joint score exceeds a "
  f"fixed threshold ({jt['score_threshold']:.2f}). This threshold is an explicit "
  "teaching assumption, not one fitted to observed sunsets; the resulting event "
  "rate must therefore not be interpreted as a weather forecast.")

# =========================================================================== #
#  SECTION 3
# =========================================================================== #
S.append(Paragraph("3.&nbsp;&nbsp;Simulation Methodology &amp; Pseudocode", h1))
P("The simulator (Python&nbsp;/&nbsp;NumPy, independent reproducible PRNG streams) "
  "instantiates large ensembles of independent photons and propagates each as a "
  "random walk. Two estimators are run: a <i>direct-transmission</i> estimator "
  "that targets the analytic <i>e</i><sup>&minus;&tau;</sup> for clean validation, "
  "and a full <i>multiple-scattering</i> random walk with a collision decision "
  "tree (absorb / forward-scatter / back-scatter).")

S.append(Paragraph("3.1&nbsp;&nbsp;Random-walk transport (collision decision tree)", h2))
pseudo = ParagraphStyle("pseudo", parent=styles["Code"], fontSize=8.3, leading=11,
                        backColor=colors.HexColor("#f4f4f2"),
                        borderColor=colors.HexColor("#cccccc"), borderWidth=0.5,
                        borderPadding=6, leftIndent=2, spaceAfter=8)
code = (
    "for each photon (vectorised over the whole ensemble):\n"
    "    x  &larr; 0                      # optical depth into the slab\n"
    "    &mu;  &larr; +1                     # direction (+1 = toward observer)\n"
    "    repeat:\n"
    "        s &larr; -ln(U),  U~Uniform(0,1)      # Exponential free path\n"
    "        x &larr; x + &mu;&middot;s\n"
    "        if x &ge; &tau;_total:  outcome &larr; TRANSMITTED;  break\n"
    "        if x &le; 0:        outcome &larr; BACK-SCATTERED; break\n"
    "        if U' &ge; &omega;:     outcome &larr; ABSORBED;      break   # Bernoulli(1-&omega;)\n"
    "        if U'' &ge; p_fwd:  &mu; &larr; -&mu;                       # Bernoulli scatter dir.\n"
    "    tally outcome")
S.append(Paragraph(code.replace("\n", "<br/>").replace("  ", "&nbsp;&nbsp;"), pseudo))
P("Here <i>&tau;</i><sub>total</sub> is the wavelength-dependent optical thickness "
  f"of the slab, <i>&omega;</i>&nbsp;=&nbsp;{rw['albedo']} the single-scattering "
  "albedo (Rayleigh scattering is nearly conservative, so absorption is rare), and "
  "<i>p</i><sub>fwd</sub>&nbsp;=&nbsp;0.5 the forward fraction of the (symmetric) "
  "Rayleigh phase function reduced to one dimension. Every random draw &mdash; free "
  "path, absorption, and scattering direction &mdash; is an application of "
  "inverse-transform or Bernoulli sampling from &sect;2.")

S.append(Paragraph("3.2&nbsp;&nbsp;Estimators and parameters", h2))
cell = ParagraphStyle("cell", parent=styles["Normal"], fontSize=9, leading=11)
cellc = ParagraphStyle("cellc", parent=cell, alignment=TA_CENTER)
hdr = ParagraphStyle("hdr", parent=cellc, textColor=colors.white)


def _c(txt, center=True):
    return Paragraph(txt, cellc if center else cell)


params = [
    [Paragraph("Quantity", hdr), Paragraph("Symbol", hdr), Paragraph("Value", hdr)],
    [_c("Photons (transmission / spectral)", False), _c("<i>N</i>"),
     _c("5&times;10<super>4</super> &ndash; 2&times;10<super>5</super>")],
    [_c("Reference zenith optical depth", False),
     _c("&tau;<sub>0</sub>(550&thinsp;nm)"), _c("0.097")],
    [_c("Horizon airmass", False), _c("<i>m</i>"), _c("38")],
    [_c("Single-scattering albedo", False), _c("&omega;"), _c(f"{rw['albedo']}")],
    [_c("Weibull turbidity (shape, scale)", False), _c("<i>k</i>, &lambda;"),
     _c(f"{jt['weibull_shape']}, {jt['weibull_scale']}")],
    [_c("Weather correlation", False), _c("&rho;(<i>P,T</i>)"),
     _c(f"{jt['corr_PT']}")],
]
tbl = Table(params, hAlign="CENTER", colWidths=[2.5*inch, 1.2*inch, 1.6*inch])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#7a1f1f")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f3eeee")]),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
S.append(tbl)
S.append(Spacer(1, 4))
P("Inverse-transform sampling is validated directly in Fig.&nbsp;1: histograms of "
  f"{200000:,} sampled free paths overlay the analytic Exponential PDF, and the "
  "empirical means reproduce 1/<i>&lambda;</i> to three significant figures "
  "(0.500 vs 0.500 and 2.001 vs 2.000), an instance of the LLN in its own right.")
fig_block("fig1_exponential.png",
          "Fig.&nbsp;1&nbsp;&mdash; Inverse-transform samples (bars) versus the "
          "analytic Exponential PDF (curves) for two collision densities. The "
          "sampler is exact up to Monte&nbsp;Carlo noise.", width=5.0*inch)

# =========================================================================== #
#  SECTION 4
# =========================================================================== #
S.append(Paragraph("4.&nbsp;&nbsp;Data Visualization &amp; Critical Discussion", h1))

S.append(Paragraph("4.1&nbsp;&nbsp;Conditional survival: why the Sun reddens", h2))
P("Fig.&nbsp;2 overlays the Monte&nbsp;Carlo survival fractions on the analytic "
  "<i>e</i><sup>&minus;&tau;(&lambda;,m)</sup> curves across airmass. Agreement is "
  "essentially perfect. The two curves diverge dramatically toward the horizon: at "
  f"<i>m</i>&nbsp;=&nbsp;38 red survival is {h['surv_red']:.3f} while blue is only "
  f"{h['surv_blue']:.1e}. The "
  f"{h['red_to_blue_ratio']:.0f}:1 ratio is the probabilistic signature of the "
  "reddening &mdash; blue photons almost never complete the grazing journey.")
fig_block("fig2_beer_lambert.png",
          "Fig.&nbsp;2&nbsp;&mdash; Conditional direct-transmission probability "
          "vs airmass. Markers are Monte&nbsp;Carlo; lines are the analytic "
          "survival function. Blue is extinguished far faster than red.",
          width=4.9*inch)

S.append(Paragraph("4.2&nbsp;&nbsp;Law of Large Numbers and CLT error scaling", h2))
P("Fig.&nbsp;3 (top) tracks the running Monte&nbsp;Carlo estimator "
  "<i>p</i><sub>N</sub>&nbsp;=&nbsp;(1/<i>N</i>)&Sigma;&thinsp;<b>1</b>{<i>X</i>&gt;&tau;} (the "
  "sample fraction of surviving photons) as "
  "<i>N</i> grows. Both red and blue estimators settle onto their analytic limits "
  f"(<i>e</i><sup>&minus;&tau;</sup>&nbsp;=&nbsp;{lln['red_limit']:.3f} and "
  f"{lln['blue_limit']:.1e}); at N&nbsp;=&nbsp;{lln['N']:,} the residual errors are "
  f"{lln['red_final_abs_error']:.1e} and {lln['blue_final_abs_error']:.1e}. This is "
  "the <b>Law of Large Numbers</b> made visible. The shaded band is the "
  "Central-Limit &plusmn;2&nbsp;standard-error envelope "
  "&radic;(<i>p</i>(1&minus;<i>p</i>)/N). The bottom panel estimates RMSE over "
  f"{lln['rmse_replicates']} independent repetitions at each sample size; fitted "
  f"log&ndash;log slopes of {lln['red_rmse_loglog_slope']:.2f} (red) and "
  f"{lln['blue_rmse_loglog_slope']:.2f} (blue) track the predicted "
  "<b>N<sup>&minus;1/2</sup></b> law. The "
  "estimator's precision thus improves only as the square root of effort &mdash; the "
  "central practical cost of Monte&nbsp;Carlo.")
fig_block("fig3_lln.png",
          "Fig.&nbsp;3&nbsp;&mdash; Top: running survival estimators converging to "
          "<i>e</i><sup>&minus;&tau;</sup> with the CLT &plusmn;2&thinsp;SE band. "
          "Bottom: repeated-run RMSE vs N with the N<sup>&minus;1/2</sup> "
          "reference.", width=4.6*inch)

S.append(Paragraph("4.3&nbsp;&nbsp;Multiple scattering: the collision decision tree", h2))
rt = 100*rw_red["transmitted"]/nrw
bt = 100*rw_blue["transmitted"]/nrw
bb = 100*rw_blue["backscattered"]/nrw
P("Fig.&nbsp;4 reports the full random walk with scattering and absorption. The "
  "terminal outcomes are starkly wavelength-dependent: "
  f"{rt:.0f}% of red photons are ultimately transmitted toward the observer versus "
  f"only {bt:.0f}% of blue, while {bb:.0f}% of blue photons are back-scattered to "
  "space. The same wavelength-selective scattering mechanism contributes to the "
  "blue daytime sky, although this one-dimensional model cannot predict angular "
  "sky radiance. The sample "
  "trajectories (Fig.&nbsp;4b) show the characteristic random walk: most blue "
  "photons reverse and escape before threading the full optical thickness "
  f"<i>&tau;</i>&nbsp;=&nbsp;{rw_blue['tau']:.1f}.")
fig_block("fig4_random_walk.png",
          "Fig.&nbsp;4&nbsp;&mdash; (a) Terminal outcome fractions of the "
          "multiple-scattering random walk. (b) Representative blue-photon "
          "trajectories in optical-depth space.", width=5.3*inch)

S.append(Paragraph("4.4&nbsp;&nbsp;Extension &mdash; direct-beam chromaticity", h2))
P("Beyond the brief, the two-colour model is extended to the <i>full visible "
  "spectrum</i>. Treating sunlight as a 5778&nbsp;K blackbody source "
  "<i>I</i><sub>0</sub>(<i>&lambda;</i>), each wavelength is transmitted through "
  "<i>e</i><sup>&minus;&tau;(&lambda;,m)</sup>, the result is integrated against the "
  "<b>CIE&nbsp;1931</b> colour-matching functions, and the XYZ result is converted to "
  "sRGB to obtain a display-normalized chromaticity of the direct beam. Fig.&nbsp;5 shows the "
  "transmitted spectrum progressively losing its blue end as airmass grows, and the "
  "rendered swatches march from near-white at the zenith to a vivid orange&ndash;red "
  f"(sRGB&nbsp;&approx;&nbsp;[{rgb[0]:.2f},&nbsp;{rgb[1]:.2f},&nbsp;{rgb[2]:.2f}]) at the "
  "horizon. Because brightness is normalized and ozone, multiple scattering, "
  "camera exposure, and visual adaptation are omitted, these swatches illustrate "
  "the model's hue shift rather than a calibrated perceived colour.")
fig_block("fig5_spectral_color.png",
          "Fig.&nbsp;5&nbsp;&mdash; Full-spectrum transmission (top) and the "
          "resulting CIE-rendered colour of the direct solar beam from zenith "
          "(m=0) to horizon (m=38).", width=5.1*inch)

S.append(Paragraph("4.5&nbsp;&nbsp;Joint weather score and sensitivity", h2))
P("Finally, Fig.&nbsp;6 samples the correlated weather vector "
  "<b>V</b>&nbsp;=&nbsp;[<i>P,T,H</i>] together with Weibull turbidity and evaluates "
  "the joint-threshold afterglow event. The marginal turbidity (6b) matches the "
  "target Weibull density. Applying the fixed teaching threshold "
  f"{jt['score_threshold']:.2f} produces an illustrative event rate of "
  f"{jt['illustrative_event_rate']:.2f}. The rate is higher for humid scenarios "
  f"({jt['illustrative_rate_given_high_humidity']:.2f} above 75% humidity versus "
  f"{jt['illustrative_rate_given_low_humidity']:.2f} below 45%) because humidity "
  "is explicitly included in the score. This is a sensitivity demonstration, "
  "not evidence that the model forecasts real sunsets. Calibration would require "
  "labelled observations matched to atmospheric measurements.")
fig_block("fig6_joint.png",
          "Fig.&nbsp;6&nbsp;&mdash; (a) Correlated [P,T] coloured by afterglow "
          "score; (b) Weibull turbidity vs analytic PDF; (c) score distribution "
          "and fixed decision threshold; (d) illustrative rate conditioned on "
          "humidity.", width=5.4*inch)

P("<b>Variance &amp; limitations.</b> Estimator variance falls as 1/N (Fig.&nbsp;3), "
  "so the rare blue-survival probability (~10<sup>&minus;4</sup>) is the noisiest "
  "quantity and would benefit from importance sampling. The transport is reduced to "
  "one dimension (two-stream), which captures extinction and the forward/back "
  "competition but not full angular redistribution; the colour model omits ozone's "
  "Chappuis absorption and multiple-scattering skylight. None of these change the "
  "qualitative probabilistic story. The weather distributions, correlations, and "
  "score threshold are pedagogical assumptions rather than calibrated climatology.")

# =========================================================================== #
#  SECTION 5
# =========================================================================== #
S.append(Paragraph("5.&nbsp;&nbsp;Conclusion", h1))
P("Modeling the sunset as a stochastic process rather than a differential equation "
  "makes it possible to derive its signature colour from first principles of "
  "probability. The "
  "Exponential free-path law and Beer&ndash;Lambert survival "
  "(<i>e</i><sup>&minus;&tau;</sup>) explain the reddening as a conditional "
  f"probability with a ~{h['red_to_blue_ratio']:.0f}:1 red-to-blue advantage at the "
  "horizon; the Monte&nbsp;Carlo simulator confirms these limits while empirically "
  "demonstrating the Law of Large Numbers and the CLT's N<sup>&minus;1/2</sup> error "
  "law; the multiple-scattering random walk illustrates wavelength-selective "
  "transport; and the joint Weibull/Gaussian weather model demonstrates how an "
  "explicit score responds to assumptions about humidity and aerosols. The CIE "
  "extension closes the educational loop by rendering the direct-beam hue shift. "
  "A real forecast remains a separate, data-calibration problem.")

S.append(Paragraph("References", h2))
P("Kasten, F. &amp; Young, A. T. (1989), <i>Revised optical air mass tables and "
  "approximation formula</i>, Applied Optics 28(22), 4735&ndash;4738. Bodhaine, "
  "B. A. et al. (1999), <i>On Rayleigh Optical Depth Calculations</i>, Journal "
  "of Atmospheric and Oceanic Technology 16, 1854&ndash;1861. CIE 015:2018, "
  "<i>Colorimetry, 4th Edition</i>.")

P("<i>Declaration.</i> This is individual work. The probabilistic derivations, the "
  "Monte&nbsp;Carlo simulator and random-walk transport, the spectral CIE colour "
  "extension and joint weather model, and all writing and figures are my own work "
  "&mdash; John&nbsp;Douglas.")
P("<i>Reproducibility.</i> All results derive from deterministic, independent "
  "experiment seeds in <font face='Courier'>sunset_afterglow.py</font>; "
  "this PDF is generated programmatically from the resulting "
  "<font face='Courier'>results.json</font> and figure set, so every number quoted "
  "above is traceable to the simulation.")


# --------------------------------------------------------------------------- #
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawString(0.9 * inch, 0.55 * inch,
                      "Stochastic Modeling of Sunset Afterglow Probability")
    canvas.drawRightString(7.6 * inch, 0.55 * inch, f"Page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(os.path.join(REPORT_DIR, "Sunset_Afterglow_Report.pdf"),
                        pagesize=letter, topMargin=0.8 * inch,
                        bottomMargin=0.8 * inch, leftMargin=0.9 * inch,
                        rightMargin=0.9 * inch,
                        title="Stochastic Modeling of Sunset Afterglow Probability",
                        author="John Douglas",
                        subject="Educational stochastic model of atmospheric transmission")
doc.build(S, onFirstPage=footer, onLaterPages=footer)
print("Report written to report/Sunset_Afterglow_Report.pdf")
