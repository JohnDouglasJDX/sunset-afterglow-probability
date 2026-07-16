import numpy as np
import pytest

from src.sunset_afterglow import (
    BLUE_NM,
    RED_NM,
    Atmosphere,
    canonicalize_results,
    direct_beam_spectrum,
    random_walk_slab,
    rayleigh_optical_depth,
    relative_airmass,
    sample_free_path,
)


def test_airmass_matches_expected_limits_and_is_monotonic():
    assert relative_airmass(90) == pytest.approx(1.0, rel=5e-4)
    assert relative_airmass(0) == pytest.approx(37.92, rel=2e-3)
    values = relative_airmass([0, 5, 15, 45, 90])
    assert np.all(np.diff(values) < 0)


def test_blue_rayleigh_depth_exceeds_red():
    red = rayleigh_optical_depth(RED_NM, 10)
    blue = rayleigh_optical_depth(BLUE_NM, 10)
    assert blue / red == pytest.approx((RED_NM / BLUE_NM) ** 4)


def test_inverse_transform_mean_and_validation():
    rng = np.random.default_rng(42)
    samples = sample_free_path(2.0, 200_000, rng)
    assert samples.mean() == pytest.approx(0.5, abs=0.004)
    with pytest.raises(ValueError):
        sample_free_path(0, 10, rng)


def test_random_walk_conserves_photons_and_is_reproducible():
    args = (2.0, 5_000, 0.99, 0.5)
    first = random_walk_slab(*args, np.random.default_rng(7))
    second = random_walk_slab(*args, np.random.default_rng(7))
    assert first == second
    assert sum(first) == 5_000


def test_direct_beam_reddens_near_horizon():
    atmosphere = Atmosphere(0.08, 1.3)
    zenith = direct_beam_spectrum(90, atmosphere)
    horizon = direct_beam_spectrum(0, atmosphere)
    assert np.all((0 <= horizon["rgb"]) & (horizon["rgb"] <= 1))
    assert horizon["rgb"][2] < zenith["rgb"][2]
    assert horizon["airmass"] > zenith["airmass"]


def test_atmosphere_rejects_nonphysical_inputs():
    with pytest.raises(ValueError):
        Atmosphere(-0.1, 1.0)
    with pytest.raises(ValueError):
        Atmosphere(0.1, -1.0)


def test_result_serialization_is_platform_stable():
    value = {"scalar": 2.0057652199937417, "nested": [1.0 / 3.0]}
    assert canonicalize_results(value) == {
        "scalar": 2.005765219994,
        "nested": [0.333333333333],
    }
