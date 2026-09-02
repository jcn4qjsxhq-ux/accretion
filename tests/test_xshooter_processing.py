from pathlib import Path

import numpy as np
import pandas as pd
from scipy import constants

from accretion import calibrate_spectra_to_photometry, load_xshooter_nir_spectra
from accretion.ultimate_common import M_sun
from accretion.final_panels import _spectral_source
from accretion.ultimate_visualisation import _stellar_mass_solar


def test_xshooter_conversion_uses_air_wavelength_and_bins_signed_pixels(tmp_path: Path):
    path = tmp_path / "final_nir_test.txt"
    path.write_text(
        "Wavelength (um)   Flux (erg/s/cm^2/AA)\n"
        "1.2000  1.0e-15\n"
        "1.2005 -0.5e-15\n"
        "1.2010  1.0e-15\n"
        "1.2015  1.0e-15\n",
        encoding="utf-8",
    )

    frame = load_xshooter_nir_spectra(
        tmp_path,
        wavelength_range=(1.19e-6, 1.21e-6),
        mask_regions=[],
        epoch_metadata={"test": {"JD": 2450000.5, "Date": "1995-10-10"}},
        bin_width_micron=0.01,
        sigma_clip=None,
    )

    expected_samples = np.array([1.0, -0.5, 1.0, 1.0]) * 1e-15
    wavelengths_um = np.array([1.2000, 1.2005, 1.2010, 1.2015])
    expected_jy = expected_samples * (wavelengths_um * 1e4) ** 2
    expected_jy /= constants.c * 1e10
    expected_jy *= 1e23

    assert len(frame) == 1
    assert frame.iloc[0]["SamplesPerBin"] == 4
    assert np.isclose(frame.iloc[0]["Flux"], np.mean(expected_jy))


def test_photometric_calibration_recovers_scale_and_colour():
    wavelength = np.array([1.20, 1.235, 1.50, 1.662, 1.70])
    true_flux = 0.01 * wavelength**2
    imposed_correction = 1.6 * (wavelength / 1.4) ** -0.6
    observed_flux = true_flux / imposed_correction
    spectrum = pd.DataFrame({
        "JD": 2460000.5,
        "JD_day": 2460000.5,
        "Lambda": wavelength,
        "Flux": observed_flux,
        "Fluxerr": observed_flux * 0.1,
    })
    photometry = pd.DataFrame({
        "JD": [2460000.5, 2460000.5],
        "Filter": ["J", "H"],
        "Lambda": [1.235, 1.662],
        "Mag": [-2.5 * np.log10(true_flux[1]), -2.5 * np.log10(true_flux[3])],
        "ZP": [1.0, 1.0],
        "Flux": [0.0, 0.0],
        "Fluxerr": [0.0, 0.0],
    })

    calibrated = calibrate_spectra_to_photometry(
        spectrum,
        photometry,
        window_half_width_micron=0.001,
    )

    assert np.allclose(calibrated["Flux"], true_flux, rtol=1e-10)
    assert np.allclose(calibrated["FluxUncalibrated"], observed_flux)


def test_stellar_mass_for_mstar_mdot_uses_model_mass():
    assert _stellar_mass_solar({"fit_info": {"stellar_mass_solar": 0.5}}) == 0.5
    assert _stellar_mass_solar({"global_params": {"M": 0.7 * M_sun}}) == 0.7
    # Legacy result bundles omitted the fixed 0.5 Msun model mass.
    assert _stellar_mass_solar({"global_params": {}, "fit_info": {}}) == 0.5


def test_spectral_plot_source_uses_saved_provenance():
    xshooter = pd.DataFrame({
        "Instrument": ["VLT/XSHOOTER"],
        "Filename": ["final_nir_202402.txt"],
    })
    irtf = pd.DataFrame({
        "Instrument": ["SpeX Spectrograph"],
        "Filename": ["merged_v346nor_sxd_20240912.txt"],
    })
    results = {"daily_data": {1.0: xshooter, 2.0: irtf}}
    assert _spectral_source(results, 1.0) == "XSHOOTER"
    assert _spectral_source(results, 2.0) == "IRTF/SpeX"
