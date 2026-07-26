#!/usr/bin/env python3
"""
test_linefit.py
====================

Simple demonstration/smoke test of :mod:`linefit_mvp` on a synthetic
3-D IFU datacube with a known velocity gradient, dispersion, flux and
continuum level.

Run directly::

    $ python test_linefit_mvp.py
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
from astropy.io import fits

from linefit import (
    C_KMS,
    fit_cube,
    gaussian_conv_model,
    load_cube,
)

LINE_CENTER = 2.1655   # microns, arbitrary (e.g. redshifted Halpha in K-band)
CDELT_UM = 0.0005       # spectral pixel scale, microns
N_WAVE = 240
NX, NY = 5, 5
TRUE_CONT = 10.0
TRUE_DISP_KMS = 80.0
TRUE_FLUX = 60.0
NOISE_LEVEL = 0.6
SEED = 0


def make_synthetic_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build a synthetic datacube with a linear velocity gradient.

    Returns
    -------
    A 4-tuple ``(wave, cube, noise_cube, truth)``: `wave` is the
    wavelength axis (microns), `cube`/`noise_cube` have shape
    ``(NX, NY, N_WAVE)``, and `truth` is a dictionary of the injected
    ``velocity``, ``dispersion``, ``flux`` and ``continuum`` maps
    (each of shape ``(NX, NY)``, except ``continuum`` which is scalar).
    """
    rng = np.random.default_rng(SEED)

    wave = LINE_CENTER - 0.05 + np.arange(N_WAVE) * CDELT_UM

    true_vel = np.linspace(-150.0, 150.0, NX)[:, None] * np.ones((1, NY))
    true_disp = np.full((NX, NY), TRUE_DISP_KMS)
    true_flux = np.full((NX, NY), TRUE_FLUX)

    cube = np.zeros((NX, NY, N_WAVE))
    noise_cube = np.full((NX, NY, N_WAVE), NOISE_LEVEL)

    velocity_axis = (wave - LINE_CENTER) / LINE_CENTER * C_KMS

    for ix in range(NX):
        for iy in range(NY):
            line_profile = gaussian_conv_model(
                velocity_axis,
                amplitude=true_flux[ix, iy],
                sigma_kms=true_disp[ix, iy],
                velocity_offset=true_vel[ix, iy],
            )
            cube[ix, iy, :] = (
                TRUE_CONT + line_profile
                + rng.normal(scale=NOISE_LEVEL, size=N_WAVE)
            )

    truth = dict(velocity=true_vel, dispersion=true_disp, flux=true_flux,
                 continuum=TRUE_CONT)
    return wave, cube, noise_cube, truth


def write_cube_to_fits(path: str, wave: np.ndarray, cube: np.ndarray,
                        noise_cube: np.ndarray) -> None:
    """Write a synthetic cube (and noise cube) to a FITS file.

    Parameters
    ----------
    path
        Output FITS file path.
    wave
        Wavelength axis (microns), used to populate CRVAL3/CDELT3.
    cube
        Data cube of shape ``(nx, ny, n_wave)``.
    noise_cube
        Noise cube of shape ``(nx, ny, n_wave)``.
    """
    data = np.transpose(cube, (2, 1, 0))       # -> (n_wave, ny, nx) for FITS
    noise = np.transpose(noise_cube, (2, 1, 0))

    hdr = fits.Header()
    hdr["NAXIS3"] = cube.shape[2]
    hdr["CRVAL3"] = float(wave[0])
    hdr["CDELT3"] = float(wave[1] - wave[0])
    hdr["CRPIX3"] = 1.0
    hdr["CRVAL1"] = 0.0
    hdr["CDELT1"] = 0.2
    hdr["CRPIX1"] = 1.0
    hdr["CUNIT1"] = "arcsec"
    hdr["CRVAL2"] = 0.0
    hdr["CDELT2"] = 0.2
    hdr["CRPIX2"] = 1.0
    hdr["CUNIT2"] = "arcsec"

    hdu0 = fits.PrimaryHDU(data.astype(np.float32), header=hdr)
    hdu1 = fits.ImageHDU(noise.astype(np.float32), name="NOISE")
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


def maybe_plot_spaxel(wave: np.ndarray, cube: np.ndarray, ix: int, iy: int,
                       line_window: tuple[float, float], line_center: float,
                       cont_windows: list[tuple[float, float]],
                       outfile: str) -> None:
    """Optionally plot one spaxel's spectrum and best-fit Gaussian model.

    Parameters
    ----------
    wave
        Wavelength axis (microns).
    cube
        Data cube of shape ``(nx, ny, n_wave)``.
    ix, iy
        Spaxel indices to plot.
    line_window
        ``(lo, hi)`` line-fitting window.
    line_center
        Rest/expected line wavelength.
    cont_windows
        Continuum windows used for the fit.
    outfile
        Output PNG path.

    Notes
    -----
    Silently does nothing if ``matplotlib`` is not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed -- skipping diagnostic plot.")
        return

    from linefit import fit_spectrum

    spec = cube[ix, iy, :]
    res = fit_spectrum(
        wave, spec, None, line_window, cont_windows, line_center,
        weight_type="uniform", return_model=True,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(wave, spec, color="k", lw=0.8, label="synthetic spectrum")
    ax.axvline(line_center, color="gray", ls="--", lw=0.8)
    if res["model_velocity"].size:
        model_wave = res["model_velocity"] / C_KMS * line_center + line_center
        model_flux = res["model_flux"] + res["cont"]
        ax.plot(model_wave, model_flux, color="crimson", lw=1.5, label="Gaussian fit")
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Flux [arb. units]")
    ax.set_title(f"Spaxel ({ix}, {iy})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    print(f"Saved diagnostic plot to '{outfile}'")


def run_test() -> None:
    """Build a synthetic cube, fit it, and check the recovered maps."""
    wave, cube, noise_cube, truth = make_synthetic_cube()

    with tempfile.TemporaryDirectory() as tmpdir:
        cube_path = os.path.join(tmpdir, "synthetic_cube.fits")
        write_cube_to_fits(cube_path, wave, cube, noise_cube)

        loaded_wave, loaded_cube, loaded_noise, header = load_cube(
            cube_path, data_ext=0, noise_ext=1,
        )
        assert np.allclose(loaded_wave, wave)
        assert loaded_cube.shape == (NX, NY, N_WAVE)

        line_window = (LINE_CENTER - 0.01, LINE_CENTER + 0.01)
        cont_windows = [
            (LINE_CENTER - 0.04, LINE_CENTER - 0.02),
            (LINE_CENTER + 0.02, LINE_CENTER + 0.04),
        ]

        rng = np.random.default_rng(SEED)
        maps = fit_cube(
            loaded_wave, loaded_cube, loaded_noise, line_window, cont_windows,
            LINE_CENTER, poly_order=0, weight_type="gauss",
            mom_threshold=0.5, instrument_fwhm_kms=0.0, n_mc=0, rng=rng,
        )

        cont_resid = np.abs(maps["cont"] - truth["continuum"])
        flux_resid = np.abs(maps["flux_gau"] - truth["flux"])
        vel_resid = np.abs(maps["vel_gau"] - truth["velocity"])
        disp_resid = np.abs(maps["disp_gau"] - truth["dispersion"])

        print("Continuum residual (max):  ", np.nanmax(cont_resid))
        print("Flux residual (max):       ", np.nanmax(flux_resid))
        print("Velocity residual (max):   ", np.nanmax(vel_resid))
        print("Dispersion residual (max): ", np.nanmax(disp_resid))

        assert np.nanmax(cont_resid) < 1.0, "continuum recovery out of tolerance"
        assert np.nanmax(flux_resid) < 0.25 * TRUE_FLUX, "flux recovery out of tolerance"
        assert np.nanmax(vel_resid) < 20.0, "velocity recovery out of tolerance"
        assert np.nanmax(disp_resid) < 25.0, "dispersion recovery out of tolerance"
        print("All checks passed.")

        maybe_plot_spaxel(
            loaded_wave, loaded_cube, ix=NX // 2, iy=NY // 2,
            line_window=line_window, line_center=LINE_CENTER,
            cont_windows=cont_windows, outfile="test_spaxel_fit.png",
        )


if __name__ == "__main__":
    run_test()
