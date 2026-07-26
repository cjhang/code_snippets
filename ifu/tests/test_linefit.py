#!/usr/bin/env python3
"""
test_linefit.py
====================

Demonstration/smoke test of :mod:`linefit` on a synthetic 3-D IFU
datacube with a known velocity gradient, dispersion, flux and
continuum level.

In addition to fitting a single spaxel, this script fits the **entire**
cube and writes the following persistent outputs to ``OUTPUT_DIR``:

- ``synthetic_cube.fits`` -- the synthetic input cube (+ noise
  extension).
- One FITS file per fitted quantity (``cont.fits``, ``flux_gau.fits``,
  ``vel_gau.fits``, ``disp_gau.fits``, etc.), written by
  :func:`linefit.save_maps`.
- ``model_cube.fits`` -- best-fit model cube (continuum + Gaussian),
  reconstructed from the fitted maps and evaluated on the full
  wavelength axis.
- ``residual_cube.fits`` -- ``data - model``.
- ``test_spaxel_fit.png`` -- diagnostic plot of one spaxel (if
  ``matplotlib`` is installed).

Run directly::

    $ python test_linefit.py
"""

from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from linefit import (
    C_KMS,
    fit_cube,
    fit_spectrum,
    gaussian_conv_model,
    load_cube,
    make_gaussian_kernel,
    save_maps,
)

LINE_CENTER = 2.1655    # microns, arbitrary (e.g. redshifted Halpha in K-band)
CDELT_UM = 0.0005        # spectral pixel scale, microns
N_WAVE = 240
NX, NY = 5, 5
TRUE_CONT = 10.0
TRUE_DISP_KMS = 80.0
TRUE_FLUX = 60.0
NOISE_LEVEL = 0.6
SEED = 0

OUTPUT_DIR = "test_linefit_output"


def make_synthetic_cube() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Build a synthetic datacube with a linear velocity gradient.

    Returns
    -------
    A 4-tuple ``(wave, cube, noise_cube, truth)``: `wave` is the
    wavelength axis (microns), `cube`/`noise_cube` have shape
    ``(NX, NY, N_WAVE)``, and `truth` is a dictionary of the injected
    ``velocity``, ``dispersion``, ``flux`` (each of shape
    ``(NX, NY)``) and ``continuum`` (scalar) maps.
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
                        noise_cube: np.ndarray | None = None) -> fits.Header:
    """Write a datacube (and optional noise cube) to a FITS file.

    Parameters
    ----------
    path
        Output FITS file path.
    wave
        Wavelength axis (microns), used to populate CRVAL3/CDELT3.
    cube
        Data cube of shape ``(nx, ny, n_wave)``.
    noise_cube
        Optional noise cube of shape ``(nx, ny, n_wave)``, written as
        a second HDU named ``"NOISE"``.

    Returns
    -------
    The primary FITS header that was written (useful for propagating
    spatial WCS keywords to derived products).
    """
    data = np.transpose(cube, (2, 1, 0))  # -> (n_wave, ny, nx) for FITS

    # NOTE: do *not* pass NAXIS* keywords in via `header=`; astropy
    # derives NAXIS/NAXIS1/NAXIS2/NAXIS3 automatically from `data` and
    # will raise a VerifyError if a manually-set NAXIS3 card conflicts
    # with the auto-generated structural keywords. Add custom WCS
    # keywords to `hdu.header` *after* construction instead.
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    hdu.header["CRVAL3"] = float(wave[0])
    hdu.header["CDELT3"] = float(wave[1] - wave[0])
    hdu.header["CRPIX3"] = 1.0
    hdu.header["CRVAL1"] = 0.0
    hdu.header["CDELT1"] = 0.2
    hdu.header["CRPIX1"] = 1.0
    hdu.header["CUNIT1"] = "arcsec"
    hdu.header["CRVAL2"] = 0.0
    hdu.header["CDELT2"] = 0.2
    hdu.header["CRPIX2"] = 1.0
    hdu.header["CUNIT2"] = "arcsec"

    hdus = [hdu]
    if noise_cube is not None:
        noise = np.transpose(noise_cube, (2, 1, 0))
        hdus.append(fits.ImageHDU(noise.astype(np.float32), name="NOISE"))

    fits.HDUList(hdus).writeto(path, overwrite=True)
    return hdu.header


def save_cube_to_fits(path: str, wave: np.ndarray, cube: np.ndarray,
                       header_template: fits.Header | None = None) -> None:
    """Write a single-extension model/residual cube to a FITS file.

    Parameters
    ----------
    path
        Output FITS file path.
    wave
        Wavelength axis (microns), used to populate CRVAL3/CDELT3.
    cube
        Cube of shape ``(nx, ny, n_wave)`` to write (e.g. a
        reconstructed model or a residual cube).
    header_template
        Optional header from which spatial WCS keywords are copied.
    """
    data = np.transpose(cube, (2, 1, 0))  # -> (n_wave, ny, nx) for FITS

    # As above: let astropy set NAXIS* from `data`, then append the
    # custom WCS keywords afterwards.
    hdu = fits.PrimaryHDU(data.astype(np.float32))
    if header_template is not None:
        for key in ("CRVAL1", "CDELT1", "CRPIX1", "CUNIT1",
                    "CRVAL2", "CDELT2", "CRPIX2", "CUNIT2"):
            if key in header_template:
                hdu.header[key] = header_template[key]
    hdu.header["CRVAL3"] = float(wave[0])
    hdu.header["CDELT3"] = float(wave[1] - wave[0])
    hdu.header["CRPIX3"] = 1.0

    hdu.writeto(path, overwrite=True)


def reconstruct_model_cube(
    wave: np.ndarray,
    maps: dict[str, np.ndarray],
    line_center: float,
    instrument_fwhm_kms: float = 0.0,
) -> np.ndarray:
    """Reconstruct a best-fit model cube from the fitted per-spaxel maps.

    For every spaxel, the model spectrum is the fitted (scalar)
    continuum level plus a Gaussian of the fitted flux, dispersion and
    velocity, evaluated on the **full** wavelength axis (not just the
    line-fitting window used internally by :func:`linefit.fit_spectrum`).

    Parameters
    ----------
    wave
        Full wavelength axis (microns).
    maps
        Dictionary of fitted maps, as returned by
        :func:`linefit.fit_cube` (must contain ``cont``,
        ``flux_gau``, ``vel_gau``, ``disp_gau``).
    line_center
        Rest/expected wavelength of the line, used to build the
        velocity axis.
    instrument_fwhm_kms
        If greater than zero, the reconstructed Gaussian is convolved
        with a fixed instrumental kernel of this FWHM (km/s), matching
        the convolution (if any) used during the fit.

    Returns
    -------
    Model cube of shape ``(nx, ny, n_wave)``. Spaxels where the fit
    failed (non-finite parameters) are set to ``NaN``.
    """
    nx, ny = maps["cont"].shape
    n_wave = wave.size
    velocity_full = (wave - line_center) / line_center * C_KMS

    kernel = None
    if instrument_fwhm_kms > 0:
        offsets = velocity_full - velocity_full[n_wave // 2]
        kernel = make_gaussian_kernel(offsets, instrument_fwhm_kms)

    model_cube = np.full((nx, ny, n_wave), np.nan)
    for ix in range(nx):
        for iy in range(ny):
            cont = maps["cont"][ix, iy]
            flux = maps["flux_gau"][ix, iy]
            vel = maps["vel_gau"][ix, iy]
            disp = maps["disp_gau"][ix, iy]
            if not np.all(np.isfinite([cont, flux, vel, disp])):
                continue
            line = gaussian_conv_model(velocity_full, flux, disp, vel, kernel)
            model_cube[ix, iy, :] = cont + line
    return model_cube


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


def run_test(outdir: str = OUTPUT_DIR) -> None:
    """Build a synthetic cube, fit it in full, and save all outputs.

    Parameters
    ----------
    outdir
        Directory in which the input cube, fitted maps, model cube,
        residual cube and diagnostic plot are written. Created if it
        does not exist.
    """
    os.makedirs(outdir, exist_ok=True)

    wave, cube, noise_cube, truth = make_synthetic_cube()

    cube_path = os.path.join(outdir, "synthetic_cube.fits")
    write_cube_to_fits(cube_path, wave, cube, noise_cube)
    print(f"Wrote synthetic input cube to '{cube_path}'")

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

    # ---- Fit the *entire* cube --------------------------------------
    rng = np.random.default_rng(SEED)
    maps = fit_cube(
        loaded_wave, loaded_cube, loaded_noise, line_window, cont_windows,
        LINE_CENTER, poly_order=0, weight_type="gauss",
        mom_threshold=0.5, instrument_fwhm_kms=0.0, n_mc=0, rng=rng,
        verbose=True,
    )

    # ---- Save fitted maps --------------------------------------------
    save_maps(outdir, maps, header=header)
    print(f"Wrote {len(maps)} fitted-map FITS files to '{outdir}'")

    # ---- Reconstruct + save best-fit model cube and residual cube ---
    model_cube = reconstruct_model_cube(loaded_wave, maps, LINE_CENTER,
                                         instrument_fwhm_kms=0.0)
    residual_cube = loaded_cube - model_cube

    model_path = os.path.join(outdir, "model_cube.fits")
    residual_path = os.path.join(outdir, "residual_cube.fits")
    save_cube_to_fits(model_path, loaded_wave, model_cube, header_template=header)
    save_cube_to_fits(residual_path, loaded_wave, residual_cube, header_template=header)
    print(f"Wrote best-fit model cube to '{model_path}'")
    print(f"Wrote residual cube to '{residual_path}'")

    # ---- Sanity checks against injected truth ------------------------
    cont_resid = np.abs(maps["cont"] - truth["continuum"])
    flux_resid = np.abs(maps["flux_gau"] - truth["flux"])
    vel_resid = np.abs(maps["vel_gau"] - truth["velocity"])
    disp_resid = np.abs(maps["disp_gau"] - truth["dispersion"])
    residual_rms = float(np.sqrt(np.nanmean(residual_cube ** 2)))

    print("Continuum residual (max):  ", np.nanmax(cont_resid))
    print("Flux residual (max):       ", np.nanmax(flux_resid))
    print("Velocity residual (max):   ", np.nanmax(vel_resid))
    print("Dispersion residual (max): ", np.nanmax(disp_resid))
    print("Residual cube RMS:         ", residual_rms)

    assert np.nanmax(cont_resid) < 1.0, "continuum recovery out of tolerance"
    assert np.nanmax(flux_resid) < 0.25 * TRUE_FLUX, "flux recovery out of tolerance"
    assert np.nanmax(vel_resid) < 20.0, "velocity recovery out of tolerance"
    assert np.nanmax(disp_resid) < 25.0, "dispersion recovery out of tolerance"
    assert residual_rms < 2.0 * NOISE_LEVEL, "residual cube RMS out of tolerance"
    print("All checks passed.")

    # ---- Optional single-spaxel diagnostic plot ----------------------
    maybe_plot_spaxel(
        loaded_wave, loaded_cube, ix=NX // 2, iy=NY // 2,
        line_window=line_window, line_center=LINE_CENTER,
        cont_windows=cont_windows,
        outfile=os.path.join(outdir, "test_spaxel_fit.png"),
    )


if __name__ == "__main__":
    run_test()
