import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
from scipy.ndimage import shift as scipy_shift
from scipy.interpolate import interp1d
from skimage.registration import phase_cross_correlation


def read_cube(path, ext=0, dtype=np.float32):
    with fits.open(path, memmap=False) as hdul:
        primary_header = hdul[0].header.copy()
        cube = np.asarray(hdul[ext].data, dtype=dtype)

        header = primary_header.copy()
        if ext != 0:
            header.extend(hdul[ext].header, update=True)

    if cube.ndim != 3:
        raise ValueError(f"{path} is not a 3D cube. Found shape {cube.shape}")

    return cube, header


def get_wavelength_axis(header, n_wave):
    """
    Simple wavelength-axis reader for common ESO-style cubes.

    Assumes numpy cube order:
        cube[wavelength, y, x]

    and FITS spectral axis:
        axis 3
    """
    if all(k in header for k in ["CRVAL3", "CRPIX3", "CDELT3"]):
        pix = np.arange(n_wave) + 1.0
        wave = header["CRVAL3"] + (pix - header["CRPIX3"]) * header["CDELT3"]
        unit = u.Unit(header.get("CUNIT3", ""))
        return wave * unit

    raise ValueError("Could not read wavelength axis from CRVAL3/CRPIX3/CDELT3.")


def make_white_image(cube, wave=None, wave_range=None, method="median"):
    """
    Collapse cube along wavelength to make a registration image.

    Parameters
    ----------
    cube : ndarray
        Shape (n_wave, ny, nx)

    wave : Quantity or None
        Wavelength array.

    wave_range : tuple or None
        Optional wavelength interval, e.g. (6500*u.AA, 7500*u.AA)

    method : {"median", "mean", "sum"}
        Collapse method.
    """
    data = cube

    if wave_range is not None:
        if wave is None:
            raise ValueError("wave must be supplied if wave_range is used.")

        w0, w1 = wave_range
        w0 = w0.to(wave.unit)
        w1 = w1.to(wave.unit)

        mask = (wave >= w0) & (wave <= w1)
        if not np.any(mask):
            raise ValueError("No wavelength planes inside requested wave_range.")

        data = cube[mask]

    if method == "median":
        img = np.nanmedian(data, axis=0)
    elif method == "mean":
        img = np.nanmean(data, axis=0)
    elif method == "sum":
        img = np.nansum(data, axis=0)
    else:
        raise ValueError("method must be 'median', 'mean', or 'sum'.")

    img = np.asarray(img, dtype=float)

    # Replace NaNs for cross-correlation.
    med = np.nanmedian(img)
    img = np.where(np.isfinite(img), img, med)

    # Remove large DC background.
    img -= np.nanmedian(img)

    return img


def measure_pixel_shift(reference_image, moving_image, upsample_factor=20):
    """
    Measure shift needed to align moving_image to reference_image.

    Returns
    -------
    shift_y, shift_x : float
        Pixel shift to apply to moving_image.
    """
    shift_yx, error, diffphase = phase_cross_correlation(
        reference_image,
        moving_image,
        upsample_factor=upsample_factor,
    )

    shift_y, shift_x = shift_yx

    return float(shift_y), float(shift_x)


def construct_common_wavelength_grid(
    wavelength_arrays,
    reference_index=0,
    mode="reference",
    wave_step=None,
):
    ref_wave = wavelength_arrays[reference_index]
    ref_unit = ref_wave.unit

    waves = [w.to(ref_unit).value for w in wavelength_arrays]
    ref_values = ref_wave.to(ref_unit).value

    if mode == "reference":
        return ref_wave

    if wave_step is None:
        dw = np.median(np.diff(np.sort(ref_values)))
    else:
        if isinstance(wave_step, u.Quantity):
            dw = wave_step.to(ref_unit).value
        else:
            dw = float(wave_step)

    if mode == "intersection":
        wmin = max(np.nanmin(w) for w in waves)
        wmax = min(np.nanmax(w) for w in waves)
    elif mode == "union":
        wmin = min(np.nanmin(w) for w in waves)
        wmax = max(np.nanmax(w) for w in waves)
    else:
        raise ValueError("mode must be 'reference', 'intersection', or 'union'.")

    if wmax <= wmin:
        raise ValueError("No overlapping wavelength range.")

    common = np.arange(wmin, wmax + 0.5 * dw, dw)

    return common * ref_unit


def combine_cubes_with_pixel_shifts(
    files,
    exposure_times,
    output_path=None,
    ext=0,
    reference_index=0,
    wavelength_grid="reference",
    wave_step=None,
    wave_range_for_shift=None,
    collapse_method="median",
    upsample_factor=20,
    shift_order=1,
    spectral_interp_kind="linear",
    dtype=np.float32,
    overwrite=True,
):
    """
    Combine 3D datacubes using empirical pixel-shift alignment.

    This is faster than full WCS reprojection and is appropriate when cubes
    have the same pixel scale and orientation.

    Parameters
    ----------
    files : list of str
        Input datacube FITS files.

    exposure_times : array-like
        True exposure times in seconds.

    output_path : str or None
        Output FITS filename.

    ext : int
        FITS extension containing the cube.

    reference_index : int
        Reference cube index.

    wavelength_grid : {"reference", "intersection", "union"}
        Output wavelength-grid choice.

    wave_step : float, Quantity, or None
        Wavelength step for intersection/union grids.

    wave_range_for_shift : tuple or None
        Optional wavelength interval used to make white-light registration image.
        Example:
            (6500*u.AA, 7500*u.AA)

    collapse_method : {"median", "mean", "sum"}
        How to collapse the cube for shift measurement.

    upsample_factor : int
        Sub-pixel precision for phase cross-correlation.
        20 means approximately 0.05 pixel precision.

    shift_order : int
        Interpolation order for spatial shifts.
        1 is linear and fast.
        3 is cubic but slower and can introduce ringing.

    spectral_interp_kind : str
        Interpolation method for wavelength alignment.

    dtype : numpy dtype
        Data type used internally.

    Returns
    -------
    result : dict
        Contains combined_cube, weight_cube, wavelength, shifts, header.
    """

    files = list(files)
    exposure_times = np.asarray(exposure_times, dtype=float)

    if len(files) != len(exposure_times):
        raise ValueError("files and exposure_times must have the same length.")

    if np.any(exposure_times <= 0):
        raise ValueError("Exposure times must be positive.")

    cubes = []
    headers = []
    waves = []

    for path in files:
        cube, header = read_cube(path, ext=ext, dtype=dtype)
        wave = get_wavelength_axis(header, cube.shape[0])

        cubes.append(cube)
        headers.append(header)
        waves.append(wave)

    ref_cube = cubes[reference_index]
    ref_wave = waves[reference_index]
    ref_header = headers[reference_index]

    n_ref_wave, ny, nx = ref_cube.shape

    common_wave = construct_common_wavelength_grid(
        waves,
        reference_index=reference_index,
        mode=wavelength_grid,
        wave_step=wave_step,
    )

    common_wave_values = common_wave.value
    wave_unit = common_wave.unit

    output_shape = (len(common_wave), ny, nx)

    numerator = np.zeros(output_shape, dtype=np.float64)
    denominator = np.zeros(output_shape, dtype=np.float64)

    ref_white = make_white_image(
        ref_cube,
        wave=ref_wave,
        wave_range=wave_range_for_shift,
        method=collapse_method,
    )

    shifts = []

    for i, path in enumerate(files):
        cube = cubes[i]
        header = headers[i]
        wave = waves[i].to(wave_unit)

        if cube.shape[1:] != (ny, nx):
            raise ValueError(
                f"{path} has spatial shape {cube.shape[1:]}, "
                f"but reference shape is {(ny, nx)}. "
                "Pixel-shift alignment requires the same spatial shape."
            )

        if i == reference_index:
            shift_y, shift_x = 0.0, 0.0
        else:
            moving_white = make_white_image(
                cube,
                wave=waves[i],
                wave_range=wave_range_for_shift,
                method=collapse_method,
            )

            shift_y, shift_x = measure_pixel_shift(
                ref_white,
                moving_white,
                upsample_factor=upsample_factor,
            )

        shifts.append(
            {
                "file": path,
                "shift_y_pix": shift_y,
                "shift_x_pix": shift_x,
            }
        )

        # Apply the same spatial shift to every wavelength plane.
        shifted_cube = scipy_shift(
            cube,
            shift=(0.0, shift_y, shift_x),
            order=shift_order,
            mode="constant",
            cval=np.nan,
            prefilter=(shift_order > 1),
        )

        # Spectral interpolation.
        wave_values = wave.value
        sort_idx = np.argsort(wave_values)

        wave_sorted = wave_values[sort_idx]
        cube_sorted = shifted_cube[sort_idx, :, :]

        interpolator = interp1d(
            wave_sorted,
            cube_sorted,
            axis=0,
            kind=spectral_interp_kind,
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )

        cube_on_common_wave = interpolator(common_wave_values)

        valid = np.isfinite(cube_on_common_wave)
        weight = exposure_times[i]

        numerator += np.where(valid, cube_on_common_wave * weight, 0.0)
        denominator += np.where(valid, weight, 0.0)

    combined_cube = np.full(output_shape, np.nan, dtype=dtype)
    good = denominator > 0
    combined_cube[good] = numerator[good] / denominator[good]

    weight_cube = denominator.astype(dtype)

    output_header = ref_header.copy()
    output_header["NAXIS"] = 3
    output_header["NAXIS1"] = nx
    output_header["NAXIS2"] = ny
    output_header["NAXIS3"] = len(common_wave)

    output_header["CRPIX3"] = 1.0
    output_header["CRVAL3"] = float(common_wave_values[0])
    if len(common_wave_values) > 1:
        output_header["CDELT3"] = float(np.median(np.diff(common_wave_values)))
    output_header["CUNIT3"] = str(wave_unit)

    output_header["NCOMBINE"] = len(files)
    output_header["COMBMETH"] = "EXPTIME_WEIGHTED_MEAN"
    output_header["ALIGN"] = "PIXEL_SHIFT"
    output_header["EXPTSUM"] = float(np.sum(exposure_times))
    output_header["HISTORY"] = "Cubes aligned by empirical spatial pixel shifts."
    output_header["HISTORY"] = "Cubes interpolated to a common wavelength grid."

    if output_path is not None:
        primary_hdu = fits.PrimaryHDU(
            data=combined_cube,
            header=output_header,
        )

        weight_hdu = fits.ImageHDU(
            data=weight_cube,
            name="WEIGHT",
        )

        wave_hdu = fits.ImageHDU(
            data=common_wave_values.astype(np.float64),
            name="WAVELENGTH",
        )
        wave_hdu.header["BUNIT"] = str(wave_unit)

        fits.HDUList([primary_hdu, weight_hdu, wave_hdu]).writeto(
            output_path,
            overwrite=overwrite,
        )

    return {
        "combined_cube": combined_cube,
        "weight_cube": weight_cube,
        "wavelength": common_wave,
        "shifts": shifts,
        "header": output_header,
    }
