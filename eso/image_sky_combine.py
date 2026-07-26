import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.stats import sigma_clip as astropy_sigma_clip


def _read_fits_image(path, ext=0):
    """
    Read image data and merge primary + extension headers.
    ESO files often store important pointing keywords in the primary header.
    """
    with fits.open(path, memmap=False) as hdul:
        primary_header = hdul[0].header.copy()

        if hdul[ext].data is None:
            raise ValueError(f"No image data found in extension {ext} of {path}")

        data = np.asarray(hdul[ext].data, dtype=float)

        header = primary_header.copy()
        if ext != 0:
            header.extend(hdul[ext].header, update=True)

    if data.ndim != 2:
        raise ValueError(f"{path} does not contain a 2D image in extension {ext}")

    return data, header


def _has_valid_celestial_wcs(header):
    try:
        w = WCS(header)
        return w.has_celestial
    except Exception:
        return False


def _coord_from_header(header):
    """
    Parse RA/DEC from the FITS header.

    Assumes numerical RA/DEC are in degrees.
    If RA is a sexagesimal string, assumes hour angle.
    """
    if "RA" not in header or "DEC" not in header:
        raise ValueError("Header does not contain RA and DEC keywords.")

    ra = header["RA"]
    dec = header["DEC"]

    try:
        return SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    except Exception:
        return SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg), frame="icrs")


def combine_vlt_fits(
    files,
    exposure_times,
    output_path=None,
    ext=0,
    reference_index=0,
    method="auto",
    pixel_scale_arcsec=None,
    sigma_clip=False,
    sigma=4.0,
    interp_order="bilinear",
    overwrite=True,
):
    """
    Align and combine astronomical FITS images with exposure-time weighting.

    Parameters
    ----------
    files : list of str
        Input FITS files.

    exposure_times : list or array
        True exposure times for each input image, in seconds.
        These are used as weights.

    output_path : str or None
        If given, write the combined FITS file to this path.

    ext : int
        FITS extension containing the science image.

    reference_index : int
        Index of the image defining the output pixel grid.

    method : {"auto", "wcs", "radec_shift"}
        Alignment method.

        "wcs":
            Use FITS WCS and reproject images to the reference frame.
            Recommended if valid WCS keywords are present.

        "radec_shift":
            Use only RA/DEC pointing offsets.
            Requires pixel_scale_arcsec.
            Assumes identical pixel scale, orientation, and no rotation.

        "auto":
            Use WCS if all files have valid celestial WCS, otherwise fall back
            to RA/DEC shifts if pixel_scale_arcsec is supplied.

    pixel_scale_arcsec : float or None
        Pixel scale in arcsec/pixel, required for method="radec_shift".

    sigma_clip : bool
        If True, perform sigma clipping along the image stack before combining.
        Useful for cosmic rays, satellite trails, etc.

    sigma : float
        Sigma threshold for sigma clipping.

    interp_order : str or int
        Interpolation order for WCS reprojection. For `reproject`, common values
        are "nearest-neighbor", "bilinear", "biquadratic", "bicubic".

    overwrite : bool
        Overwrite output FITS file if it exists.

    Returns
    -------
    result : dict
        Dictionary containing:

        combined : 2D ndarray
            Exposure-time-weighted combined image.

        weight_map : 2D ndarray
            Sum of valid exposure-time weights contributing to each pixel.

        header : fits.Header
            Output FITS header.

        offsets : list of dict
            Approximate pointing offsets relative to the reference image.

        method : str
            Alignment method actually used.
    """

    files = list(files)
    exposure_times = np.asarray(exposure_times, dtype=float)

    if len(files) == 0:
        raise ValueError("No input files supplied.")

    if len(files) != len(exposure_times):
        raise ValueError("files and exposure_times must have the same length.")

    if np.any(exposure_times <= 0):
        raise ValueError("All exposure times must be positive.")

    images = []
    headers = []

    for path in files:
        data, header = _read_fits_image(path, ext=ext)
        images.append(data)
        headers.append(header)

    ref_image = images[reference_index]
    ref_header = headers[reference_index]
    ny, nx = ref_image.shape

    output_header = ref_header.copy()
    output_header["NAXIS"] = 2
    output_header["NAXIS1"] = nx
    output_header["NAXIS2"] = ny

    all_have_wcs = all(_has_valid_celestial_wcs(h) for h in headers)

    if method == "auto":
        if all_have_wcs:
            method_used = "wcs"
        else:
            method_used = "radec_shift"
    else:
        method_used = method

    aligned_images = []
    footprints = []
    offsets = []

    # ------------------------------------------------------------
    # WCS-based alignment
    # ------------------------------------------------------------
    if method_used == "wcs":
        if not all_have_wcs:
            raise ValueError(
                "method='wcs' requested, but not all input images have valid "
                "celestial WCS information."
            )

        try:
            from reproject import reproject_interp
        except ImportError as exc:
            raise ImportError(
                "WCS alignment requires the 'reproject' package. "
                "Install with: pip install reproject"
            ) from exc

        ref_wcs = WCS(output_header).celestial
        ref_xc = (nx - 1) / 2.0
        ref_yc = (ny - 1) / 2.0

        ref_coord = _coord_from_header(ref_header)

        for path, image, header in zip(files, images, headers):
            reproj, footprint = reproject_interp(
                (image, header),
                output_header,
                shape_out=(ny, nx),
                order=interp_order,
            )

            aligned_images.append(reproj)
            footprints.append(footprint > 0)

            coord = _coord_from_header(header)

            dra_arcsec = (
                (coord.ra.deg - ref_coord.ra.deg)
                * np.cos(ref_coord.dec.radian)
                * 3600.0
            )
            ddec_arcsec = (coord.dec.deg - ref_coord.dec.deg) * 3600.0

            try:
                x_ref, y_ref = ref_wcs.world_to_pixel(coord)
                dx_pix = x_ref - ref_xc
                dy_pix = y_ref - ref_yc
            except Exception:
                dx_pix = np.nan
                dy_pix = np.nan

            offsets.append(
                {
                    "file": path,
                    "dra_arcsec": dra_arcsec,
                    "ddec_arcsec": ddec_arcsec,
                    "x_pointing_in_reference_pix": dx_pix,
                    "y_pointing_in_reference_pix": dy_pix,
                }
            )

    # ------------------------------------------------------------
    # RA/DEC shift-only alignment
    # ------------------------------------------------------------
    elif method_used == "radec_shift":
        if pixel_scale_arcsec is None:
            raise ValueError(
                "RA/DEC-only alignment requires pixel_scale_arcsec. "
                "RA and DEC alone are not enough to determine pixel shifts "
                "unless the pixel scale and orientation are known."
            )

        try:
            from scipy.ndimage import shift as scipy_shift
        except ImportError as exc:
            raise ImportError(
                "RA/DEC shift alignment requires scipy. "
                "Install with: pip install scipy"
            ) from exc

        ref_coord = _coord_from_header(ref_header)

        for path, image, header in zip(files, images, headers):
            coord = _coord_from_header(header)

            dra_arcsec = (
                (coord.ra.deg - ref_coord.ra.deg)
                * np.cos(ref_coord.dec.radian)
                * 3600.0
            )
            ddec_arcsec = (coord.dec.deg - ref_coord.dec.deg) * 3600.0

            # Assumption:
            # y increases northward and x increases westward, as in many
            # north-up astronomical images. This may not hold for all data.
            shift_y = ddec_arcsec / pixel_scale_arcsec
            shift_x = -dra_arcsec / pixel_scale_arcsec

            shifted = scipy_shift(
                image,
                shift=(shift_y, shift_x),
                order=1,
                mode="constant",
                cval=np.nan,
                prefilter=False,
            )

            footprint = scipy_shift(
                np.ones_like(image, dtype=float),
                shift=(shift_y, shift_x),
                order=0,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ) > 0.5

            aligned_images.append(shifted)
            footprints.append(footprint)

            offsets.append(
                {
                    "file": path,
                    "dra_arcsec": dra_arcsec,
                    "ddec_arcsec": ddec_arcsec,
                    "applied_shift_x_pix": shift_x,
                    "applied_shift_y_pix": shift_y,
                }
            )

    else:
        raise ValueError("method must be one of 'auto', 'wcs', or 'radec_shift'.")

    stack = np.asarray(aligned_images, dtype=float)
    footprint_stack = np.asarray(footprints, dtype=bool)

    finite = np.isfinite(stack)
    valid = footprint_stack & finite

    weights = exposure_times[:, None, None]

    if sigma_clip:
        masked_stack = np.ma.array(stack, mask=~valid)
        clipped = astropy_sigma_clip(
            masked_stack,
            sigma=sigma,
            axis=0,
            masked=True,
        )
        valid = ~clipped.mask & finite & footprint_stack

    numerator = np.sum(np.where(valid, stack * weights, 0.0), axis=0)
    denominator = np.sum(np.where(valid, weights, 0.0), axis=0)

    combined = np.full((ny, nx), np.nan, dtype=float)
    good = denominator > 0
    combined[good] = numerator[good] / denominator[good]

    weight_map = denominator

    output_header["NCOMBINE"] = len(files)
    output_header["COMBMETH"] = "EXPTIME_WEIGHTED_MEAN"
    output_header["EXPTSUM"] = float(np.sum(exposure_times))
    output_header["ALIGN"] = method_used
    output_header["HISTORY"] = "Images aligned and combined with exposure-time weights."

    if sigma_clip:
        output_header["SIGCLIP"] = True
        output_header["CLIPSIG"] = float(sigma)

    if output_path is not None:
        primary_hdu = fits.PrimaryHDU(data=combined, header=output_header)
        weight_hdu = fits.ImageHDU(data=weight_map, name="WEIGHT")
        fits.HDUList([primary_hdu, weight_hdu]).writeto(
            output_path,
            overwrite=overwrite,
        )

    return {
        "combined": combined,
        "weight_map": weight_map,
        "header": output_header,
        "offsets": offsets,
        "method": method_used,
    }
