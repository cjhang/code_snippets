#!/usr/bin/env python3

"""
Interactive datacube and fitted-cube viewer.

This program displays a 2-D map on the left and spectra on the right. It can be
used as:

1. A simple datacube viewer:

       python fit_cube_viewer.py --data datacube.fits

2. A data-vs-fit comparison viewer:

       python fit_cube_viewer.py --data datacube.fits --fitted fitted_cube.fits

3. A demo/debug viewer with synthetic data:

       python fit_cube_viewer.py --demo

Expected array shapes
---------------------
data_cube:
    [nwave, ny, nx]

fitted_cube:
    [nwave, ny, nx]

maps:
    [ny, nx] or [nmap, ny, nx]

GUI controls
------------
Left image panel:
    Normal left-click drag:
        Zoom into selected image region.

    Double left-click:
        Reset image zoom.

    Command-click / Ctrl-click:
        Toggle individual pixels into the combined spectrum selection.

    Command-drag / Ctrl-drag:
        Select a rectangular spatial region for spectrum extraction.

    Normal single click:
        Does nothing.

Right spectrum panel:
    Normal left-click drag:
        Zoom into selected spectral region.

    Double left-click:
        Reset spectrum zoom.

    Reset spec zoom button:
        Reset spectrum zoom.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, RectangleSelector

from astropy.io import fits
from astropy.io.fits import Header
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval
import astropy.units as u
from astropy.units import Quantity


ArrayLike = Union[np.ndarray, Quantity]
PathLike = Union[str, Path]


def read_fits_data_and_header(
    filename: PathLike,
    ext: int | str = 0,
) -> tuple[np.ndarray, Header]:
    """
    Read image data and header from a FITS file.

    Parameters
    -----------
    filename : the FITS file path
    ext : the FITS extension name or index

    Returns
    --------
    data : the image data array
    header : the FITS header
    """
    with fits.open(filename) as hdul:
        data = hdul[ext].data
        header = hdul[ext].header.copy()

    if data is None:
        raise ValueError(f"No data found in extension {ext!r} of {filename}")

    return np.asarray(data), header


def parse_fits_ext(ext: str) -> int | str:
    """
    Parse a FITS extension argument.

    Parameters
    -----------
    ext : the extension string from the command line

    Returns
    --------
    parsed_ext : the extension as an integer if possible, otherwise as a string
    """
    try:
        return int(ext)
    except ValueError:
        return ext


def read_wavelength_from_header(
    header: Header,
    nwave: int,
    spectral_axis: int = 3,
) -> Quantity | np.ndarray:
    """
    Construct a wavelength array from a FITS header.

    Parameters
    -----------
    header : the FITS header containing spectral WCS keywords
    nwave : the number of wavelength pixels
    spectral_axis : the FITS axis number corresponding to the wavelength axis

    Returns
    --------
    wavelength : the wavelength array, with units if CUNIT is available
    """
    crval_key = f"CRVAL{spectral_axis}"
    crpix_key = f"CRPIX{spectral_axis}"
    cdelt_key = f"CDELT{spectral_axis}"
    cd_key = f"CD{spectral_axis}_{spectral_axis}"
    cunit_key = f"CUNIT{spectral_axis}"

    missing: list[str] = []

    if crval_key not in header:
        missing.append(crval_key)

    if crpix_key not in header:
        missing.append(crpix_key)

    if cdelt_key not in header and cd_key not in header:
        missing.append(f"{cdelt_key} or {cd_key}")

    if missing:
        raise KeyError(
            "Cannot construct wavelength from FITS header. Missing: "
            + ", ".join(missing)
        )

    crval = float(header[crval_key])
    crpix = float(header[crpix_key])
    cdelt = float(header[cdelt_key]) if cdelt_key in header else float(header[cd_key])

    pixels = np.arange(nwave, dtype=float)
    wavelength = crval + (pixels + 1.0 - crpix) * cdelt

    if cunit_key in header:
        unit_string = str(header[cunit_key]).strip()

        if unit_string:
            try:
                return wavelength * u.Unit(unit_string)
            except Exception:
                warnings.warn(
                    f"Could not parse wavelength unit {unit_string!r}; "
                    "returning wavelength without units.",
                    RuntimeWarning,
                )

    return wavelength


def load_wavelength(
    filename: Optional[PathLike],
    header: Optional[Header],
    nwave: int,
) -> Quantity | np.ndarray:
    """
    Load or construct a wavelength array.

    Parameters
    -----------
    filename : the optional wavelength file
    header : the optional FITS header used to construct the wavelength
    nwave : the expected wavelength length

    Returns
    --------
    wavelength : the wavelength array
    """
    if filename is not None:
        path = Path(filename)
        suffix = path.suffix.lower()

        if suffix == ".npy":
            wavelength: Quantity | np.ndarray = np.load(path)
        elif suffix in {".fits", ".fit", ".fts"}:
            wavelength, _ = read_fits_data_and_header(path)
        else:
            wavelength = np.loadtxt(path)

    elif header is not None:
        try:
            wavelength = read_wavelength_from_header(header=header, nwave=nwave)
        except Exception as exc:
            warnings.warn(
                "Could not read wavelength from the FITS header. "
                f"Using spectral pixel index instead. Reason: {exc}",
                RuntimeWarning,
            )
            wavelength = np.arange(nwave, dtype=float)

    else:
        warnings.warn(
            "No wavelength file or FITS header was supplied. "
            "Using spectral pixel index.",
            RuntimeWarning,
        )
        wavelength = np.arange(nwave, dtype=float)

    if isinstance(wavelength, Quantity):
        values = np.asarray(wavelength.value)
    else:
        values = np.asarray(wavelength)

    if values.ndim != 1:
        values = np.ravel(values)

        if isinstance(wavelength, Quantity):
            wavelength = values * wavelength.unit
        else:
            wavelength = values

    if len(values) != nwave:
        raise ValueError(
            f"Wavelength length is {len(values)}, but cube has nwave={nwave}."
        )

    return wavelength


def to_float_array(data: ArrayLike, name: str) -> np.ndarray:
    """
    Convert input data to a floating-point NumPy array.

    Parameters
    -----------
    data : the user input array or Astropy Quantity
    name : the name used in error messages

    Returns
    --------
    array : the converted floating-point array
    """
    if isinstance(data, Quantity):
        array = np.asarray(data.value, dtype=float)
    else:
        array = np.asarray(data, dtype=float)

    if array.size == 0:
        raise ValueError(f"{name} is empty.")

    return array


def normalize_cube(cube: ArrayLike, name: str = "cube") -> np.ndarray:
    """
    Validate and normalize a spectral cube.

    Parameters
    -----------
    cube : the user input datacube, with a shape of [nwave, ny, nx]
    name : the name used in error messages

    Returns
    --------
    normalized_cube : the validated cube
    """
    array = to_float_array(cube, name=name)

    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [nwave, ny, nx]. Got {array.shape}.")

    return array


def normalize_maps(
    maps: Optional[ArrayLike],
    data_cube: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Validate maps or create a default white-light map from the datacube.

    Parameters
    -----------
    maps : the user input map image or map stack, with a shape of [ny, nx] or [nmap, ny, nx]
    data_cube : the original datacube used to check spatial dimensions

    Returns
    --------
    maps_array : the validated map stack, with a shape of [nmap, ny, nx]
    default_names : the default names for the maps
    """
    _, ny, nx = data_cube.shape

    if maps is None:
        white_light = np.nansum(data_cube, axis=0)
        return white_light[np.newaxis, :, :], ["White-light data"]

    array = to_float_array(maps, name="maps")

    if array.ndim == 2:
        if array.shape != (ny, nx):
            raise ValueError(
                f"2-D maps must have shape [ny, nx] = [{ny}, {nx}]. "
                f"Got {array.shape}."
            )

        return array[np.newaxis, :, :], ["Map 0"]

    if array.ndim == 3:
        if array.shape[1:] != (ny, nx):
            raise ValueError(
                f"3-D maps must have shape [nmap, ny, nx] with "
                f"[ny, nx] = [{ny}, {nx}]. Got {array.shape}."
            )

        return array, [f"Map {i}" for i in range(array.shape[0])]

    raise ValueError(
        f"maps must have shape [ny, nx] or [nmap, ny, nx]. Got {array.shape}."
    )


def make_image_norm(data: np.ndarray) -> ImageNormalize | None:
    """
    Create a robust display normalization for an image.

    Parameters
    -----------
    data : the two-dimensional image data

    Returns
    --------
    norm : the ZScale-based image normalization
    """
    finite = np.asarray(data)[np.isfinite(data)]

    if finite.size == 0:
        return None

    return ImageNormalize(
        finite,
        interval=ZScaleInterval(),
        stretch=LinearStretch(),
    )


def parse_map_names(names: Optional[str]) -> Optional[list[str]]:
    """
    Parse comma-separated map names.

    Parameters
    -----------
    names : the comma-separated map names from the command line

    Returns
    --------
    parsed_names : the parsed list of map names
    """
    if names is None:
        return None

    parsed = [item.strip() for item in names.split(",") if item.strip()]

    if not parsed:
        return None

    return parsed


class FitCubeViewer:
    """
    Interactive viewer for a datacube and an optional fitted cube.

    Parameters
    -----------
    data_cube : the original datacube, with a shape of [nwave, ny, nx]
    fitted_cube : the optional fitted cube, with a shape of [nwave, ny, nx]
    maps : the optional map image or map stack, with a shape of [ny, nx] or [nmap, ny, nx]
    wavelength : the optional wavelength array
    header : the optional FITS header used to construct the wavelength
    map_names : the optional names of the displayed maps
    combine : the method used to combine spectra in a selected region
    cmap : the Matplotlib colormap used for the map panel
    """

    def __init__(
        self,
        data_cube: ArrayLike,
        fitted_cube: Optional[ArrayLike] = None,
        maps: Optional[ArrayLike] = None,
        wavelength: Optional[ArrayLike] = None,
        header: Optional[Header] = None,
        map_names: Optional[Sequence[str]] = None,
        combine: str = "mean",
        cmap: str = "viridis",
    ) -> None:
        """
        Initialize the viewer and create the interactive window.

        Parameters
        -----------
        data_cube : the original datacube, with a shape of [nwave, ny, nx]
        fitted_cube : the optional fitted cube, with a shape of [nwave, ny, nx]
        maps : the optional map image or map stack, with a shape of [ny, nx] or [nmap, ny, nx]
        wavelength : the optional wavelength array
        header : the optional FITS header used to construct the wavelength
        map_names : the optional names of the displayed maps
        combine : the method used to combine spectra in a selected region
        cmap : the Matplotlib colormap used for the map panel
        """
        if combine not in {"mean", "sum"}:
            raise ValueError("combine must be either 'mean' or 'sum'.")

        self.data_cube: np.ndarray = normalize_cube(data_cube, name="data_cube")
        self.nwave, self.ny, self.nx = self.data_cube.shape

        self.fitted_cube: Optional[np.ndarray] = None

        if fitted_cube is not None:
            fitted = normalize_cube(fitted_cube, name="fitted_cube")

            if fitted.shape != self.data_cube.shape:
                raise ValueError(
                    "fitted_cube must have the same shape as data_cube. "
                    f"data_cube={self.data_cube.shape}, fitted_cube={fitted.shape}"
                )

            self.fitted_cube = fitted

        self.maps, default_map_names = normalize_maps(maps, self.data_cube)
        self.nmap = self.maps.shape[0]

        if map_names is None:
            self.map_names = default_map_names
        else:
            self.map_names = [str(name) for name in map_names]

            if len(self.map_names) != self.nmap:
                raise ValueError(
                    f"Expected {self.nmap} map names, got {len(self.map_names)}."
                )

        if wavelength is None:
            self.wavelength = load_wavelength(
                filename=None,
                header=header,
                nwave=self.nwave,
            )
        else:
            self.wavelength = wavelength

        self.wave_values, self.wave_label = self._prepare_wavelength(self.wavelength)

        if len(self.wave_values) != self.nwave:
            raise ValueError(
                f"wavelength length is {len(self.wave_values)}, "
                f"but cube has nwave={self.nwave}."
            )

        self.combine = combine
        self.cmap = cmap

        self.current_map_index = 0
        self.current_mask: Optional[np.ndarray] = None

        self.spectrum_zoom_xlim: Optional[tuple[float, float]] = None
        self.spectrum_zoom_ylim: Optional[tuple[float, float]] = None
        self.spectrum_full_xlim: Optional[tuple[float, float]] = None
        self.spectrum_full_ylim: Optional[tuple[float, float]] = None

        self.map_zoom_xlim: Optional[tuple[float, float]] = None
        self.map_zoom_ylim: Optional[tuple[float, float]] = None
        self.map_full_xlim: Optional[tuple[float, float]] = None
        self.map_full_ylim: Optional[tuple[float, float]] = None

        self._map_press_event: Optional[Any] = None

        self.fig: Figure
        self.ax_map: Axes
        self.ax_spec: Axes
        self.image: AxesImage
        self.cbar: Any
        self.pixel_marker: Any
        self.region_patch: Rectangle
        self.selector: RectangleSelector
        self.spec_selector: RectangleSelector
        self.btn_prev: Button
        self.btn_next: Button
        self.btn_clear_selection: Button
        self.btn_reset_zoom: Button

        self._build_figure()
        self._connect_callbacks()

        self.select_pixel(self.nx // 2, self.ny // 2)

    @staticmethod
    def _prepare_wavelength(wavelength: ArrayLike) -> tuple[np.ndarray, str]:
        """
        Convert wavelength input into numeric plotting values and an axis label.

        Parameters
        -----------
        wavelength : the wavelength array

        Returns
        --------
        values : the numeric wavelength values
        label : the x-axis label for the spectrum panel
        """
        if isinstance(wavelength, Quantity):
            unit_label = wavelength.unit.to_string("latex_inline")
            return np.asarray(wavelength.value), f"Wavelength [{unit_label}]"

        return np.asarray(wavelength), "Wavelength / spectral pixel"

    def _build_figure(self) -> None:
        """
        Build the Matplotlib graphical interface.
        """
        self.fig = plt.figure(figsize=(12, 6))

        grid = self.fig.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[1.0, 1.35],
            left=0.07,
            right=0.97,
            bottom=0.22,
            top=0.92,
            wspace=0.28,
        )

        self.ax_map = self.fig.add_subplot(grid[0, 0])
        self.ax_spec = self.fig.add_subplot(grid[0, 1])

        current_map = self.maps[self.current_map_index]

        self.image = self.ax_map.imshow(
            current_map,
            origin="lower",
            interpolation="nearest",
            cmap=self.cmap,
            norm=make_image_norm(current_map),
        )

        self.cbar = self.fig.colorbar(
            self.image,
            ax=self.ax_map,
            fraction=0.046,
            pad=0.04,
        )
        self.cbar.set_label(self.map_names[self.current_map_index])

        self.ax_map.set_title(self.map_names[self.current_map_index])
        self.ax_map.set_xlabel("X pixel")
        self.ax_map.set_ylabel("Y pixel")

        self.ax_spec.set_xlabel(self.wave_label)
        self.ax_spec.set_ylabel("Flux")
        self.ax_spec.grid(alpha=0.25)

        self.pixel_marker, = self.ax_map.plot(
            [],
            [],
            marker="+",
            markersize=14,
            markeredgewidth=2,
            color="cyan",
            linestyle="none",
            zorder=5,
        )

        self.region_patch = Rectangle(
            xy=(0, 0),
            width=1,
            height=1,
            fill=False,
            edgecolor="cyan",
            linewidth=2,
            zorder=5,
        )
        self.region_patch.set_visible(False)
        self.ax_map.add_patch(self.region_patch)

        self._remember_full_map_limits()

        ax_prev = self.fig.add_axes([0.15, 0.045, 0.11, 0.055])
        ax_next = self.fig.add_axes([0.28, 0.045, 0.11, 0.055])
        ax_clear_selection = self.fig.add_axes([0.42, 0.045, 0.17, 0.055])
        ax_reset_zoom = self.fig.add_axes([0.78, 0.045, 0.19, 0.055])

        self.btn_prev = Button(ax_prev, "Previous")
        self.btn_next = Button(ax_next, "Next")
        self.btn_clear_selection = Button(ax_clear_selection, "Clear selection")
        self.btn_reset_zoom = Button(ax_reset_zoom, "Reset spec zoom")


    def _connect_callbacks(self) -> None:
        """
        Connect Matplotlib callbacks for map zoom, map selection, spectrum zoom and buttons.
        """
        self.fig.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.fig.canvas.mpl_connect("button_release_event", self._on_button_release)

        self.btn_prev.on_clicked(self.previous_map)
        self.btn_next.on_clicked(self.next_map)
        self.btn_clear_selection.on_clicked(self.clear_selection)
        self.btn_reset_zoom.on_clicked(self.reset_spectrum_zoom)

        self.selector = RectangleSelector(
            self.ax_map,
            self._on_map_rectangle_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            props={
                "facecolor": "none",
                "edgecolor": "cyan",
                "linewidth": 1.5,
                "alpha": 0.9,
            },
        )

        self.spec_selector = RectangleSelector(
            self.ax_spec,
            self._on_spectrum_zoom_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=False,
            props={
                "facecolor": "tab:orange",
                "edgecolor": "tab:orange",
                "linewidth": 1.2,
                "alpha": 0.25,
            },
        )

    def _toolbar_is_active(self) -> bool:
        """
        Check whether the Matplotlib navigation toolbar is active.

        Returns
        --------
        active : whether pan, zoom or another toolbar mode is active
        """
        manager = self.fig.canvas.manager
        toolbar = getattr(manager, "toolbar", None)

        if toolbar is None:
            return False

        mode = getattr(toolbar, "mode", "")
        return bool(mode)

    def _event_has_multi_select_modifier(self, event: Any) -> bool:
        """
        Check whether a mouse event has a multi-selection modifier key.

        Parameters
        -----------
        event : the Matplotlib mouse event

        Returns
        --------
        has_modifier : whether the event has Command, Ctrl, Meta or Super pressed
        """
        key = getattr(event, "key", None)

        if key is None:
            return False

        if not isinstance(key, str):
            return False

        parts = {item.strip().lower() for item in key.split("+")}

        modifier_names = {
            "cmd",
            "command",
            "super",
            "meta",
            "control",
            "ctrl",
        }

        return bool(parts & modifier_names)

    def _on_button_press(self, event: Any) -> None:
        """
        Handle mouse-button press events.

        Parameters
        -----------
        event : the Matplotlib mouse event
        """
        if self._toolbar_is_active():
            return

        if event.inaxes is self.ax_spec:
            if event.button == 1 and getattr(event, "dblclick", False):
                self.reset_spectrum_zoom(event)
            return

        if event.inaxes is self.ax_map:
            if event.button == 1 and getattr(event, "dblclick", False):
                self.reset_map_zoom(event)
                self._map_press_event = None
                return

            if event.button == 1:
                self._map_press_event = event

    def _on_button_release(self, event: Any) -> None:
        """
        Handle mouse-button release events.

        Parameters
        -----------
        event : the Matplotlib mouse event
        """
        if self._toolbar_is_active():
            self._map_press_event = None
            return

        if event.inaxes is not self.ax_map:
            self._map_press_event = None
            return

        if event.button != 1:
            self._map_press_event = None
            return

        if self._map_press_event is None:
            return

        press_event = self._map_press_event
        self._map_press_event = None

        if press_event.xdata is None or press_event.ydata is None:
            return

        if event.xdata is None or event.ydata is None:
            return

        has_modifier = (
            self._event_has_multi_select_modifier(press_event)
            or self._event_has_multi_select_modifier(event)
        )

        if not has_modifier:
            return

        dx_pixels = float(event.x - press_event.x)
        dy_pixels = float(event.y - press_event.y)
        drag_distance = np.hypot(dx_pixels, dy_pixels)

        if drag_distance > 5.0:
            return

        x = int(np.round(event.xdata))
        y = int(np.round(event.ydata))

        self.toggle_pixel_selection(x=x, y=y)

    def _on_map_rectangle_select(self, eclick: Any, erelease: Any) -> None:
        """
        Handle rectangular selection on the image panel.

        Parameters
        -----------
        eclick : the mouse press event
        erelease : the mouse release event
        """
        if self._toolbar_is_active():
            return

        if eclick.xdata is None or eclick.ydata is None:
            return

        if erelease.xdata is None or erelease.ydata is None:
            return

        xmin, xmax = sorted([float(eclick.xdata), float(erelease.xdata)])
        ymin, ymax = sorted([float(eclick.ydata), float(erelease.ydata)])

        if np.isclose(xmin, xmax) or np.isclose(ymin, ymax):
            return

        has_modifier = (
            self._event_has_multi_select_modifier(eclick)
            or self._event_has_multi_select_modifier(erelease)
        )

        if has_modifier:
            self.select_region_from_bounds(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )
        else:
            self.zoom_map_to_region(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )

    def _on_spectrum_zoom_select(self, eclick: Any, erelease: Any) -> None:
        """
        Zoom into the spectrum panel using a rectangular selection.

        Parameters
        -----------
        eclick : the mouse press event
        erelease : the mouse release event
        """
        if self._toolbar_is_active():
            return

        if eclick.xdata is None or eclick.ydata is None:
            return

        if erelease.xdata is None or erelease.ydata is None:
            return

        xmin, xmax = sorted([float(eclick.xdata), float(erelease.xdata)])
        ymin, ymax = sorted([float(eclick.ydata), float(erelease.ydata)])

        if np.isclose(xmin, xmax) or np.isclose(ymin, ymax):
            return

        self.spectrum_zoom_xlim = (xmin, xmax)
        self.spectrum_zoom_ylim = (ymin, ymax)

        self.ax_spec.set_xlim(self.spectrum_zoom_xlim)
        self.ax_spec.set_ylim(self.spectrum_zoom_ylim)

        self.fig.canvas.draw_idle()

    def previous_map(self, event: Any = None) -> None:
        """
        Switch to the previous map.

        Parameters
        -----------
        event : the Matplotlib button event
        """
        self.current_map_index = (self.current_map_index - 1) % self.nmap
        self.update_map()

    def next_map(self, event: Any = None) -> None:
        """
        Switch to the next map.

        Parameters
        -----------
        event : the Matplotlib button event
        """
        self.current_map_index = (self.current_map_index + 1) % self.nmap
        self.update_map()

    def update_map(self) -> None:
        """
        Refresh the left image panel after changing the active map.
        """
        current_map = self.maps[self.current_map_index]
        current_name = self.map_names[self.current_map_index]

        old_xlim = self.ax_map.get_xlim()
        old_ylim = self.ax_map.get_ylim()

        self.image.set_data(current_map)
        self.image.set_norm(make_image_norm(current_map))

        self.ax_map.set_title(current_name)
        self.cbar.set_label(current_name)

        self.ax_map.set_xlim(old_xlim)
        self.ax_map.set_ylim(old_ylim)

        self.fig.canvas.draw_idle()

    def _remember_full_map_limits(self) -> None:
        """
        Store the full image-panel limits.
        """
        self.map_full_xlim = tuple(float(v) for v in self.ax_map.get_xlim())
        self.map_full_ylim = tuple(float(v) for v in self.ax_map.get_ylim())

    def reset_map_zoom(self, event: Any = None) -> None:
        """
        Reset the image panel to the full map view.

        Parameters
        -----------
        event : the Matplotlib mouse or button event
        """
        self.map_zoom_xlim = None
        self.map_zoom_ylim = None

        if self.map_full_xlim is not None:
            self.ax_map.set_xlim(self.map_full_xlim)

        if self.map_full_ylim is not None:
            self.ax_map.set_ylim(self.map_full_ylim)

        self.fig.canvas.draw_idle()

    def zoom_map_to_region(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        """
        Zoom the left image panel into a selected rectangular region.

        Parameters
        -----------
        xmin : the minimum x coordinate of the zoom region
        xmax : the maximum x coordinate of the zoom region
        ymin : the minimum y coordinate of the zoom region
        ymax : the maximum y coordinate of the zoom region
        """
        xmin, xmax = sorted([xmin, xmax])
        ymin, ymax = sorted([ymin, ymax])

        xmin = float(np.clip(xmin, -0.5, self.nx - 0.5))
        xmax = float(np.clip(xmax, -0.5, self.nx - 0.5))
        ymin = float(np.clip(ymin, -0.5, self.ny - 0.5))
        ymax = float(np.clip(ymax, -0.5, self.ny - 0.5))

        if np.isclose(xmin, xmax) or np.isclose(ymin, ymax):
            return

        self.map_zoom_xlim = (xmin, xmax)
        self.map_zoom_ylim = (ymin, ymax)

        self.ax_map.set_xlim(self.map_zoom_xlim)
        self.ax_map.set_ylim(self.map_zoom_ylim)

        self.fig.canvas.draw_idle()

    def select_region_from_bounds(
        self,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        """
        Select a rectangular spatial region from coordinate bounds.

        Parameters
        -----------
        xmin : the minimum x coordinate of the selected region
        xmax : the maximum x coordinate of the selected region
        ymin : the minimum y coordinate of the selected region
        ymax : the maximum y coordinate of the selected region
        """
        xmin, xmax = sorted([xmin, xmax])
        ymin, ymax = sorted([ymin, ymax])

        xx, yy = np.meshgrid(np.arange(self.nx), np.arange(self.ny))

        mask = (
            (xx >= xmin)
            & (xx <= xmax)
            & (yy >= ymin)
            & (yy <= ymax)
        )

        if not np.any(mask):
            return

        self.select_region(
            mask=mask,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
        )

    def select_pixel(self, x: int, y: int) -> None:
        """
        Select a single spaxel and update the spectrum panel.

        Parameters
        -----------
        x : the x pixel coordinate
        y : the y pixel coordinate
        """
        x = int(np.clip(x, 0, self.nx - 1))
        y = int(np.clip(y, 0, self.ny - 1))

        mask = np.zeros((self.ny, self.nx), dtype=bool)
        mask[y, x] = True

        self.current_mask = mask

        self.pixel_marker.set_data([x], [y])
        self.pixel_marker.set_visible(True)

        self.region_patch.set_visible(False)

        self.update_spectrum(title=f"Spaxel x={x}, y={y}")

    def toggle_pixel_selection(self, x: int, y: int) -> None:
        """
        Toggle one pixel in the custom multi-pixel selection.

        Parameters
        -----------
        x : the x pixel coordinate
        y : the y pixel coordinate
        """
        x = int(np.clip(x, 0, self.nx - 1))
        y = int(np.clip(y, 0, self.ny - 1))

        if self.current_mask is None:
            mask = np.zeros((self.ny, self.nx), dtype=bool)
        else:
            mask = self.current_mask.copy()

        mask[y, x] = ~mask[y, x]

        if not np.any(mask):
            self.clear_selection()
            return

        self.select_custom_mask(
            mask=mask,
            title=f"Custom selection: {int(np.sum(mask))} spaxels, combine={self.combine}",
        )

    def select_region(
        self,
        mask: np.ndarray,
        xmin: float,
        xmax: float,
        ymin: float,
        ymax: float,
    ) -> None:
        """
        Select a rectangular spatial region and update the spectrum panel.

        Parameters
        -----------
        mask : the mask of the datacube
        xmin : the minimum x coordinate of the selected region
        xmax : the maximum x coordinate of the selected region
        ymin : the minimum y coordinate of the selected region
        ymax : the maximum y coordinate of the selected region
        """
        if mask.shape != (self.ny, self.nx):
            raise ValueError(f"mask must have shape {(self.ny, self.nx)}.")

        self.current_mask = mask

        nspaxels = int(np.sum(mask))

        self.pixel_marker.set_visible(False)

        self.region_patch.set_xy((xmin, ymin))
        self.region_patch.set_width(xmax - xmin)
        self.region_patch.set_height(ymax - ymin)
        self.region_patch.set_visible(True)

        self.update_spectrum(
            title=f"Region: {nspaxels} spaxels, combine={self.combine}"
        )

    def select_custom_mask(self, mask: np.ndarray, title: str = "") -> None:
        """
        Select an arbitrary custom spatial mask and update the spectrum panel.

        Parameters
        -----------
        mask : the mask of the datacube
        title : the spectrum panel title
        """
        if mask.shape != (self.ny, self.nx):
            raise ValueError(f"mask must have shape {(self.ny, self.nx)}.")

        if not np.any(mask):
            self.clear_selection()
            return

        self.current_mask = mask

        y_selected, x_selected = np.where(mask)

        self.pixel_marker.set_data(x_selected, y_selected)
        self.pixel_marker.set_visible(True)

        self.region_patch.set_visible(False)

        self.update_spectrum(title=title)

    def clear_selection(self, event: Any = None) -> None:
        """
        Clear the current spatial selection and reset the spectrum panel.

        Parameters
        -----------
        event : the Matplotlib button event
        """
        self.current_mask = None

        self.pixel_marker.set_data([], [])
        self.pixel_marker.set_visible(False)

        self.region_patch.set_visible(False)

        self.spectrum_zoom_xlim = None
        self.spectrum_zoom_ylim = None
        self.spectrum_full_xlim = None
        self.spectrum_full_ylim = None

        self.ax_spec.clear()
        self.ax_spec.set_title("No spaxels selected")
        self.ax_spec.set_xlabel(self.wave_label)
        self.ax_spec.set_ylabel("Flux")
        self.ax_spec.grid(alpha=0.25)

        self.fig.canvas.draw_idle()

    def extract_spectrum(
        self,
        cube: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Extract a single or combined spectrum from a cube.

        Parameters
        -----------
        cube : the user input datacube, with a shape of [nwave, ny, nx]
        mask : the mask of the datacube

        Returns
        --------
        spectrum : the extracted spectrum
        """
        spectra = cube[:, mask]

        if self.combine == "mean":
            return np.nanmean(spectra, axis=1)

        return np.nansum(spectra, axis=1)

    def _remember_full_spectrum_limits(self) -> None:
        """
        Store the autoscaled full-spectrum limits for the current selection.
        """
        self.ax_spec.relim()
        self.ax_spec.autoscale_view()

        self.spectrum_full_xlim = tuple(float(v) for v in self.ax_spec.get_xlim())
        self.spectrum_full_ylim = tuple(float(v) for v in self.ax_spec.get_ylim())

    def _apply_spectrum_zoom_if_needed(self) -> None:
        """
        Apply the current spectrum zoom state if a zoom region is active.
        """
        if self.spectrum_zoom_xlim is not None:
            self.ax_spec.set_xlim(self.spectrum_zoom_xlim)

        if self.spectrum_zoom_ylim is not None:
            self.ax_spec.set_ylim(self.spectrum_zoom_ylim)

    def reset_spectrum_zoom(self, event: Any = None) -> None:
        """
        Reset the spectrum panel to the full current spectral range.

        Parameters
        -----------
        event : the Matplotlib button or mouse event
        """
        self.spectrum_zoom_xlim = None
        self.spectrum_zoom_ylim = None

        if self.spectrum_full_xlim is not None:
            self.ax_spec.set_xlim(self.spectrum_full_xlim)

        if self.spectrum_full_ylim is not None:
            self.ax_spec.set_ylim(self.spectrum_full_ylim)

        self.fig.canvas.draw_idle()

    def update_spectrum(self, title: str = "") -> None:
        """
        Refresh the spectrum panel.

        Parameters
        -----------
        title : the spectrum panel title
        """
        if self.current_mask is None:
            return

        data_spectrum = self.extract_spectrum(
            cube=self.data_cube,
            mask=self.current_mask,
        )

        self.ax_spec.clear()

        self.ax_spec.plot(
            self.wave_values,
            data_spectrum,
            color="0.25",
            lw=1.2,
            label="Data",
        )

        if self.fitted_cube is not None:
            fitted_spectrum = self.extract_spectrum(
                cube=self.fitted_cube,
                mask=self.current_mask,
            )

            self.ax_spec.plot(
                self.wave_values,
                fitted_spectrum,
                color="crimson",
                lw=1.6,
                label="Fitted",
            )

            residual = data_spectrum - fitted_spectrum

            self.ax_spec.plot(
                self.wave_values,
                residual,
                color="tab:blue",
                lw=1.0,
                alpha=0.75,
                label="Data - fitted",
            )

        self.ax_spec.set_title(title)
        self.ax_spec.set_xlabel(self.wave_label)
        self.ax_spec.set_ylabel("Flux")
        self.ax_spec.grid(alpha=0.25)
        self.ax_spec.legend(loc="best")

        self._remember_full_spectrum_limits()
        self._apply_spectrum_zoom_if_needed()

        self.fig.canvas.draw_idle()


def view_cube_results(
    data_cube: ArrayLike,
    fitted_cube: Optional[ArrayLike] = None,
    maps: Optional[ArrayLike] = None,
    wavelength: Optional[ArrayLike] = None,
    header: Optional[Header] = None,
    map_names: Optional[Sequence[str]] = None,
    combine: str = "mean",
    cmap: str = "viridis",
    block: Optional[bool] = None,
) -> FitCubeViewer:
    """
    Open the interactive datacube and fitted-cube viewer.

    Parameters
    -----------
    data_cube : the original datacube, with a shape of [nwave, ny, nx]
    fitted_cube : the optional fitted cube, with a shape of [nwave, ny, nx]
    maps : the optional map image or map stack, with a shape of [ny, nx] or [nmap, ny, nx]
    wavelength : the optional wavelength array
    header : the optional FITS header used to construct the wavelength
    map_names : the optional names of the displayed maps
    combine : the method used to combine spectra in a selected region
    cmap : the Matplotlib colormap used for the map panel
    block : the Matplotlib show blocking option

    Returns
    --------
    viewer : the interactive viewer instance
    """
    viewer = FitCubeViewer(
        data_cube=data_cube,
        fitted_cube=fitted_cube,
        maps=maps,
        wavelength=wavelength,
        header=header,
        map_names=map_names,
        combine=combine,
        cmap=cmap,
    )

    plt.show(block=block)
    return viewer


def demo_cube_viewer(
    ny: int = 40,
    nx: int = 50,
    nwave: int = 600,
    random_seed: int = 4,
    with_fit: bool = True,
) -> FitCubeViewer:
    """
    Demonstrate and debug the viewer with synthetic data.

    Parameters
    -----------
    ny : the number of y pixels in the synthetic cube
    nx : the number of x pixels in the synthetic cube
    nwave : the number of wavelength pixels in the synthetic cube
    random_seed : the random seed used to generate noise
    with_fit : whether to include a fitted cube in the demo

    Returns
    --------
    viewer : the interactive viewer instance
    """
    rng = np.random.default_rng(random_seed)

    wavelength = np.linspace(6500.0, 6600.0, nwave) * u.AA

    y_grid, x_grid = np.mgrid[0:ny, 0:nx]

    amplitude = 20.0 + 80.0 * np.exp(
        -((x_grid - 0.55 * nx) ** 2 + (y_grid - 0.45 * ny) ** 2)
        / (2.0 * (0.18 * nx) ** 2)
    )

    velocity = 100.0 * (x_grid - nx / 2.0) / nx
    sigma = 1.2 + 1.0 * y_grid / max(ny - 1, 1)
    continuum = 5.0 + 0.03 * (x_grid + y_grid)

    rest_wave = 6563.0
    center = rest_wave * (1.0 + velocity / 299792.458)

    wave_3d = wavelength.value[:, np.newaxis, np.newaxis]

    fitted_cube = (
        continuum[np.newaxis, :, :]
        + amplitude[np.newaxis, :, :]
        * np.exp(
            -0.5
            * ((wave_3d - center[np.newaxis, :, :]) / sigma[np.newaxis, :, :]) ** 2
        )
    )

    noise = rng.normal(scale=2.0, size=fitted_cube.shape)
    data_cube = fitted_cube + noise

    chi2 = 1.0 + 0.2 * rng.normal(size=(ny, nx))

    maps = np.stack(
        [
            np.nansum(data_cube, axis=0),
            amplitude,
            velocity,
            sigma,
            chi2,
        ],
        axis=0,
    )

    map_names = [
        "White-light data",
        "Line amplitude",
        "Velocity",
        "Sigma",
        "Reduced chi2",
    ]

    return view_cube_results(
        data_cube=data_cube,
        fitted_cube=fitted_cube if with_fit else None,
        maps=maps,
        wavelength=wavelength,
        map_names=map_names,
        combine="mean",
        cmap="viridis",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser.

    Returns
    --------
    parser : the configured argument parser
    """
    examples = """
Examples
--------

Run the automatic demo with synthetic datacube and fitted cube:

  python fit_cube_viewer.py --demo

Run the automatic demo as a cube viewer without fitted data:

  python fit_cube_viewer.py --demo-only-data

Open an observed datacube:

  python fit_cube_viewer.py --data datacube.fits

Compare an observed datacube with a fitted cube:

  python fit_cube_viewer.py --data datacube.fits --fitted fitted_cube.fits

Compare data and fit, and show diagnostic maps:

  python fit_cube_viewer.py \\
      --data datacube.fits \\
      --fitted fitted_cube.fits \\
      --maps fitting_maps.fits \\
      --map-names "Flux,Velocity,Sigma,Chi2"

Provide a separate wavelength file:

  python fit_cube_viewer.py \\
      --data datacube.fits \\
      --fitted fitted_cube.fits \\
      --wavelength wavelength.txt

Use summed spectra for selected regions instead of mean spectra:

  python fit_cube_viewer.py \\
      --data datacube.fits \\
      --fitted fitted_cube.fits \\
      --combine sum
"""

    parser = argparse.ArgumentParser(
        prog="fit-cube-viewer",
        description=(
            "Interactive viewer for an astronomical datacube and optional "
            "fitted spectral cube."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_argument_group("Run mode")

    mode_group.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Run a demo using automatically generated synthetic datacube, "
            "fitted cube and maps."
        ),
    )

    mode_group.add_argument(
        "--demo-only-data",
        action="store_true",
        help=(
            "Run a demo using only an automatically generated datacube, "
            "without a fitted cube."
        ),
    )

    input_group = parser.add_argument_group("Input files")

    input_group.add_argument(
        "--data",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Original observed datacube FITS file. Expected shape: "
            "[nwave, ny, nx]. Required unless --demo or --demo-only-data is used."
        ),
    )

    input_group.add_argument(
        "--data-ext",
        type=str,
        default="0",
        metavar="EXT",
        help=(
            "FITS extension for the original datacube. Can be an integer index "
            "or extension name. Default: 0."
        ),
    )

    input_group.add_argument(
        "--fitted",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Optional fitted cube FITS file. Expected shape: [nwave, ny, nx]. "
            "If provided, data and fitted spectra are plotted together."
        ),
    )

    input_group.add_argument(
        "--fitted-ext",
        type=str,
        default="0",
        metavar="EXT",
        help=(
            "FITS extension for the fitted cube. Can be an integer index or "
            "extension name. Default: 0."
        ),
    )

    input_group.add_argument(
        "--maps",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Optional FITS file containing maps. Supported shapes: [ny, nx] "
            "or [nmap, ny, nx]. If omitted, a white-light map is generated "
            "from the datacube."
        ),
    )

    input_group.add_argument(
        "--maps-ext",
        type=str,
        default="0",
        metavar="EXT",
        help=(
            "FITS extension for the maps. Can be an integer index or extension "
            "name. Default: 0."
        ),
    )

    input_group.add_argument(
        "--wavelength",
        type=str,
        default=None,
        metavar="FILE",
        help=(
            "Optional wavelength file. Supported formats: .npy, text file, "
            "or FITS file containing a 1-D wavelength array. If omitted, the "
            "wavelength axis is read from the datacube header when possible."
        ),
    )

    display_group = parser.add_argument_group("Display options")

    display_group.add_argument(
        "--map-names",
        type=str,
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated names for the displayed maps. Example: "
            '"Flux,Velocity,Sigma,Chi2".'
        ),
    )

    display_group.add_argument(
        "--combine",
        type=str,
        choices=["mean", "sum"],
        default="mean",
        help=(
            "How to combine spectra when a spatial region is selected. "
            "Default: mean."
        ),
    )

    display_group.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Matplotlib colormap for the left image panel. Default: viridis.",
    )

    demo_group = parser.add_argument_group("Demo options")

    demo_group.add_argument(
        "--demo-ny",
        type=int,
        default=40,
        help="Number of y pixels in the synthetic demo cube. Default: 40.",
    )

    demo_group.add_argument(
        "--demo-nx",
        type=int,
        default=50,
        help="Number of x pixels in the synthetic demo cube. Default: 50.",
    )

    demo_group.add_argument(
        "--demo-nwave",
        type=int,
        default=600,
        help="Number of wavelength pixels in the synthetic demo cube. Default: 600.",
    )

    demo_group.add_argument(
        "--demo-seed",
        type=int,
        default=4,
        help="Random seed for the synthetic demo data. Default: 4.",
    )

    misc_group = parser.add_argument_group("Miscellaneous")

    misc_group.add_argument(
        "--version",
        action="version",
        version="fit-cube-viewer 1.0.0",
        help="Show the program version and exit.",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Command-line entry point.

    Parameters
    -----------
    argv : the optional command-line argument sequence
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.demo and args.demo_only_data:
        parser.error("Please use only one of --demo or --demo-only-data.")

    if args.demo:
        demo_cube_viewer(
            ny=args.demo_ny,
            nx=args.demo_nx,
            nwave=args.demo_nwave,
            random_seed=args.demo_seed,
            with_fit=True,
        )
        return

    if args.demo_only_data:
        demo_cube_viewer(
            ny=args.demo_ny,
            nx=args.demo_nx,
            nwave=args.demo_nwave,
            random_seed=args.demo_seed,
            with_fit=False,
        )
        return

    if args.data is None:
        parser.error(
            "No datacube was provided. Use --data DATACUBE.fits, "
            "or run the synthetic demo with --demo."
        )

    data_ext = parse_fits_ext(args.data_ext)
    fitted_ext = parse_fits_ext(args.fitted_ext)
    maps_ext = parse_fits_ext(args.maps_ext)

    data_cube, data_header = read_fits_data_and_header(
        filename=args.data,
        ext=data_ext,
    )

    fitted_cube = None

    if args.fitted is not None:
        fitted_cube, _ = read_fits_data_and_header(
            filename=args.fitted,
            ext=fitted_ext,
        )

    maps = None

    if args.maps is not None:
        maps, _ = read_fits_data_and_header(
            filename=args.maps,
            ext=maps_ext,
        )

    wavelength = load_wavelength(
        filename=args.wavelength,
        header=data_header,
        nwave=np.asarray(data_cube).shape[0],
    )

    map_names = parse_map_names(args.map_names)

    view_cube_results(
        data_cube=data_cube,
        fitted_cube=fitted_cube,
        maps=maps,
        wavelength=wavelength,
        header=data_header,
        map_names=map_names,
        combine=args.combine,
        cmap=args.cmap,
    )


if __name__ == "__main__":
    main()
