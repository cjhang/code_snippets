#!/usr/bin/env python3
"""A minimalist tool to deal with dust polarisation 
(probably also other full-poliration data)

Author: Jianhang Chen, cjhastro@gmail.com
History:
    2023-05-11: started the utility

Requirement:
    numpy
    matplotlib
    astropy >= 5.0
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import collections  as mc
from matplotlib.transforms import blended_transform_factory

from astropy.io import fits
from astropy import stats 

def read_SOFIA(fitsfile):
    """read the full polarisation data from SOFIA/HAWC+
    """

    with fits.open(fitsfile) as hdu:
        imgheader = hdu[0].header
        I = hdu['STOKES I'].data
        Q = hdu['STOKES Q'].data
        U = hdu['STOKES U'].data
        V = np.zeros_like(I)
        Ierr = hdu['ERROR I'].data
        Qerr = hdu['ERROR Q'].data
        Uerr = hdu['ERROR U'].data
        Verr = np.zeros_like(Ierr)
    return imgheader, np.array([I,Q,U,V]), np.array([Ierr,Qerr,Uerr,Verr])

def read_ALMA(fitsfile):
    """read the full polarisation data from ALMA
    """
    with fits.open(fitsfile) as hdu:
        imgheader = hdu[0].header
        data = hdu[0].data
    I = data[0,0]
    Q = data[1,0]
    U = data[2,0]
    V = data[3,0]
    #TODO: calculate the error using beam_stats
    Ierr = np.zeros_like(I)
    Qerr = np.zeros_like(Q)
    Uerr = np.zeros_like(U)
    Verr = np.zeros_like(V)
    return imgheader, np.array([I,Q,U,V]), np.array([Ierr,Qerr,Uerr,Verr])

def _resolve_stokes(data, I=None):
    """split Stokes data into I, Q and U maps.

    ``data`` may be a 3-D array (n, ny, nx) or a tuple/list of 2-D maps, e.g.
    ``(I, Q, U)`` or ``(Q, U)``. If the first dimension / number of maps is
    >= 3 the first three are taken as (I, Q, U); if it is 2 the two maps are
    taken as (Q, U) and ``I`` falls back to the optional ``I`` argument (or
    None). Extra maps beyond the third (e.g. V) are ignored.

    Shared input resolution for :func:`make_pola` and :func:`make_poli`.
    """
    if isinstance(data, (tuple, list)):
        data = [np.asarray(m, float) for m in data]   # keep as a list of maps
        n = len(data)
    else:
        data = np.asarray(data)
        if data.ndim != 3:
            raise ValueError(f"`data` must be a 3-D (n, ny, nx) array or a "
                             f"tuple/list of maps, got array shape {data.shape}")
        n = data.shape[0]
    if n >= 3:
        I, Q, U = data[0], data[1], data[2]
    elif n == 2:
        Q, U = data[0], data[1]
    else:
        raise ValueError(f"`data` must have at least 2 Stokes maps/slices, got {n}")
    Q = np.asarray(Q, float)
    U = np.asarray(U, float)
    if I is not None:
        I = np.asarray(I, float)
    return Q, U, I


def _polar_angle(Q, U):
    """polarisation angle in radian: 0.5*arctan2(U, Q)."""
    return 0.5*np.arctan2(U, Q)


def _polar_intensity(Q, U, norm):
    """polarisation intensity/fraction: sqrt(Q**2 + U**2)/norm."""
    return np.sqrt(Q**2 + U**2)/norm


def make_pola(data=None, step=1, offset=None, mask=None, func='median'):
    """calculate the polarisation angle, returning its grid coordinates.

    The angle is computed from the Stokes Q and U after block-reducing them
    onto an auto-centred regular grid, so ``step`` need not divide the image
    size, no ragged edge blocks are produced and the angle survives averaging.

    Args:
        data: 3-D Stokes data (n, ny, nx). If n >= 3 the first three slices are
            (I, Q, U); if n == 2 the two slices are (Q, U)
        step: block size in pixels (1 = one block per pixel)
        offset: (off_y, off_x) grid offset in pixels; None auto-centres
        mask: 2D boolean mask (True = bad); masks whole blocks
        func: 'median' or 'mean' block reduction

    Return:
        (xs, ys, pola): grid coordinates and the polarisation angle in radian
    """
    Q, U, _ = _resolve_stokes(data)
    Qs, ny, off_y, nx, off_x = _downsample(Q, step, mask, func, offset, True)
    Us = _downsample(U, step, mask, func, offset)
    pola = _polar_angle(Qs, Us)
    return _centres(nx, off_x, step), _centres(ny, off_y, step), pola


def make_poli(data=None, norm=None, step=1, offset=None, mask=None, func='median'):
    """calculate the polarisation intensity/fraction, returning its grid coordinates.

    The intensity/fraction is computed from the Stokes Q and U (and the norm)
    after block-reducing them onto an auto-centred regular grid, so ``step``
    need not divide the image size and the values survive averaging.

    Args:
        data: 3-D Stokes data (n, ny, nx). If n >= 3 the first three slices are
            (I, Q, U); if n == 2 the two slices are (Q, U)
        norm: divide by ``norm`` (default data[0], i.e. Stokes I, or a ones map
            if no data, giving a fraction)
        step: block size in pixels (1 = one block per pixel)
        offset: (off_y, off_x) grid offset in pixels; None auto-centres
        mask: 2D boolean mask (True = bad); masks whole blocks
        func: 'median' or 'mean' block reduction

    Return:
        (xs, ys, poli): grid coordinates and the intensity/fraction map

    Example:
        # polarisation fraction
        data = read_SOFIA("filename")
        xs, ys, poli = make_poli(data)
    """
    Q, U, I = _resolve_stokes(data)
    if norm is None:
        norm = I if I is not None else np.ones_like(Q)
    Qs, ny, off_y, nx, off_x = _downsample(Q, step, mask, func, offset, True)
    Us = _downsample(U, step, mask, func, offset)
    Ns = _downsample(np.asarray(norm, float), step, mask, func, offset)
    poli = _polar_intensity(Qs, Us, Ns)
    return _centres(nx, off_x, step), _centres(ny, off_y, step), poli

def _draw_ruler(ax, Lr, color='lightblue', lw=1, fontsize=14, ruler=2):
    """draw a physical-length '%' scale bar pinned to the bottom-right of the panel.

    The bar's position (right end and baseline) is fixed in axes-fraction
    coordinates via ``ax.transAxes``, so it stays in the corner regardless of the
    vector/data range. Only the horizontal extent ``Lr`` is in data units, so the
    bar still represents the physical length (e.g. ``2*ruler*scale*pix``).
    """
    x_r_ax, y_ax = 0.95, 0.05            # right end, near the bottom of the panel
    # data x-coordinate of that bottom-right point (gives the bar its physical length)
    x_r = ax.transData.inverted().transform(ax.transAxes.transform((x_r_ax, y_ax)))[0]
    # bar baseline in axes fraction, horizontal extent in data units
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot([x_r - Lr, x_r], [y_ax, y_ax], transform=trans,
            color=color, lw=lw, clip_on=False, solid_capstyle='butt')
    # label centred above the bar, fully in axes-fraction coordinates
    x_l_ax = ax.transAxes.inverted().transform(ax.transData.transform((x_r - Lr, 0)))[0]
    ax.text(0.5*(x_l_ax + x_r_ax), y_ax + 0.01, f'{ruler:g}%',
            transform=ax.transAxes, ha='center', va='bottom',
            fontsize=fontsize, color=color)


def show_vectors(image=None, data=None, I=None, step=10,
                 func='median', mask=None, offset=None,
                 length_mode='intensity', field='E', color_by=None,
                 scale=1, rotate=0, pixel_scale=1,
                 color='lightblue', cmap='magma', lw=1,
                 show_ruler=True, ruler=2, show_cbar=False, fontsize=12,
                 origin='lower', show_image=True, ax=None, average_mode=None, **kwargs):
    """visualize polarization (E/B field) pseudo-vectors.

    The vectors are first resampled onto an auto-centred regular grid (see
    :func:`make_pola` / :func:`make_poli`), making the plot robust to noise and
    removing the constraint that ``step`` divides the image size. Resampling is performed on the
    Stokes Q and U (the coherent components) so both the angle and the length
    information survive the averaging. Vectors are drawn as headless segments
    (``LineCollection``) for pixel-exact length scaling; the averaging is always
    done upstream (quiver cannot average).

    Args:
        image: 2D background image
        data: 3-D Stokes data (n, ny, nx); Q and U are resampled for the vectors.
            If n >= 3 the first three slices are (I, Q, U); if n == 2 they are
            (Q, U)
        I: optional Stokes I map (only needed when ``data`` has just Q, U and
            you want intensity rather than fraction; default ones => fraction)
        step: block size in pixels for resampling (grid is auto-centred)
        func: 'median' or 'mean' block reduction
        mask: 2D boolean mask (True = bad)
        length_mode: 'intensity' scales length ~ poli; 'uniform' uses a fixed length
        field: 'E' keeps the E-vector angle, 'B' adds 90 deg (magnetic field)
        color_by: None, 'poli' or 'pola' to colour the segments with a colormap
        scale: length scale of the vectors
        rotate: extra rotation of the vectors, in radian
        pixel_scale: physical size of one pixel; rescales coordinates and lengths
        color: segment colour (used when color_by is None)
        cmap: colormap for the background image and/or color_by
    """
    # full map size (needed for coordinate centring; driven by the vector source)
    if data is None and image is None:
        raise ValueError("provide `data` (or `image`)")
    if data is not None:
        full_ys, full_xs = data[0].shape
    else:
        full_ys, full_xs = image.shape
    if average_mode is not None:            # backward-compatible alias
        func = average_mode
    if func not in ('median', 'mean'):
        raise ValueError("func must be 'median' or 'mean'")

    # ---- resample the data (grid from make_pola, shared by make_poli) ----
    xs_c, ys_c, a = make_pola(data=data, step=step, func=func, mask=mask, offset=offset)
    # poli is only needed to scale/colour the vectors; skip it otherwise
    if length_mode == 'intensity' or color_by == 'poli':
        _, _, f = make_poli(data=data, norm=I, step=step, func=func, mask=mask, offset=offset)
    else:
        f = a                      # reuse pola as the shape/mask template

    # ---- orient the vectors ----
    a = np.ma.asarray(a, float) + rotate + (np.pi/2 if field == 'B' else 0)
    a = np.mod(a, np.pi)

    # ---- coordinate frame (optional physical scaling, offset from centre) ----
    if pixel_scale is not None:
        xx = (xs_c - 0.5*full_xs) * pixel_scale
        yy = (ys_c - 0.5*full_ys) * pixel_scale
    else:
        xx, yy = xs_c, ys_c

    # ---- vector lengths ----
    pix = pixel_scale if pixel_scale is not None else 1
    f = np.ma.asarray(f, float)
    if length_mode == 'intensity':
        r = f * scale * 100 * pix          # x100 recovers the percent
    else:                                   # uniform length
        r = np.ma.masked_array(np.full_like(f, scale*pix), mask=np.ma.getmaskarray(f))

    # ---- build the headless segments, dropping masked/non-finite entries ----
    X, Y = np.meshgrid(xx, yy)
    x1 = X + r*np.sin(a);  y1 = Y - r*np.cos(a)
    x2 = X - r*np.sin(a);  y2 = Y + r*np.cos(a)
    seg = np.stack([np.stack([x1, y1], -1), np.stack([x2, y2], -1)], 2)
    ok = ~np.ma.getmaskarray(f) & ~np.ma.getmaskarray(a) & np.isfinite(f) & np.isfinite(a)
    ok = np.asarray(ok, bool)
    seg = np.asarray(seg[ok], float)
    if color_by is not None:
        vals = {'poli': f, 'pola': a}[color_by]
        vals = np.asarray(vals[ok], float)
    else:
        vals = None

    lc = mc.LineCollection(seg, colors=None if vals is not None else color, linewidths=lw)
    if vals is not None:
        lc.set_array(vals)
        lc.set_cmap(cmap)

    if ax is None:
        fig, ax = plt.subplots()

    if (image is not None) and show_image:
        im_opts = dict(kwargs)
        if pixel_scale is not None:
            # put the background image in the same centred/scaled frame as the vectors
            im_opts['extent'] = (-0.5*full_xs*pixel_scale, 0.5*full_xs*pixel_scale,
                                 -0.5*full_ys*pixel_scale, 0.5*full_ys*pixel_scale)
        im = ax.imshow(image, origin=origin, cmap=cmap, **im_opts)
        if show_cbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('[mJy/beam]', fontsize=fontsize)

    ax.add_collection(lc)

    # ---- '%' ruler pinned to the bottom-right of the panel ----
    if show_ruler and ok.any():
        _draw_ruler(ax, 2*ruler*scale*pix, color=color, lw=lw,
                    fontsize=fontsize, ruler=ruler)
    return ax


def show_vectors_pix(image=None, data=None, I=None, step=10,
                     func='median', mask=None, offset=None,
                     length_mode='intensity', field='E', color_by=None,
                     scale=1, rotate=0,
                     color='lightblue', cmap='magma', lw=1,
                     show_ruler=True, ruler=2, show_cbar=False, fontsize=12,
                     origin='lower', show_image=True, ax=None, average_mode=None, **kwargs):
    """visualize polarization pseudo-vectors in raw pixel coordinates.

    Same rendering as :func:`show_vectors` but with the background image and
    the vectors drawn in the original pixel frame (origin at the bottom-left, 1
    pixel = 1 unit), i.e. the vectors are *not* offset relative to the map
    centre and are not scaled by ``pixel_scale``. Useful when you want the
    vectors aligned 1:1 with a pixel grid or don't need physical (arcsec)
    coordinates. Block centres fall on pixel centres (``offset + step*(i+0.5)``
    in raw pixels) so the plot matches ``plt.imshow(image, origin='lower')``.

    Args:
        image: 2D background image (pixel frame)
        data: 3-D Stokes data (n, ny, nx); Q and U are resampled for the vectors.
            If n >= 3 the first three slices are (I, Q, U); if n == 2 they are
            (Q, U)
        I: optional Stokes I map (only needed when ``data`` has just Q, U and
            you want intensity rather than fraction; default ones => fraction)
        step: coarse-grid spacing in pixels (need not divide the image size)
        func: 'median' or 'mean' reduction used when resampling
        mask: 2D mask; masks whole blocks (True = bad)
        offset: (off_y, off_x) block-grid offset in pixels (default auto-centres)
        length_mode: 'intensity' (length = poli*scale*100) or 'uniform'
        field: 'E' or 'B' (rotates the angle by 90 deg for B)
        color_by: 'poli' or 'pola' to colour the vectors by that map
        scale: length scale (percent/fraction depending on ``length_mode``)
        rotate: extra rotation (radian) added to the angle
        color, cmap, lw: segment colour, colormap and line width
        show_ruler, ruler: draw a '%' length ruler of ``ruler`` percent
        show_cbar: add a colour bar for the background image
        fontsize, origin: text size and image origin ('lower' or 'upper')
        show_image: draw the background image (default True)
        ax: matplotlib axis to plot into
        average_mode: legacy alias for ``func``
        **kwargs: passed to ``imshow`` for the background image

    Returns:
        The matplotlib axis
    """
    if data is None and image is None:
        raise ValueError("provide `data` (or `image`)")
    if data is None:
        raise ValueError("provide `data` (full-Stokes cube)")
    if average_mode is not None:
        func = average_mode
    if func not in ('median', 'mean'):
        raise ValueError("func must be 'median' or 'mean'")

    # ---- resample onto the coarse grid (raw pixel coordinates) ----
    xs_c, ys_c, a = make_pola(data=data, step=step, func=func, mask=mask, offset=offset)
    # poli is only needed to scale/colour the vectors; skip it otherwise
    if length_mode == 'intensity' or color_by == 'poli':
        _, _, f = make_poli(data=data, norm=I, step=step, func=func, mask=mask, offset=offset)
    else:
        f = a                      # reuse pola as the shape/mask template

    a = np.ma.asarray(a, float) + rotate + (np.pi/2 if field == 'B' else 0)
    a = np.mod(a, np.pi)
    f = np.ma.asarray(f, float)
    if length_mode == 'intensity':
        r = f * scale * 100          # length in raw pixels = percent of map scale
    else:
        r = np.ma.masked_array(np.full_like(f, scale), mask=np.ma.getmaskarray(f))

    X, Y = np.meshgrid(xs_c, ys_c)
    x1 = X + r*np.sin(a);  y1 = Y - r*np.cos(a)
    x2 = X - r*np.sin(a);  y2 = Y + r*np.cos(a)
    seg = np.stack([np.stack([x1, y1], -1), np.stack([x2, y2], -1)], 2)
    ok = ~np.ma.getmaskarray(f) & ~np.ma.getmaskarray(a) & np.isfinite(f) & np.isfinite(a)
    ok = np.asarray(ok, bool)
    seg = np.asarray(seg[ok], float)
    vals = None
    if color_by is not None:
        vals = np.asarray({'poli': f, 'pola': a}[color_by][ok], float)

    lc = mc.LineCollection(seg, colors=None if vals is not None else color, linewidths=lw)
    if vals is not None:
        lc.set_array(vals); lc.set_cmap(cmap)

    if ax is None:
        fig, ax = plt.subplots()
    if (image is not None) and show_image:
        im = ax.imshow(image, origin=origin, cmap=cmap, **kwargs)
        if show_cbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('[mJy/beam]', fontsize=fontsize)
    ax.add_collection(lc)

    # ---- '%' ruler pinned to the bottom-right of the panel ----
    if show_ruler and ok.any():
        _draw_ruler(ax, 2*ruler*scale, color=color, lw=lw,
                    fontsize=fontsize, ruler=ruler)
    return ax


def _axis_grid(n, step, offset=None):
    """number of blocks and grid offset along one axis (auto-centred).

    The grid is centred so the block size is always exactly ``step`` and ``step``
    does not need to divide the image size.
    """
    if int(step) <= 0:
        raise ValueError("step must be a positive integer")
    if n < step:
        raise ValueError(f"step ({step}) larger than axis ({n}); use a smaller step")
    if offset is None:
        nblocks = max(1, n//step)
        off = max(0, (n - nblocks*step)//2)
    else:
        off = int(offset)
        nblocks = max(1, (n - 2*off)//step)
    return nblocks, off


def _block_agg(ima, ny, nx, off_y, off_x, step, func, mask):
    """apply a median/mean reduction over non-overlapping step x step blocks."""
    sub = ima[off_y:off_y+ny*step, off_x:off_x+nx*step]
    blk = sub.reshape(ny, step, nx, step)
    if mask is not None:
        msub = mask[off_y:off_y+ny*step, off_x:off_x+nx*step].reshape(ny, step, nx, step)
        marr = np.ma.array(blk, mask=msub)
        f = np.ma.median if func == 'median' else np.ma.mean
    else:
        marr = blk
        f = np.median if func == 'median' else np.mean
    out = f(marr, axis=3)
    return f(out, axis=1)


def _centres(nblocks, off, step):
    """block-centre coordinates along one axis."""
    return off + step*(np.arange(nblocks) + 0.5)


def _downsample(ima, step, mask=None, func='median', offset=None, return_grid=False):
    """block-reduce a 2D map onto an auto-centred grid (vectorised).

    The single grid-reduction primitive shared by :func:`make_pola` and
    :func:`make_poli`. If ``return_grid`` is True, also return the (ny, off_y,
    nx, off_x) grid layout (needed by callers that reconstruct block-centre
    coordinates).
    """
    ima = np.asarray(ima, float)
    ys, xs = ima.shape
    ny, off_y = _axis_grid(ys, step, None if offset is None else offset[0])
    nx, off_x = _axis_grid(xs, step, None if offset is None else offset[1])
    arr = _block_agg(ima, ny, nx, off_y, off_x, step, func, mask)
    if return_grid:
        return arr, ny, off_y, nx, off_x
    return arr

def velvec(velmap):
    """calculate the velocity vectors

    Return:
        [radian_map, gradiant_map] the angle in radian and gradiant
    """
    yg, xg = np.gradient(velmap)
    radian_map = np.arctan2(yg, xg)
    gradiant_map = np.sqrt(yg**2 + xg**2)
    return radian_map, gradiant_map

def show_gradient_vectors(velmap, step=1, xstep=None, ystep=None, start=0, mask=None,
                          ax=None, show_cbar=True, vector_width=None, **kwargs):
    # get the size of the image
    ys, xs = velmap.shape
    # get the gradient in both direction
    yg, xg = np.gradient(velmap,)
    if xstep is None:
        xstep = step
    if ystep is None:
        ystep = step
    x_steps = np.arange(start,xs+1-xstep,xstep)
    y_steps = np.arange(start,ys+1-ystep,ystep)
    u_steps = np.zeros((len(y_steps), len(x_steps)))
    v_steps = np.zeros((len(y_steps), len(x_steps)))

    for yidx in range(y_steps.size):
        y = y_steps[yidx]
        for xidx in range(x_steps.size):
            x = x_steps[xidx]
            if mask is not None:
                mask_step = mask[y:y+ystep,x:x+xstep]
            else:
                mask_step = None
            v_steps[yidx,xidx] = np.ma.median(np.ma.array(yg[y:y+ystep,x:x+xstep], mask=mask_step))
            u_steps[yidx,xidx] = np.ma.median(np.ma.array(xg[y:y+ystep,x:x+xstep], mask=mask_step))
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=(8,7))
    im = ax.imshow(velmap, cmap='RdBu_r', origin='lower', **kwargs)
    if show_cbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.quiver(x_steps-0.5+0.5*xstep, y_steps-0.5+0.5*ystep, u_steps, v_steps, 
              angles='xy',scale_units='xy', scale=None, width=vector_width, 
              headwidth=5, headlength=5)

def show_vectors4(image=None, pola=None, poli=None, step=1, scale=1, rotate=0, 
                  mask=None, ax=None, average_mode='median',
                  cmap='magma', show_ruler=True, ruler=2, 
                  color='lightblue', lw=1, fontsize=12, show_cbar=False, 
                  arrow_color='white', arrow_width=1,
                  **kwargs):
    """optimized to show velocity grandient

    To replace show_vectors in the future.

    Args:
        image: the 2D image data
        pola: the polarisation angle, in radian
        poli: the polarisation intensity, can be any 2D scalers to scale the length of the vectors
        data: the 3-D polarisation data
        rotate: the additional rotation of the vectors, in radian
    """
    if image is not None:
        (ys,xs) = image.shape
    else:
        (ys,xs) = pola.shape
    if ax is None:
        fig, ax = plt.subplots()
    for y in range(0,ys-step,step):
        for x in range(0,xs-step,step):
            if mask is not None:
                mask_step = mask[y:y+step,x:x+step]
            else:
                mask_step = None
            if pola is not None:
                pola_step = pola[y:y+step,x:x+step] + rotate
                a = np.ma.median(np.ma.array(pola_step, mask=mask_step))
            if poli is not None:
                poli_step = poli[y:y+step,x:x+step]
                f = np.ma.median(np.ma.array(poli_step, mask=mask_step))
            else:
                f = 1.0
            if np.ma.is_masked(f):
                continue
            x_center = x+step*0.5
            y_center = y+step*0.5
            # draw the vectors
            r = f*scale  # x100 here is to compensate the per cent
            x1=x_center+r*np.sin(a)
            y1=y_center-r*np.cos(a)
            x2=x_center-r*np.sin(a)
            y2=y_center+r*np.cos(a)
            ax.arrow(x1, y1, x1-x2, y1-y2, length_includes_head=True,
                     width=arrow_width,
                     color=arrow_color)

    # plot the 1% ruler
    if show_ruler:
        ax.plot([0.85, 0.85+2*ruler*scale/xs], [0.1, 0.1], transform=ax.transAxes, lw=lw, color=color)
        ax.text(0.85+ruler*scale/xs, 0.12, f'{ruler:.0f}%', transform=ax.transAxes, fontsize=12, 
                color='white', ha='center')

    if image is not None:
        im = ax.imshow(image, origin='lower', cmap=cmap, **kwargs)
        if show_cbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('[mJy/beam]', fontsize=fontsize)
    return ax
