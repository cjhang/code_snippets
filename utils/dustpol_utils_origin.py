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

def make_pola(data=None, Q=None, U=None, step=1, mask=None):
    """calculate the polarisation angle

    Args: 
        data: the 3-D data, with the first dimesion is the Stokes dimension
        mask: 2D mask for the maps, applied to all the Stokes
    
    Return:
        The raidan angle of the polarisation
    """
    if data is not None:
        Q = data[1]
        U = data[2]
    if mask is not None:
        pola = 0.5*(np.arctan2(U[mask], Q[mask]))
    else:
        pola = 0.5*(np.arctan2(U, Q))
    return pola

def resample_pola(data=None, Q=None, U=None, mask=None):
    """resample the data to create pola map"""
    if data is not None:
        Q = data[1]
        U = data[2]
    ys, xs = Q.shape
    for y in range(0,ys-step,step):
        for x in range(0,xs-step,step):
            if mask is not None:
                mask_step = mask[y:y+step,x:x+step]
            else:
                mask_step = None
            if data is not None:
                I_step = I[y:y+step,x:x+step]
                Q_step = Q[y:y+step,x:x+step]
                U_step = U[y:y+step,x:x+step]
                a = make_pola(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step))) + rotate
    return 

def make_poli(data=None, Q=None, U=None, norm=None, mask=None):
    """calculate the polarisation intensity or fraction

    Args:
        data: the 3-D polarisation data
        mask: 2D mask
        norm : divide the data by norm

    Return:
        2D map

    Example:
        # calculate the polarisation fraction
        data = read_SOFIA("filename")
        poli = make_poli(data, norm=data[0])

    """
    if norm is None:
        if data is not None:
            norm = data[0]
        else:
            norm = 1
    if data is not None:
        Q = data[1]
        U = data[2]
    
    return np.sqrt(Q**2+U**2)/norm

def resample_poli(data=None, Q=None, U=None, mask=None, step=1):
    ys, xs = Q.shape
    for y in range(0,ys-step,step):
        for x in range(0,xs-step,step):
            if mask is not None:
                mask_step = mask[y:y+step,x:x+step]
            else:
                mask_step = None
            if data is not None:
                I_step = I[y:y+step,x:x+step]
                Q_step = Q[y:y+step,x:x+step]
                U_step = U[y:y+step,x:x+step]
                a = make_pola(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step))) + rotate
 
                f = make_poli(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step)),
                              norm=np.ma.median(np.ma.array(I_step, mask=mask_step)))
    return

def show_vectors(image, pola, poli=None, step=1, scale=1, rotate=0, mask=None, ax=None, 
                 edgecolors='white', facecolors='cyan', lw=1, fontsize=12, show_cbar=False, 
                 **kwargs):
    """simple visualization tools for vectors, designed to show the geometry of magnetic fields

    Args:
        image: the 2D image data
        pola: the polarisation angle, in radian
        poli: the polarisation intensity, can be any 2D scalers to scale the length of the vectors
        rotate: the additional rotation of the vectors, in radian
    """
    pola = pola + rotate

    if image is not None:
        (ys,xs) = image.shape
    else:
        (ys,xs) = pola.shape
    linelist=[]
    for y in range(0,ys,step):
        for x in range(0,xs,step):
            if mask is not None:
                if mask[y,x]:
                    continue
            if poli is not None:
                f = poli[y,x]
            else:
                f = 1
            r=f*scale
            a=pola[y,x]
            x1=x+r*np.sin(a)
            y1=y-r*np.cos(a)
            x2=x-r*np.sin(a)
            y2=y+r*np.cos(a)
            line =[(x1,y1),(x2,y2)]
            linelist.append(line)
    lc = mc.LineCollection(linelist, edgecolors=edgecolors, facecolors=facecolors, linewidths=lw)
    if ax is None:
        fig, ax = plt.subplots()
    if image is not None:
        im = ax.imshow(image, origin='lower', cmap='magma', **kwargs)
        if show_cbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('[mJy/beam]', fontsize=fontsize)
    ax.add_collection(lc)
    return ax

def resample_vectors(data=None, pola=None, poli=None, step=1,
                     mask=None, average_mode='median', **kwargs):
    """resample vectors

    Args:
        data: the 3-D polarisation data
        pola: the polarisation angle, in radian
        poli: the polarisation intensity, can be any 2D scalers to scale the length of the vectors

    Return:
        ndarray: [poli, pola] after the resampling
    """
    if data is not None:
        I = data[0]
        Q = data[1]
        U = data[2]
        ys, xs = I.shape
    else:
        (ys,xs) = pola.shape
    vector_list=[]
    for y in range(0,ys-step,step):
        for x in range(0,xs-step,step):
            if mask is not None:
                mask_step = mask[y:y+step,x:x+step]
            else:
                mask_step = None
            if data is not None:
                I_step = I[y:y+step,x:x+step]
                Q_step = Q[y:y+step,x:x+step]
                U_step = U[y:y+step,x:x+step]
                a = make_pola(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step)))
            elif pola is not None:
                pola_step = pola[y:y+step,x:x+step] + rotate
                a = np.ma.median(np.ma.array(pola_step, mask=mask_step))
            if data is not None:
                f = make_poli(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step)),
                              norm=np.ma.median(np.ma.array(I_step, mask=mask_step)))
            elif poli is not None:
                poli_step = poli[y:y+step,x:x+step]
                f = np.ma.median(np.ma.array(poli_step, mask=mask_step))
            else:
                f = 1.0
            if np.ma.is_masked(f):
                continue
            vector_list.append([f, a])
    return np.array(vector_list).T

def show_vectors2(image=None, pola=None, poli=None, data=None, step=1, scale=1, rotate=0, 
                  mask=None, ax=None, average_mode='median',
                  cmap='magma', show_ruler=True, ruler=2, origin='lower',
                  color='lightblue', lw=1, fontsize=12, show_cbar=False, show_image=True, 
                  **kwargs):
    """simple visualization tools for vectors, designed to show the geometry of magnetic fields

    To replace show_vectors in the future.

    Args:
        image: the 2D image data
        pola: the polarisation angle, in radian
        poli: the polarisation intensity, can be any 2D scalers to scale the length of the vectors
        data: the 3-D polarisation data
        rotate: the additional rotation of the vectors, in radian
    """
    if data is not None:
        I = data[0]
        Q = data[1]
        U = data[2]
    if image is not None:
        (ys,xs) = image.shape
    else:
        (ys,xs) = pola.shape
    linelist=[]
    for y in range(0,ys-step,step):
        for x in range(0,xs-step,step):
            if mask is not None:
                mask_step = mask[y:y+step,x:x+step]
            else:
                mask_step = None
            if data is not None:
                I_step = I[y:y+step,x:x+step]
                Q_step = Q[y:y+step,x:x+step]
                U_step = U[y:y+step,x:x+step]
                a = make_pola(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step))) + rotate
            elif pola is not None:
                pola_step = pola[y:y+step,x:x+step] + rotate
                a = np.ma.median(np.ma.array(pola_step, mask=mask_step))
            if data is not None:
                f = make_poli(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step)),
                              norm=np.ma.median(np.ma.array(I_step, mask=mask_step)))
            elif poli is not None:
                poli_step = poli[y:y+step,x:x+step]
                f = np.ma.median(np.ma.array(poli_step, mask=mask_step))
            else:
                f = 1.0
            if np.ma.is_masked(f):
                continue
            x_center = x+step*0.5
            y_center = y+step*0.5
            # draw the vectors
            r = f*scale*100  # x100 here is to compensate the per cent
            x1=x_center+r*np.sin(a)
            y1=y_center-r*np.cos(a)
            x2=x_center-r*np.sin(a)
            y2=y_center+r*np.cos(a)
            line =[(x1,y1),(x2,y2)]
            linelist.append(line)
    lc = mc.LineCollection(linelist, colors=color, linewidths=lw)
    if ax is None:
        fig, ax = plt.subplots()

    # plot the 1% ruler
    if show_ruler:
        ax.plot([0.95-2*ruler*scale/xs, 0.95], [0.1, 0.1], transform=ax.transAxes, 
                lw=lw, color=color)
        ax.text(0.95-ruler*scale/xs, 0.12, f'{ruler:.0f}%', transform=ax.transAxes, 
                fontsize=12, color=color, ha='center')

    if (image is not None) and show_image:
        im = ax.imshow(image, origin=origin, cmap=cmap, **kwargs)
        if show_cbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('[mJy/beam]', fontsize=fontsize)
    ax.add_collection(lc)
    return ax

def show_vectors3(image=None, pola=None, poli=None, data=None, step=1, scale=1, rotate=0, 
                  mask=None, ax=None, average_mode='median', pixel_scale=1,
                  cmap='magma', show_ruler=True, ruler=2,
                  color='lightblue', lw=1, fontsize=12, show_cbar=False, show_image=True, 
                  **kwargs):
    """simple visualization tools for vectors, designed to show the geometry of magnetic fields

    To replace show_vectors in the future.

    Args:
        image: the 2D image data
        pola: the polarisation angle, in radian
        poli: the polarisation intensity, can be any 2D scalers to scale the length of the vectors
        data: the 3-D polarisation data
        rotate: the additional rotation of the vectors, in radian
    """
    if data is not None:
        I = data[0]
        Q = data[1]
        U = data[2]
    if image is not None:
        (ys,xs) = image.shape
    else:
        (ys,xs) = I.shape
    linelist=[]
    center = [0.5*xs, 0.5*ys]
    for y in range(0,ys-step,step):
        for x in range(0,xs-step,step):
            if mask is not None:
                mask_step = mask[y:y+step,x:x+step]
            else:
                mask_step = None
            if data is not None:
                I_step = I[y:y+step,x:x+step]
                Q_step = Q[y:y+step,x:x+step]
                U_step = U[y:y+step,x:x+step]
                a = make_pola(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step))) + rotate
            elif pola is not None:
                pola_step = pola[y:y+step,x:x+step] + rotate
                a = np.ma.median(np.ma.array(pola_step, mask=mask_step))
            if data is not None:
                f = make_poli(Q=np.ma.median(np.ma.array(Q_step, mask=mask_step)), 
                              U=np.ma.median(np.ma.array(U_step, mask=mask_step)),
                              norm=np.ma.median(np.ma.array(I_step, mask=mask_step)))
            elif poli is not None:
                poli_step = poli[y:y+step,x:x+step]
                f = np.ma.median(np.ma.array(poli_step, mask=mask_step))
            else:
                f = 1.0
            if np.ma.is_masked(f):
                continue
            x_center = (x+step*0.5 - center[0])*pixel_scale
            y_center = (y+step*0.5 - center[1])*pixel_scale
            # draw the vectors
            r = f*scale*100*pixel_scale  # x100 here is to compensate the per cent
            x1=x_center+r*np.sin(a)
            y1=y_center-r*np.cos(a)
            x2=x_center-r*np.sin(a)
            y2=y_center+r*np.cos(a)
            line =[(x1,y1),(x2,y2)]
            linelist.append(line)
    lc = mc.LineCollection(linelist, colors=color, linewidths=lw)
    if ax is None:
        fig, ax = plt.subplots()

    # plot the 1% ruler
    if show_ruler:
        ax.plot([0.85, 0.85+2*ruler*scale*pixel_scale], [0.1, 0.1], transform=ax.transAxes, lw=lw, color=color)
        ax.text(0.85+ruler*scale*pixel_scale, 0.12, f'{ruler:.0f}%', transform=ax.transAxes, fontsize=12, 
                color='white', ha='center')

    if (image is not None) and show_image:
        im = ax.imshow(image, origin='lower', cmap=cmap, **kwargs)
        if show_cbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.set_ylabel('[mJy/beam]', fontsize=fontsize)
    ax.add_collection(lc)
    return ax

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
