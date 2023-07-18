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

def make_pola(data, mask=None):
    """calculate the polarisation angle

    Args: 
        data: the 3-D data, with the first dimesion is the Stokes dimension
        mask: 2D mask for the maps, applied to all the Stokes
    
    Return:
        The raidan angle of the polarisation
    """
    Q = data[1]
    U = data[2]
    if mask is not None:
        pola = 0.5*(np.arctan2(U[mask], Q[mask]))
    else:
        pola = 0.5*(np.arctan2(U, Q))
    return pola


def make_poli(data, norm=1, mask=None):
    """calculate the polarisation intensity

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
    return np.sqrt(data[1]**2+data[1]**2)/norm

def show_vectors(image, pola, poli=None, step=1, scale=1, rotate=0, mask=None, ax=None, 
                 edgecolors='white', facecolors='cyan', lw=1):
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
        ax.imshow(image, origin='lower', cmap='magma')
    ax.add_collection(lc)
    return ax

