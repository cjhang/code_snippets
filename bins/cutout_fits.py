"""this tools used to cutout fits files
"""

import os
import argparse
import numpy as np
import astropy.wcs as wcs
import astropy.coordinates as coordinates
from astropy.io import fits


def read_fits(filename, n=0):
    with fits.open(filename) as hdu:
        data = hdu[n].data
        header = hdu[n].header
    return data, header

def cutout_fits(filename=None, data=None, header=None, outfile=None, n=0, box=None, 
                pixel_center=None, pixel_radius=None, sky_center=None, sky_radius=None,
                drop_stokes=False):
    """
    Args:
        pixel_center: [pc-pr:pc+pr]
        box: [x_low, y_low, x_up, y_up] or 'x_low,y_low,x_up,y_up'
        sky_center: [ra, dec], ra and dec can be float, which is in units of degree
                    ra, dec can be string, which is in the format of '00h42m30s' and '+41d12m00s'
                    or, it can be Quantity with units, this is not support for cmd
    """
    if not os.path.isdir(os.path.dirname(outfile)):
        os.system(f'mkdir -p {os.path.dirname(outfile)}')
    if filename:
        data, header = read_fits(filename, n=n)
    ndim = data.ndim
    shape = data.shape
    wcs_origin = wcs.WCS(header)
    if sky_radius is not None:
        pixel2arcsec_ra, pixel2arcsec_dec = abs(header['CDELT1'])*3600, abs(header['CDELT2'])*3600
        pixel2arcsec = 0.5 * (pixel2arcsec_ra + pixel2arcsec_dec)
        pixel_radius = sky_radius / pixel2arcsec
    if box is not None:
        if isinstance(box, str):
            idxs = np.array(box.split(',')).astype(int)
        elif isinstance(box, (list,tuple,np.ndarray)):
            idxs = box
    if sky_center is not None:
        pixels_coords = wcs.utils.skycoord_to_pixel(sky_center, wcs=wcs_origin)
        # round up the pixels to integrals
        pixel_center = [int(np.round(pixels_coords[0])), int(np.round(pixels_coords[1]))]
        print("pixel_center,", pixel_center)
    if pixel_center is not None:
        cx, cy = pixel_center
        if isinstance(pixel_radius, (list,tuple,np.ndarray)):
            px, py = pixel_radius
        else:
            px = py = pixel_radius
        idxs = [cx-px,cy-py,cx+px,cy+py]
    else:
        raise ValueError("No valid sub section for image could be applied!")
    image_slices = np.s_[idxs[1]:idxs[-1],idxs[0]:idxs[-2]]
    if ndim == 2:
        array_slice = image_slices
    elif ndim == 3:
        array_slice = [np.s_[:],image_slices[0],image_slices[1]]
    elif ndim == 4:
        array_slice = [np.s_[:],np.s_[:],image_slices[0],image_slices[1]]
    else:
        print(f'input data dimension and shape are: {ndim} {shape}')
        raise ValueError("Wrong dimension of the input data!")
    
    # slice the data and wcs
    data_sliced = data[array_slice]
    wcs_sliced = wcs_origin[array_slice]
    
    # for full axes: ['longitude','latitude','spectral','stokes']
    if drop_stokes:
        do_drop_stokes = True
        if ndim != 4: 
            do_drop_stokes = False
            print("Warning: no stokes axis is found! Skipping...")
        if shape[-1] > 1:
            do_drop_stokes = False
            print("Warning: multiple stokes is found. Skipping... ")
        if do_drop_stokes:
            axes_sub = ['longitude','latitude','spectral']
            wcs_sliced = wcs_sliced.sub(axes_sub)
            data_sliced = data_sliced[0]
    if outfile is not None:
        primary_hdu = fits.PrimaryHDU(data=data_sliced, header=wcs_sliced.to_header())
        primary_hdu.writeto(outfile, overwrite=True)
    return data_sliced, wcs_sliced

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cutout fits files")
    parser.add_argument('infile', type=str, help='input filename')
    # Optional arguments
    parser.add_argument('--pixc', type=str, default=None, help='pixel center, str: x,y')
    parser.add_argument('--pixr', type=int, default=None, help='pixel radius, int: n')
    parser.add_argument('--skyr', type=float, help='sky radius, float: arcsec')
    parser.add_argument('--ra', type=str, default=None, help='sky coordinate ra, str: 1h2m3s')
    parser.add_argument('--dec', type=str, default=None, help='sky coordinate dec, str: 1d2m3s')
    parser.add_argument('--box', type=str, help='box selection', default=None)
    parser.add_argument('--outfile', type=str, help='output filename', default='output.fits')
    parser.add_argument('--ext', type=int, default=0,
                        help='the extention of the header, default to be 0')
    # Switch arguments
    parser.add_argument('--drop_stokes', action='store_true',
                        help='A boolean switch to drop stokes dimenton if it has only one dimention')
    parser.add_argument('--debug', action='store_true',
                        help='dry run and print out all the input parameters')
    args = parser.parse_args()
    # debug:
    if args.debug:
        print(f'infile is {args.infile}')
        print(f'pixc is {args.pixc}')
        print(f'pixr is {args.pixr}')
        print(f'skyr is {args.skyr}')
        print(f'ra, dec is {args.ra, args.dec}')
        print(f'--outfile is {args.outfile}')
        print(f'--box is {args.box}')
        print(f'--ext is {args.ext}')
        print(f'--drop_stokes is {args.drop_stokes}')
    else:
        if (args.ra is not None) and (args.dec is not None):
            sky_center = coordinates.SkyCoord(ra=args.ra, dec=args.dec)
        else:
            sky_center = None
        if args.pixc is not None:
            pixel_center = list(map(int, args.pixc.split(',')))
        else:
            pixel_center = None
        _,_ = cutout_fits(args.infile, pixel_center=pixel_center, pixel_radius=args.pixr, 
                    sky_center=sky_center, sky_radius=args.skyr,
                    outfile=args.outfile, box=args.box, drop_stokes=args.drop_stokes,
                    n=args.ext)

