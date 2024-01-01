#!/usr/bin/env python3
"""test file for the image_tools """

import os
from subprocess import getstatusoutput, getoutput
import pytest

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

from astropy.wcs import WCS

from python_tools.image_tools import Image, make_gaussian_image, source_finder, measure_flux, aperture_correction

def test_create_image():
    # generate the image data
    image = make_gaussian_image((181,181), fwhm=(12,8), theta=np.pi/12, area=100)
    noise = (np.random.randn(181*181)).reshape((181,181))*0.1
    kernel = make_gaussian_image((41,41), fwhm=(10,10), theta=0, area=1, normalize=0)
    noise_convolved = convolve(noise, kernel, mode='same')
    image_observed = image + noise_convolved
    # create a mock wcs
    wcs_mock = WCS(naxis=2)
    wcs_mock.wcs.crpix = 90, 90
    wcs_mock.wcs.cdelt = -0.1/3600, 0.1/3600
    wcs_mock.wcs.cunit = 'deg', 'deg'
    wcs_mock.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs_mock.array_shape = [181, 181]
    # create an Image and test read the image
    # test image wishout beam
    im1 = Image(data=image_observed, wcs=wcs_mock)
    im1.writefits('/tmp/image_tmp.fits', overwrite=1)
    im1_read = Image.read('/tmp/image_tmp.fits')
    assert im1_read.shape == (181,181)
    # test with beam
    im1 = Image(data=image_observed, wcs=wcs_mock, beam=(1.,1.,0))
    im1.writefits('/tmp/image_tmp.fits', overwrite=1)
    im1_read = Image.read('/tmp/image_tmp.fits')

def test_source_finder_sep(plot=False):
    data = make_gaussian_image(imsize=(21,21), fwhm=(4,2), offset=(1,-1), theta=np.pi/12, area=1)
    std = np.max(data)*0.1 # snr=10
    noise = np.random.random((21,21)) * std
    image = data + noise
    fmax = np.max(image)
    assert image.shape == (21,21)
    sources_found = source_finder(image, std=0.01, plot=plot)
    assert len(sources_found) == 1
    sf1 = sources_found[0]
    print(sf1['x', 'y', 'theta', 'peak_value'])
    assert abs(sf1['x']-11) < 1
    assert abs(sf1['y']-9) < 1
    assert abs(sf1['theta'] - np.pi/12) < 0.1
    assert abs(sf1['peak_value'] - fmax) < 0.1*fmax
    if plot: 
        plt.show()

def test_measure_flux(plot=False):
    flux_in = 1
    image = make_gaussian_image((21,21), fwhm=(4,2), offset=(1, -1), theta=np.pi/12, area=flux_in)
    flux_table = measure_flux(image, coords=[[12,10],], apertures=[4,2,np.pi/12], plot=plot)
    if plot:
        plt.title(f"flux={flux_in}")
        plt.show()
    assert abs(flux_table['flux'] - flux_in) < 0.2*flux_in

def test_measure_flux_with_noise(plot=False, accurary=0.1):
    # 
    flux_in = 100
    image = make_gaussian_image((181,181), fwhm=(12,8), theta=np.pi/12, area=flux_in)
    noise = (np.random.randn(181*181)).reshape((181,181))*0.1
    kernel = make_gaussian_image((41,41), fwhm=(10,10), theta=0, area=1, normalize=0)
    noise_convolved = convolve(noise, kernel, mode='same')
    image_with_noise = image + noise
    image_with_correlated_noise = image + noise_convolved
    flux_gaussian = measure_flux(image_with_noise, coords=(90,90), 
                                 apertures=(18,10,np.pi/12), method='gaussian', 
                                 plot=False, debug=False, noise_fwhm=None)
    flux_gaussian2 = measure_flux(image_with_correlated_noise, coords=(90,90), 
                                 apertures=(18,10,np.pi/12), method='gaussian', 
                                 plot=False, debug=False, noise_fwhm=10)
    flux_aperture = measure_flux(image_with_noise, coords=(90,90), 
                                 apertures=(18,10,np.pi/12), method='single-aperture', 
                                 plot=False, debug=False)
    flux_aperture2 = measure_flux(image_with_correlated_noise, coords=(90,90), 
                                 apertures=(18,10,np.pi/12), method='single-aperture', 
                                 plot=False, debug=False)
    print(flux_gaussian)
    print(flux_gaussian2)
    print(flux_aperture)
    print(flux_aperture2)
    assert abs(flux_gaussian['flux'] - flux_in) < accurary*flux_in
    assert abs(flux_gaussian2['flux'] - flux_in) < accurary*flux_in
    assert abs(flux_gaussian['flux_err']) < accurary*flux_in
    assert abs(flux_gaussian2['flux_err']) < abs(flux_gaussian['flux_err'])
    assert abs(flux_aperture['flux'] - flux_in) < accurary*flux_in
    assert abs(flux_aperture['flux_err']) < accurary*flux_in

    # create a mock wcs
    wcs_mock = WCS(naxis=2)
    wcs_mock.wcs.crpix = 90, 90
    wcs_mock.wcs.cdelt = -0.1/3600, 0.1/3600
    wcs_mock.wcs.cunit = 'deg', 'deg'
    wcs_mock.wcs.ctype = 'RA---TAN', 'DEC--TAN'
    wcs_mock.array_shape = [181, 181]
    # create an Image
    im1 = Image(data=image_with_correlated_noise, wcs=wcs_mock, beam=(1.,1.,0))
    im1.writefits('/tmp/image_tmp.fits', overwrite=1)

def test_aperture_correction(debug=False):
    flux_in = 1
    image = make_gaussian_image(imsize=(11,11), fwhm=(4,2,0), area=flux_in)
    flux_table = measure_flux(image, coords=[[5.,5.],], apertures=[4,2,0], plot=debug)
    correction = aperture_correction(aperture=[4,2], psf='Gaussian', fwhm=(4,2), debug=debug)
    # print(f"expected: {flux_in/flux}")
    # print(f"correction = {correction}")
    assert abs(flux_in/flux_table['flux'] - correction) < 1e-3
    if debug:
        plt.show()
