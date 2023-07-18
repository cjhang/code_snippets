#!/usr/bin/env python3
"""test file for the image_tools """

import os
from subprocess import getstatusoutput, getoutput
import pytest

import numpy as np
import matplotlib.pyplot as plt

from python_tools.image_tools import make_gaussian_image, source_finder, measure_flux, aperture_correction


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
    assert abs(sf1['theta'] - np.pi/12) < 0.05
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
