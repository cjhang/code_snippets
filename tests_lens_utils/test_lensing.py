
import pytest

import numpy as np
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt

from lens_utils.light_profile import SersicSource, SourceProfile
from lens_utils.mass_profile import SIELens, ExternalShear, MassPotential
from lens_utils.lensing import ray_trace


def test_lensing():

    xim = np.arange(-2.,2.,.02)
    yim = np.arange(-2.,2.,.02)

    xim,yim = np.meshgrid(xim,yim)

    zLens,zSource = 0.8,5.656
    xLens,yLens = 0.,0.
    MLens,eLens,PALens = 2.87e11,0.5,70.
    xSource,ySource,FSource = 0.216,0.24,0.02 # arcsec, arcsec, Jy
    aSource,nSource,arSource,PAsource = 0.1,0.5,1.0,120.-90 # arcsec,[],[],deg CCW from x-axis
    shear,shearangle = 0.12, 120.


    SIE = SIELens(zLens,xLens,yLens,MLens,eLens,PALens)
    Shear = ExternalShear(shear,shearangle)
    Source = SersicSource(zSource,True,xSource,ySource,FSource,aSource,nSource,arSource,PAsource)
    Mass = MassPotential(SIE, shear=Shear)
    Dd = cosmo.angular_diameter_distance(zLens).value
    Ds = cosmo.angular_diameter_distance(zSource).value
    Dds = cosmo.angular_diameter_distance_z1z2(zLens,zSource).value

    xsource,ysource = ray_trace(xim,yim,Mass,Dd,Ds,Dds)

    imbg = SourceProfile(xim,yim,Source,[SIE,Shear])
    imlensed = SourceProfile(xsource,ysource,Source,[SIE,Shear])


    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(imbg, origin='lower')
    ax.imshow(imlensed, origin='lower')
    plt.show()

