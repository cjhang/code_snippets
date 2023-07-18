# utilities for lensing

import numpy as np
from astropy.cosmology import Planck18

def ray_trace(xim, yim, potentials, Dd, Ds, Dds):
    """
    Wrapper to pass off lensing calculations to any number of functions
    defined below, accumulating lensing offsets from multiple lenses
    and shear as we go.
    """
    potentials = list(np.array([potentials]).flatten())
    ximage = xim.copy()
    yimage = yim.copy()
    
    for lens in potentials:
          deflected_x, deflected_y = lens.deflect(xim,yim,Dd,Ds,Dds)
          ximage += deflected_x; yimage += deflected_y
    
    return ximage,yimage

def einstain_ring(ML, zL, zS, cosmology=None):
    if cosmology is None:
        cosmology = Planck18
    Dd = cosmo.angular_diameter_distance(zL).value # in Mpc
    Ds = cosmo.angular_diameter_distance(zS).value
    Dds= cosmo.angular_diameter_distance_z1z2(zL,zS).value
    
    thetaE = np.sqrt((4*G*ML*Msun*Dds) / (c**2 * Dd*Ds*Mpc)) * rad2arcsec
    
    return thetaE
 

