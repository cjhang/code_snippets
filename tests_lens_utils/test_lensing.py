
import numpy as np
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt

from lens_utils.light_profile import SersicSource
from lens_utils.mass_profile import SIELens, ExternalShear, MassPotential
from lens_utils.lensing import ray_trace

# test the lensing with image_slider
from python_tools.utils import image_slider

def test_lensing_slider():
    # create the uniform grid
    xim = np.arange(-2.,2.,.02)
    yim = np.arange(-2.,2.,.02)
    xgrid, ygrid = np.meshgrid(xim, yim)

    ## define the source properties
    # z_lens, z_source = 0.8, 5.656
    # x_lens, y_lens = 0., 0. # in arcsec
    # m_lens = 2.87e11 # in solar mass
    # ell_lens = 0.5 # ellipicity
    # PA_lens = 70. # in radian
    # x_source, y_source = 0.216, 0.24 # in arcsec
    # flux_source = 0.02 # Jy
    # Re_source, ell_source = 0.1, 0.5
    # nsersic_source = 0.5
    # theta_source = -np.pi/2 # arcsec,[],[],deg CCW from x-axis
    # shear,shearangle = 0.12, 120.

    def _slider_wrapper(xgrid, ygrip, z_lens=0.8, z_source=5.656, 
                        x_lens=0., y_lens=0., m_lens=2.87e11, ell_lens=0.5, theta_lens=np.pi/4., 
                        shear=0.12, shearangle=2*np.pi/3,
                        x_source=0.2, y_source=0.24, flux_source=0.02, Re_source=0.1, ell_source=0.5, 
                        nsersic_source=0.5, theta_source=-np.pi/2,
                        ):
        SIE = SIELens(z_lens, x_lens, y_lens, m_lens, ell_lens, theta_lens)
        Shear = ExternalShear(shear, shearangle)
        Source = SersicSource(z=z_source, x=x_source, y=y_source, 
                              flux=flux_source,Re=Re_source, n=nsersic_source, ell=ell_source, theta=theta_source)
        Mass = MassPotential(SIE, shear=Shear)
        Dd = cosmo.angular_diameter_distance(z_lens).value
        Ds = cosmo.angular_diameter_distance(z_source).value
        Dds = cosmo.angular_diameter_distance_z1z2(z_lens, z_source).value

        xgrid_lensed, ygrid_lensed = ray_trace(xgrid, ygrid, Mass, Dd, Ds, Dds)

        return Source.profile_on_grid(xgrid_lensed, ygrid_lensed) + Source.profile_on_grid(xgrid, ygrid)

    res = image_slider(_slider_wrapper, [xgrid, ygrid], 
                 {'z_lens':(0.8, [0,2]), 'z_source':(5.656, [2,8]),
                  'x_lens':(0, [-2,2]), 'y_lens':(0,[-2,2]), 'm_lens':(3e11, [3e9,3e12]), 
                  'ell_lens':(0.5, [0.,1]), 'theta_lens':(np.pi/4, [0,np.pi]), 
                  'shear':(0.1, [0.,0.8]), 'shearangle':(2*np.pi/3, [0,np.pi]),
                  'x_source':(0.2, [-2,2]), 'y_source':(0.24, [-2,2]), 
                  'flux_source':(0.02,[1e-3,4e-2]), 
                  'Re_source':(0.1, [0,0.5]), 'ell_source':(0.5, [0,1]), 
                  'nsersic_source':(0.5, [0.2,4]), 
                  'theta_source':(-np.pi/2, [-np.pi,np.pi]),
                  }, 
                 origin='lower')
    print(res)

