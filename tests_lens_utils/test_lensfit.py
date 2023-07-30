# this file includes the rountines used for the lens fiting

from scipy.optimize import minimize

import numpy as np
from astropy.cosmology import Planck18 as cosmo
import matplotlib.pyplot as plt

from lens_utils.light_profile import SersicSource, GaussSource
from lens_utils.mass_profile import SIELens, ExternalShear, MassPotential
from lens_utils.lensing import ray_trace


xim = np.arange(-2.,2.,.02)
yim = np.arange(-2.,2.,.02)

xgrid, ygrid = np.meshgrid(xim,yim)

# set up the lens
z_lens = 0.8
z_source = 5.656
x_lens, y_lens = 0., 0. # in arcsec
m_lens, ell_lens, theta_lens = 2.87e11, 0.5, 70/180*np.pi
shear,shearangle = 0.12, 2/3*np.pi

SIE = SIELens(z=z_lens, x=x_lens, y=y_lens, mass=m_lens, ell=ell_lens, theta=theta_lens)
Shear = ExternalShear(shear, shearangle)
Mass = MassPotential(SIE, shear=Shear)

# construct the lensing
Dd = cosmo.angular_diameter_distance(z_lens).value
Ds = cosmo.angular_diameter_distance(z_source).value
Dds = cosmo.angular_diameter_distance_z1z2(z_lens,z_source).value
xgrid_lensed, ygrid_lensed = ray_trace(xgrid, ygrid, Mass, Dd, Ds, Dds)


# set up the sources
x_source, y_source, flux_source = 0.216, 0.24, 0.02 # arcsec, arcsec, Jy
Re_source, ell_source,theta_source, n_source, = 1.0, 0.5, -np.pi/2, 0.5 
source_model = SersicSource(z=z_source, x=x_source, y=y_source, flux=flux_source, 
                            Re=Re_source, ell=ell_source, theta=theta_source, n=n_source)
image_model = source_model.profile_on_grid(xgrid_lensed, ygrid_lensed)


# fit with the gaussian source
#init_guess = [0.2, 0.2, 0.02, 0.1] # x, y, flux, width
#source_func = GaussSource

# fit with the sersic source
init_guess = [x_source-0.1, y_source+0.1, flux_source*0.8, Re_source*0.8, ell_source-0.1, theta_source+np.pi/10, n_source+1]
source_func = SersicSource

source_guess = source_func(z_source, *init_guess)
image_guess = source_guess.profile_on_grid(xgrid_lensed, ygrid_lensed)

def fit_image(image, inits=None, constraints=None, bounds=None):

    # imbg = SourceProfile(xim,yim,Source,[SIE,Shear])

    def costf(x):
        # the cost function of the modelling
        source = source_func(z_lens, *x)
        #source = GaussSource(z=zLens, lensed=True, *x)
        imlensed = source.profile_on_grid(xgrid_lensed, ygrid_lensed)
        return np.sum((imlensed - image)**2)

    res = minimize(costf, inits, constraints=constraints, bounds=bounds)

    return res

res = fit_image(image_model, init_guess)
print("Initial guess: {}".format(init_guess))
print("Best fit: {}".format(res.x))

source_fit = source_func(z_source, *res.x)
image_fit = source_fit.profile_on_grid(xgrid_lensed, ygrid_lensed)
image_residual = image_fit - image_model

fig = plt.figure(figsize=(18,4))
ax = fig.subplots(1,4)
ax[0].imshow(image_model, origin='lower')
ax[1].imshow(image_guess, origin='lower')
ax[2].imshow(image_fit, origin='lower')
ax[3].imshow(image_residual, origin='lower')
plt.show()

