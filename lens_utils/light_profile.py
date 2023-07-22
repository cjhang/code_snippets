import numpy as np
import scipy.special as spsc

from .utils import Parameter

arcsec2rad = np.pi/(180.*3600.)
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.
rad2deg = 180./np.pi

class GaussSource(object):
    """Gauss Source
    """
    def __init__(self, z, x0=None, y0=None, flux=None, width=None):
        """input handling.

        Parameters
        ----------
        z : float
            Redshift
        x0 : float
            Coordinates in x axis, the real units depends on the grid
        y0 : float
            Coordinates in y axis
        flux : float
            The total flux of the Gausian
        width : float
            The FWHM of the Gaussian
        """
        if not isinstance(x0, Parameter):
              x0 = Parameter(x0)
        if not isinstance(y0, Parameter):
              y0 = Parameter(y0)
        if not isinstance(flux, Parameter):
              flux = Parameter(flux)
        if not isinstance(width, Parameter):
              x0 = Parameter(x0)

        self.z = z
        self.x0 = x0
        self.y0 = y0
        self.flux = flux
        self.width = width

    def profile_on_grid(self, xgrid, ygrid):
        """Calculate the projection on the grid

        Parameters
        ----------
        xgrid : numpy.ndarray
            2D array including all the x coordinates
        ygrid : numpy.ndarray
            2D array including all the y coordinates
            
        Return : 2D array
        """
        sigma = self.width['value']
        amp   = self.flux['value']/(2.*np.pi*sigma**2.)
        xs = self.xoff['value']
        ys = self.yoff['value']
        return amp * np.exp(-0.5 * (np.sqrt((xgrid-xs)**2.+(ygrid-ys)**2.)/sigma)**2.)

 
class SersicSource(object):
    """Sersic source
    """

    def __init__(self, z=None, x0=None, y0=None, flux=None, Re=None,
                 ell=None, theta=None, n=None):
        """input handling
    
        Parameters
        ----------
        z : float
            Redshift
        x0 : float
            Coordinates in x axis, the real units depends on the grid
        y0 : float
            Coordinates in y axis
        flux : float
            The total flux of the source
        Re : float
            Effective radius
        ell : float
            Ellipicity, ell = 1-a/b
        theta : float
            Position angle, in radian, CCW from the lens major axis
        n : float
            Sersic index
        """
        if not isinstance(x0, Parameter):
            x0 = Parameter(x0)
        if not isinstance(y0, Parameter):
            y0 = Parameter(y0)
        if not isinstance(flux, Parameter):
            flux = Parameter(flux)
        if not isinstance(Re, Parameter):
            Re = Parameter(Re)
        if not isinstance(ell, Parameter):
            ell = Parameter(ell)
        if not isinstance(theta, Parameter):
            theta = Parameter(theta)
        if not isinstance(n, Parameter):
            n = Parameter(n)
        # initial all the parameters
        self.z = z
        self.x0 = x0
        self.y0 = y0
        self.flux = flux
        self.Re = Re
        self.ell = ell
        self.theta = theta
        self.n = n

    def profile_on_grid(self, xgrid, ygrid):
        """calculate the profile based on the provided grids

        Parameters
        ----------
        xgrid : numpy.ndarray
            2D array including all the x coordinates
        ygrid : numpy.ndarray
            2D array including all the y coordinates
            
        Return : 2D array

        """
        xs = self.x0.value
        ys = self.y0.value
        theta = self.theta.value
        axis_ratio = 1 - self.ell.value
        Re = self.Re.value
        sersic_n = self.n.value
        flux = self.flux.value
        dx = (xgrid-xs)*np.cos(theta) + (ygrid-ys)*np.sin(theta)
        dy = (-(xgrid-xs)*np.sin(theta) + (ygrid-ys)*np.cos(theta))/axis_ratio
        rmap = np.sqrt(dx**2. + dy**2.)
        
        bn = spsc.gammaincinv(2. * sersic_n, 0.5)
        
        # Backing out from the integral to R=inf of a general sersic profile
        Ieff = flux * bn**(2*sersic_n) / (2*np.pi*Re**2*axis_ratio*np.exp(bn)*sersic_n*spsc.gamma(2*sersic_n))
        
        return Ieff * np.exp(-bn*((rmap/Re)**(1./sersic_n)-1.))
 
