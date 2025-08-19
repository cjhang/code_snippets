#!/usr/bin/env python

"""
Authors: Jianhang Chen
Email: cjhastro@gmail.com

History:
    2024-11-08: modeling become a separate module, v0.1
"""

__version__ = '0.1.2'


import numpy as np
from numpy import linalg
import warnings
from multiprocessing import Pool
from scipy import interpolate, special, optimize
import matplotlib.pyplot as plt
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18 as cosm
import time

import emcee

###############################################
# Parameter
###############################################
def gaussian_log_prior(x, mean, sigma):
    return np.log(1.0/(np.sqrt(2*np.pi)*sigma))-0.5*(x-mean)**2/sigma**2

class Parameter:
    """the base class of model parameter
    """
    def __init__(self, value=None, limits=None, fprior=None, prior=None,
                 fixed=None, name=None):
        # initial all the build_in properties
        self.value = self.limits = self._fixed = None
        self.fprior = self.prior = None
        if (name is not None) and ('.' in name):
            raise ValueError("'.' is not allowed in the name!")
        self.name = name
        # initialize from inputs
        if isinstance(value, Parameter):
            for item, value in value.__dict__.items():
                self.__dict__[item] = value
        else:
            self.value = value
        if limits is not None: self.limits = limits
        if prior is not None: self.prior = prior
        if fprior is not None: self.fprior = fprior
        if fixed is not None: self._fixed = fixed
        if name is not None: self.name = name
        if self.fprior is None:
            if prior is not None:
                self.fprior = lambda x: gaussian_log_prior(x, *prior)
    @property
    def is_fixed(self):
        if self._fixed is not None:
            return self._fixed
        if self.limits is None:
            return True
        else:
            return False
    
    def set_prior(self, prior, fprior=None):
        self.prior = prior
        if self.fprior is None:
            self.fprior = gaussian_log_prior
 
    def get_log_prior(self, value):
        if self.is_fixed:
            return 0
        limits = self.limits
        if limits[0] <= value <= limits[1]:
            if self.prior is None:
                return 0
            else:
                return self.fprior(value, *self.prior)
        else:
            return -np.inf


###############################################
# Grid 
###############################################
class Grid:
    """the base class of grid to calculate the model
    """
    def __init__(self):
        pass
    @property
    def grid(self):
        if self.ndim == 1:
            return (self.x,)
        elif self.ndim == 2:
            # return self.meshgrid
            return (self.y, self.x)
        elif self.ndim == 3:
            return (self.z, self.y, self.x)

class Grid1D(Grid):
    def __init__(self, size=None, pixelsize=1, center=0.):
        super().__init__()
        self.ndim = 1
        self.size = size
        self.pixelsize = pixelsize
        self.center = center
        self._x = None
    def _create_grid(self):
        return (np.arange(self.size) - self.size*0.5 + 0.5) * self.pixelsize + self.center
    @property
    def x(self):
        if self._x is not None:
            return self._x
        else:
            return self._create_grid()
    @x.setter
    def x(self, x):
        self._x = x
        self.size = len(self._x)
        self.pixelsize = x[1] - x[0]
        self.center = np.mean(x)

class Grid2D(Grid):
    def __init__(self, size=None, pixelsize=1, center=(0.,0.)):
        """the class to hand the gridding of the models
        """
        super().__init__()
        self.ndim = 2
        if isinstance(size, int):
            self._ysize = self._xsize = size
        elif isinstance(size, (list,tuple)):
            if len(size) != 2:
                raise ValueError("Support only 2D array shape!")
            self._ysize, self._xsize = size
        else:
            self._ysize, self._xsize = None
        self.pixelsize = pixelsize
        self._ycenter, self._xcenter = center
        self._uvgrid = None
        self._x = None
        self._y = None
    @property
    def x(self):
        if self._x is not None:
            return self._x
        else:
            return (np.arange(self._xsize) - self._xsize*0.5 + 0.5) * self.pixelsize + self._xcenter
    @x.setter
    def x(self, x):
        self._x = x
        self._xsize = len(x)
        self._xcenter = np.mean(x)
    @property
    def y(self):
        if self._y is not None:
            return self._y
        else:
            return (np.arange(self._ysize) - self._ysize*0.5 + 0.5) * self.pixelsize + self._ycenter
    @y.setter
    def y(self, y):
        self._y = y
        self._ysize = len(x)
        self.pixelsize = y[1] - y[0]
        self._ycenter = np.mean(y)
    @property
    def meshgrid(self):
        return np.meshgrid(self.y, self.x, indexing='ij')
    @property
    def meshgrid_xy(self):
        return np.meshgrid(self.x, self.y, indexing='xy') 
    @property
    def uvgrid(self):
        if self._uvgrid is not None:
            return self._uvgrid
        kmax = 0.5/(self.pixelsize*np.pi/180/3600) # arcsec to radian
        self._uvgrid = np.linspace(-kmax, kmax, self.imsize)
        return self._uvgrid

###############################################
# Models
###############################################
class Model:
    """the base class of model
    """
    def __init__(self, name=None):
        if (name is not None) and ('.' in name):
            raise ValueError("'.' is not allowed in the name!")
        self._name = name
        self._parameters = None
        self._keymaps = None
    def __call__(self, *args):
        if isinstance(args[0], Grid):
            return self.model_on_grid(args[0])
        else:
            return self.evaluate(*args)
    @property
    def name(self):
        if self._name is None:
            return self.__class__.__name__
        else:
            return self._name
    @name.setter
    def name(self, name):
        self._name = name
    @property
    def keymaps(self):
        # return the mapping between the given name and internal name
        if self._keymaps is not None:
            return self._keymaps
        keymaps = {}
        params_list_keys = list(self.parameters.keys())
        for key_model, value in self.__dict__.items():
            if not isinstance(value, Parameter):
                continue
            if not value.is_fixed:
                if value.name in params_list_keys:
                    keymaps[value.name] = key_model
                else:
                    keymaps[key_model] = key_model
        self._keymaps = keymaps
        return self._keymaps
    @property
    def parameters(self):
        # get all the parameters
        # if self._parameters is not None:
            # return self._parameters
        parameters = {}
        for key,value in self.__dict__.items():
            if not isinstance(value, Parameter):
                continue
            if not value.is_fixed:
                if value.name is not None:
                    parameters[value.name] = value.value
                else:
                    parameters[key] = value.value
        # self._parameters = parameters
        return parameters
    def update_parameters(self, params):
        params_list_keys = list(self.parameters.keys())
        keymaps = self.keymaps # used to map the key name to the __dict__.keys
        if isinstance(params, dict):
            for key, value in params.items():
                if key in params_list_keys:
                    self.__dict__[keymaps[key]].value = value
        else:
            for key, value in zip(params_list_keys, params):
                self.__dict__[keymaps[key]].value = value
    @property
    def limits(self):
        limits = {}
        for key,value in self.__dict__.items():
            if not isinstance(value, Parameter):
                continue
            if not value.is_fixed:
                if value.name is not None:
                    limits[value.name] = value.limits
                else:
                    limits[key] = value.limits
        # self._parameters = parameters
        return limits 
    def get_priors(self, params_dict=None):
        if params_dict is None:
            params_dict = self.parameters
        log_priors = 0
        for key,value in params_dict.items():
            log_priors += self.__dict__[self.keymaps[key]].get_log_prior(value)
        return log_priors

    def model_on_grid(self, grid):
        if grid.ndim == 1:
            return self.evaluate(grid.x)
        if grid.ndim == 2:
            return self.evaluate(*grid.meshgrid)

class Models:
    """
    If a model compose of several same Models, they should have different name
    """
    def __init__(self, models):
        if isinstance(models, (list, tuple)):
            self.models = models
        else:
            self.models = [models,]
        self._parameters = None
        # assigned new names for these who share the same name
        model_names = []
        name_idx = 1
        for model in self.models:
            if model.name not in model_names:
                model_names.append(model.name)
            else:
                model.name = model.name + f'_{name_idx}'
                name_idx += 1
    def __getitem__(self, i):
        return self.models[i]
    @property
    def parameters(self):
        parameters = {}
        for model in self.models:
            model_parameters = model.parameters
            model_keymaps = model.keymaps
            for key,value in model_parameters.items():
                if key in parameters.keys():
                    #check whether it is the same Parameter
                    if parameters[key] != value:
                        warnings.warn("Different paramteres share a same name!")
                else:
                    if key != model_keymaps[key]: # user-defined paramter
                        parameters[key] = value
                    else:
                        parameters[model.name+'.'+key] = value
        return parameters
    def update_parameters(self, params):
        if not isinstance(params, dict):
            parameter_keys = self.parameters.keys()
            params = dict(zip(parameter_keys, params))
        for model in self.models:
            model_parameters_keys = model.parameters.keys()
            for name, value in params.items():
                if name in model_parameters_keys:
                    model.update_parameters({name: value})
                elif model.name == name.split('.')[0]:
                    model.update_parameters({name.split('.')[-1]: value})

    # def parameters_old(self, params_dict):
        # params_keys = list(self.parameters.keys())
        # models_params_dict = {}
        # # get model specified paramters
        # for model in self.models:
            # model_params = []
            # for name in params_dict.keys():
                # if model.name in name:
                    # model_params.append(name)
            # models_params_dict[model.name] = model_params
        # # get the shared paramters
        # union_params = set()
        # for params in models_params_dict.values()
            # union_params = union_params.union(params)
        # common_params = list(set(params_keys) - union_params)
        # # passing through paramters to each model

    def get_parameters(self):
        return self.parameters
        # params_list = {}
        # for model in self.models:
            # for key,value in model.__dict__.items():
                # if not isinstance(value, Parameter):
                    # continue
                # if not value.is_fixed:
                    # if value.name not in params_list.keys():
                        # params_list[model.name+'.'+key] = value.value
        # return params_list
    def get_limits(self):
        limits = {}
        for model in self.models:
            model_limits = model.limits
            model_keymaps = model.keymaps
            for key,value in model_limits.items():
                if key in limits.keys():
                    #check whether it is the same Parameter
                    if limits[key] != value:
                        warnings.warn("Different paramteres share a same name!")
                else:
                    if key != model_keymaps[key]: # user-defined paramter
                        limits[key] = value
                    else:
                        limits[model.name+'.'+key] = value
        return limits 
        # limits_list = {}
        # for model in self.models:
            # for key,value in model.__dict__.items():
                # if not isinstance(value, Parameter):
                    # continue
                # if not value.is_fixed:
                    # limits_list[model.name+'.'+key] = value.limits
        # return limits_list
    # def update_parameters(self, params_dict):
        # for model in self.models:
            # for key in params_dict.keys():
                # model_name, model_param_name = key.split('.')
                # if model_param_name in model.__dict__.keys():
                    # model.__dict__[model_param_name].value = params_dict[key]
    def get_priors(self, params_dict=None):
        if params_dict is None:
            params_dict = self.parameters
        log_priors = 0
        for model in self.models:
            # log_priors += model.get_priors()
            model_parameters_keys = model.parameters.keys()
            for key,value in params_dict.items():
                if key in model_parameters_keys:
                    log_priors += model.get_priors({key:value})
                else:
                    model_name, model_param_name = key.split('.')
                    if model_name != model.name:
                        continue
                    log_priors += model.get_priors({model_param_name:value})
                # model_name, model_param_name = key.split('.')
                # if model_name != model.name:
                    # continue
                # if model_param_name in model.__dict__.keys():
                    # log_priors += model.__dict__[model_param_name].get_log_prior(params_dict[key])
        return log_priors
    def create_model(self, var):
        if isinstance(var, Grid):
            return self.model_on_grid(var)
        model_value = 0.
        for model in self.models:
            model_value += model(var)
        return model_value
    
    def model_on_grid(self, grid):
        # parameter_names = self.get_parameters()
        # self.update_parameters(dict(zip(parameter_names, theta)))
        model_value = 0.
        for model in self.models:
            model_value += model.model_on_grid(grid)
        return model_value

class Line1D(Model):
    def __init__(self, m=None, b=None, name=None):
        super().__init__(name=name)
        self.m = Parameter(m)
        self.b = Parameter(b)
    def evaluate(self, grid_1d):
        x = grid_1d.x
        return self.m.value * x + self.b.value

class ScaleUncertainty(Model):
    def __init__(self, factor=None, name=None):
        super().__init__(name=name)
        self.factor = Parameter(factor)
    def evaluate(self, x):
        return 0

class Gaussian1D(Model):
    def __init__(self, amplitude=None, mean=None, sigma=None, cont=0., name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude)
        self.mean = Parameter(mean)
        self.sigma = Parameter(sigma)
        self.cont = Parameter(cont)
    def evaluate(self, x):
        if self.cont is not None:
            cont_value = self.cont.value
        else:
            cont_value = 0
        return self.amplitude.value * np.exp(-0.5*(x-self.mean.value)**2/self.sigma.value**2) + cont_value

class Sersic1D(Model):
    def __init__(self, amplitude=None, reff=None, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude)
        self.reff = Parameter(sigma)
    def evaluate(self, x):
        pass

def gaussian_2d(grid, amplitude, y_sigma, x_sigma, y0, x0, theta):
    """
    grid = [ygrid, xgrid]
    """
    y, x = grid
    a = np.cos(theta)**2/(2*x_sigma**2) + np.sin(theta)**2/(2*y_sigma**2)
    b = np.sin(2*theta)/(2*x_sigma**2) - np.sin(2*theta)/(2*y_sigma**2)
    c = np.sin(theta)**2/(2*x_sigma**2) + np.cos(theta)**2/(2*y_sigma**2)
    return amplitude * np.exp(-a*(x-x0)**2-b*(x-x0)*(y-y0)-c*(y-y0)**2)

class Gaussian2D(Model):
    def __init__(self, amplitude=None, x0=None, y0=None, x_sigma=None, y_sigma=None, 
                 theta=None, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude)
        self.x0 = Parameter(x0)
        self.y0 = Parameter(y0)
        self.x_sigma = Parameter(x_sigma)
        self.y_sigma = Parameter(y_sigma)
        self.theta = Parameter(theta)
        self.name = name
    def evaluate(self, y, x):
        # y, x = grid2d.meshgrid
        y0, x0 = self.y0.value, self.x0.value
        y_sigma, x_sigma = self.y_sigma.value, self.x_sigma.value
        amplitude, theta  = self.amplitude.value, self.theta.value
        a = np.cos(theta)**2/(2*x_sigma**2) + np.sin(theta)**2/(2*y_sigma**2)
        b = np.sin(2*theta)/(2*x_sigma**2) - np.sin(2*theta)/(2*y_sigma**2)
        c = np.sin(theta)**2/(2*x_sigma**2) + np.cos(theta)**2/(2*y_sigma**2)
        return amplitude * np.exp(-a*(x-x0)**2-b*(x-x0)*(y-y0)-c*(y-y0)**2)

class Sersic2D(Model):
    def __init__(self, amplitude=None, reff=None, n=None, x0=None, y0=None, 
                 ellip=None, theta=None, name=None):
        super().__init__(name=name)
        self.amplitude = Parameter(amplitude)
        self.reff = Parameter(reff)
        self.n = Parameter(n)
        self.x0 = Parameter(x0)
        self.y0 = Parameter(y0)
        self.ellip = Parameter(ellip)
        self.theta = Parameter(theta)
        self.name = name
    def evaluate(self, y, x):
        # y, x = grid.meshgrid
        bn = special.gammaincinv(2.0 * self.n.value, 0.5)
        cos_theta = np.cos(self.theta.value)
        sin_theta = np.sin(self.theta.value)
        x_maj = np.abs((x - self.x0.value) * cos_theta + (y - self.y0.value) * sin_theta)
        x_min = np.abs(-(x - self.x0.value) * sin_theta + (y - self.y0.value) * cos_theta)

        b = (1 - self.ellip.value) * self.reff.value
        expon = 2.0
        inv_expon = 1.0 / expon
        z = ((x_maj / self.reff.value) ** expon + (x_min / b) ** expon) ** inv_expon
        return self.amplitude.value * np.exp(-bn * (z ** (1 / self.n.value) - 1.0))

def blackbody_nu(nu, temperature):
    """
    Args:
        nu: frequency, in Hz
        temperature: in Kelvin
        
    Return:
     Bv(v, T): erg/u.s/u.sr/u.cm**2/u.Hz 
    """
    const_h = const.h.cgs.value
    const_c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    return 2*const_h*nu**3 / const_c**2 / (np.exp(const_h*nu/(k_B*temperature))-1)

def blackbody_lambda(lam, temperature):
    """
    Args:
        lam: frequency, in cm
        temperature: in Kelvin

    Return:
     Bv(v, T): erg/u.s/u.sr/u.cm**2/u.Hz 
    """
    const_h = const.h.cgs.value
    const_c = const.c.cgs.value
    k_B = const.k_B.cgs.value
    return 2*const_h*const_c**2 / lam**5 / (np.exp(const_h*const_c/(
                                              lam*k_B*temperature))-1)
def dust_temperature_at_z(temperature, z, beta):
    """da Cunha et al. 2013

    Args:
        temperature: dust temperature in K
        z: redshift
        beta: emissivity
    """
    T0 = 2.73*(1+z)
    return (temperature**(4+beta)+T0**(4+beta)*((1+z)**(4+beta)-1))**(1/(4+beta))

def modified_blackbody_nu(nu, temperature=None, z=0, beta=2.0, dustmass=None, area=None):
    """flux density of modified blackbody
    Args:
        nu: restframe wavelength, in Hz
        Tdust: K
        Mdust: g
        A: cm2
    """
    kappa0 = 0.45 # cm2/g
    nu0 = 250*u.GHz.to(u.Hz)
    Tcmb = 2.73
    D_L = cosm.luminosity_distance(z).to(u.cm).value
    Td = dust_temperature_at_z(temperature, z, beta)
    tau_nu = dustmass/area * kappa0 *(nu/nu0)**beta
    B_Td = blackbody_nu(nu, Td)
    B_Tcmb = blackbody_nu(nu, Tcmb)
    return (1+z)*area/D_L**2*(1-np.exp(-tau_nu))*(B_Td - B_Tcmb)

class Blackbody(Model):
    def __init__(self, temperature=None, name=None):
        super().__init__(name=name)
        self.temperature = Parameter(temperature)
    def __call__(self, nu):
        """calculate the flux density
        Args:
            nu: frequency in GHz

        Return:
            spectral radiance: erg/s/sr/cm2/Hz
        """
        nu_Hz = nu * 1e9
        return blackbody_nu(nu_Hz, self.temperature.value)
    def evaluate(self, grid):
        nu = grid.x
        return self(nu, self.temperature.value)

class MBlackbody(Model):
    def __init__(self, temperature=None, z=None, beta=None,
                 dustmass=None, area=None, name=None):
        """Modified blackbody
        dustmass in solarmass
        area in kpc**2
        """
        super().__init__(name=name)
        self.temperature = Parameter(temperature)
        self.z = Parameter(z)
        self.beta = Parameter(beta)
        self.dustmass = Parameter(dustmass)
        self.area = Parameter(area)
    def evaluate(self, nu):
        """calculate the flux density
        Args:
            nu: frequency in GHz
            dustmass: Msun
            area: kpc
        Return:
            flux: in mJy
        """
        msun_to_g = 1.98841e33
        kpc2_to_cm2 = 9.521406e42
        nu_Hz = nu * 1e9
        cgs_to_mJy = 1e26
        mbb = modified_blackbody_nu(nu_Hz, 
                                    temperature = self.temperature.value,
                                    z = self.z.value,
                                    beta = self.beta.value,
                                    dustmass = self.dustmass.value*msun_to_g,
                                    area = self.area.value*kpc2_to_cm2)
        return cgs_to_mJy * mbb

class MBlackbody_log(Model):
    def __init__(self, temperature=None, z=None, beta=None,
                 dustmass=None, area=None, name=None):
        """Modified blackbody
        dustmass in solarmass
        area in kpc**2
        """
        super().__init__(name=name)
        self.temperature = Parameter(temperature)
        self.z = Parameter(z)
        self.beta = Parameter(beta)
        self.dustmass = Parameter(dustmass)
        self.area = Parameter(area)
    def evaluate(self, nu):
        """calculate the flux density
        Args:
            nu: frequency in GHz

        Return:
            flux: in mJy
        """
        msun_to_g = 1.98841e33
        kpc2_to_cm2 = 9.521406e42
        nu_Hz = nu * 1e9
        cgs_to_mJy = 1e26
        mbb = modified_blackbody_nu(nu_Hz, 
                                    temperature = self.temperature.value,
                                    z = self.z.value,
                                    beta = self.beta.value,
                                    dustmass = 10**self.dustmass.value*msun_to_g,
                                    area = self.area.value*kpc2_to_cm2)
        return cgs_to_mJy * mbb


###############################################
# Fitters
###############################################
class Fitter():
    def __init__(self, models, grid, data, name=None):
        self.models = models
        self.grid = grid
        self.data = data
        self.parameters = models.get_parameters()
        self.initial_guess = list(self.parameters.values())
        self.keys = list(self.parameters.keys())
        self.n_params = len(self.parameters)
        self.limits = models.get_limits()
        self.name = name
    def _calculate_diff(self):
        """internal function to compare the model and data
        """
        model = self.models.create_model(self.grid)
        data = self.data
        var = data['var']
        try: var_err = data['var_err']
        except: var_err = 0
        obs = data['data']
        try: obs_err = data['data_err']
        except: obs_err = 0
        sigma2 = var_err**2 + obs_err**2 #+ err2_addition

        if np.sum(sigma2 - 0.) < 1e-8:
            sigma2 = 1
        
        # interpolate the model grid to the observed coordinates
        if obs.ndim == 1:
            model_interp = interpolate.interpn(self.grid.grid, model, var)
        if obs.ndim == 2:
            model_interp = interpolate.interpn(
                    self.grid.grid, model, np.array([var[0].flatten(),var[1].flatten()]).T).reshape(obs.shape)
        if obs.ndim == 3:
            model_interp = interpolate.interpn(
                    self.grid.grid, model, 
                    np.array([var[0].flatten(),var[1].flatten(), var[2].flatten()
                                                  ]).T).reshape(obs.shape)
        diff2 = (obs-model_interp)**2/sigma2
        return diff2
 
def calculate_diff(models, grid, data):
    """internal function to compare the model and data
    """
    pass


def emcee_log_probability(theta, models, grid, data):
    models.update_parameters(theta)
    params_priors = models.get_priors()
    if not np.isfinite(params_priors):
        return -np.inf

    model = models.create_model(grid)
    var = data['var']
    try: var_err = data['var_err']
    except: var_err = 0
    obs = data['data']
    try: obs_err = data['data_err']
    except: obs_err = 0
    sigma2 = var_err**2 + obs_err**2 #+ err2_addition
    
    # interpolate the model grid to the observed coordinates
    if obs.ndim == 1:
        model_interp = interpolate.interpn(grid.grid, model, var)
    if obs.ndim == 2:
        model_interp = interpolate.interpn(
                grid.grid, model, 
                np.array([var[0].flatten(),var[1].flatten()]).T).reshape(obs.shape)
    if obs.ndim == 3:
        model_interp = interpolate.interpn(
                grid.grid, model, 
                np.array([var[0].flatten(),var[1].flatten(), var[2].flatten()
                                              ]).T).reshape(obs.shape)
    diff2 = (obs-model_interp)**2/sigma2
    return -0.5*np.sum(diff2) + params_priors

class EmceeFitter(Fitter):
    def __init__(self, models, grid, data, nwalkers=None, name=None,
                 log_probability=None, 
                 backend=None, backend_name='emceefitter', reset=True,
                 ):
        super().__init__(models=models, grid=grid, data=data, name=name)
        if nwalkers is None:
            self.nwalkers = 4*self.n_params
        else:
            self.nwalkers = nwalkers
        if log_probability is None:
            self.log_probability = emcee_log_probability
        else:
            self.log_probability = log_probability

        #setup the backend, initial posistions
        if backend is not None:
            self.backend = emcee.backends.HDFBackend(backend, name=backend_name)
            if reset:
                self.backend.reset(self.nwalkers, self.n_params)
            try:
                self.iterations = backend.iteration
            except:
                self.iterations = 0
        else:
            self.backend = None
            self.iterations = 0
        if self.iterations > 0:
            print(f"Found existing steps={iterations_finished}, continue")
            self.pos = None # continue the previous iterations
        else:
            limits = self.models.get_limits()
            limits_ranges = np.array(list(limits.values()))
            limits_diff = np.diff(limits_ranges).flatten()
            self.pos = np.array(self.initial_guess) \
                    + 1e-2*limits_diff[None,:]*np.random.randn(
                            self.nwalkers, self.n_params)
 
    def run(self, steps=5000, progress=True, force=False):
        """a simple run, with fixed steps 
        """
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers, self.n_params, self.log_probability, 
            args=(self.models, self.grid, self.data),
            backend=self.backend,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='emcee')
            self.sampler.run_mcmc(self.pos, steps, progress=progress, 
                                  skip_initial_state_check=force)

    def auto_run(self, max_steps=10000, progress=True, check_steps=1000):
        """run with automatic correlation time checking, stop automatically
        """
        self.sampler = emcee.EnsembleSampler(
            self.nwalkers, self.n_params, self.log_probability, 
            args=(self.models, self.grid, self.data), 
            backend=self.backend)
        check_steps = int(check_steps)
        index = 0
        old_tau = np.inf
        autocorr = np.empty(max_steps)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', module='emcee')
            for sample in self.sampler.sample(self.pos, iterations=max_steps, 
                                              progress=progress):
                if self.sampler.iteration % check_steps:
                    continue
                # compute the autocorrelation
                tau = self.sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                # check convergence
                converged = np.all(tau*check_steps < self.sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau

    def multi_run(self, steps=5000, ncores=None, progress=True, 
                  backend=None, backend_name='emceefitter', reset=True,
                 ):
        """a simple run, with fixed steps 

        Args:
            reset: set to False to continue the mcmc chain from existing backend
        """
        with Pool(ncores) as pool:
            self.sampler = emcee.EnsembleSampler(
                self.nwalkers, self.n_params, self.log_probability, 
                args=(self.models, self.grid, self.data),
                backend=self.backend, pool=pool)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', module='emcee')
                self.sampler.run_mcmc(self.pos, steps, progress=progress, 
                                      )

    def samples(self, discard=None, thin=None, flat=True):
        tau = self.sampler.get_autocorr_time()
        if discard is None:
            discard = int(2*np.max(tau))
        if thin is None:
            thin = int(0.5*np.min(tau))
        samples = self.sampler.get_chain(discard=discard, flat=flat, thin=thin)
        return samples 
    
    def plot(self, ax=None):
        samples = self.sampler.get_chain()
        if ax is None:
            fig = plt.figure(figsize=(16, 4))
        # ax_main = plt.subplot2grid((self.n_params,3), (0,0), rowspan=self.n_params)
        # ax_main.plot(self.x, self.y_data)
        # samples = self.sampler.get_chain()
        # inds = np.random.randint(len(samples), size=100)
        # for ind in inds:
            # sample = samples[ind]
            # models = self.models
            # models.update_parameters(dict(zip(self.keys, np.nanmedian(sample, axis=0))))
        #     ax_main.plot(self.x, models.create_model(self.x), 'tomato', alpha=0.1)
        for i in range(self.n_params):
            if ax is None:
                ax_i = plt.subplot2grid((self.n_params,2), (i,0), colspan=2)
            else:
                ax_i = ax[i]
            ax_i.plot(samples[:, :, i], "k", alpha=0.2)
            # if truths is not None:
                # ax.hlines(truths[i], 0, len(samples), 'r')
            ax_i.set_xlim(0, len(samples))
            if i<self.n_params-1: ax_i.axes.get_xaxis().set_ticks([])
            ax_i.text(0, 0.8, self.keys[i], ha='left', transform=ax_i.transAxes, fontsize=10)
        ax_i.set_xlabel('steps')
        if ax is None:
            plt.show()

    def corner_plot(self, truths=None):
        try:
            import corner
        except:
            print("Install corner for corner_plot")
            return
        samples = self.samples()
        fig = corner.corner(
                samples, labels=list(self.models.get_parameters().keys()),
                truths=truths
                )
        plt.show()

class MinimizeFitter(Fitter):

    def __init__(self, models, grid, data, name=None):
        super().__init__(models=models, grid=grid, data=data, name=name)
        limits = models.get_limits()
        limits_ranges = np.array(list(limits.values()))
        self.bounds = limits_ranges
        # self.vector_cost = None
    def cost(self, theta):
        self.models.update_parameters(theta)
        params_priors = self.models.get_priors()
        if not np.isfinite(params_priors):
            return np.inf
        cost2 = self._calculate_diff()
        return np.sum(cost2) - params_priors

    def vector_cost(self, theta):
        self.models.update_parameters(theta)
        params_priors = self.models.get_priors()
        cost2 = self._calculate_diff()
        return cost2.flatten() - params_priors

    def run(self, initial_guess=None, debug=False):
        """
        simple run, only fit with constrained minimize, no errorbar for parameters
        """
        if initial_guess is None:
            initial_guess = self.initial_guess
        # run a first round with constrained minimization
        fit_result = optimize.minimize(
                self.cost, initial_guess, 
                args=(), bounds=self.bounds)
        # run a second least_squre minimization to get the error bar
        # TODO: add option to reject outliers for second fit
        self.best_fit0 = fit_result.x
        self.best_fit = fit_result.x
        self.best_fit_error = None
        print(f'first fit: {self.best_fit0}')
        if False: # just for testing, going to be removed
            fit_result_lsq = optimize.leastsq(
                    self.vector_cost, fit_result.x, full_output=True,
                    args=()) #self.models, self.grid, self.data))
            popt, pcov, infodict, errmsg, ier = fit_result_lsq
            print(f'second fit: {popt}')
            self.best_fit = popt # replace with the new fitting
            data_size = len(infodict['fvec'])
            cost = np.sum(infodict['fvec'] ** 2)
            if (data_size > len(popt)) and (pcov is not None):
                reduce_chi2 = cost / (data_size - len(popt))
                print('cost', cost,'reduce_chi2', reduce_chi2)
                pcov = pcov * reduce_chi2
                self.best_fit_error = np.sqrt(np.diag(pcov))
            else: 
                self.best_fit_error = None

    def auto_run(self, initial_guess=None, debug=False, maxiter=None):
        """The default method to also return the errorbar of the parameters

        compared to 'run', the least_squares also supports boundaries
        """
        if initial_guess is None:
            initial_guess = self.initial_guess

        first_result = optimize.minimize(
                self.cost, initial_guess, 
                args=(),#self.models, self.grid, self.data), 
                bounds=self.bounds)
        # compute the chi2 of the first fit
        try:
            data_size = self.data['data'].size
        except:
            print("Failed to get the size of the data from data['data'].size")
            data_size = 1
        if debug:
            first_cost = self.cost(first_result.x)
            first_chi2 = first_cost/(data_size - len(initial_guess))
            print("Parameter:", list(self.models.parameters.keys()))
            print("first best_fit:", first_result.x)
            print(f"data_size: {data_size}; first_cost {first_cost};")
            print(f"Reduced chi2 of first minimize: {first_chi2}")
        # second time with least_squre fit, derive the errorbar
        res = optimize.least_squares(
                self.vector_cost, first_result.x, jac='3-point', 
                bounds=self.bounds.T, method='trf', loss='soft_l1',
                max_nfev=maxiter, verbose=debug,
                args=(), #(self.models, self.grid, self.data),
                )
        # adopted from scipy.curve_fit
        if not res.success:
            print('Warning: Optimal parameters not found: ' + res.message)
            # raise RuntimeError("Optimal parameters not found: " + res.message)

        infodict = dict(nfev=res.nfev, fvec=res.fun)
        ier = res.status
        errmsg = res.message

        data_size = len(res.fun)
        cost = 2 * res.cost  # res.cost is half sum of squares!
        popt = res.x

        # Do Moore-Penrose inverse discarding zero singular values.
        _, s, VT = linalg.svd(res.jac, full_matrices=False)
        threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
        s = s[s > threshold]
        VT = VT[:s.size]
        pcov = np.dot(VT.T / s**2, VT)

        if data_size > len(popt):
            reduce_chi2 = cost / (data_size - len(popt))
            fit_cost = np.sum(self.vector_cost(popt))
            reduce_chi2_mine = fit_cost/(data_size-len(initial_guess))
            pcov_scaled = pcov * reduce_chi2
            best_fit_error = np.sqrt(np.diag(pcov_scaled))
            best_fit_error2 = np.sqrt(np.diag(pcov))
            if debug:
                print(f"data_size: {data_size}; cost: {cost}")
                print(f"Reduced chi2 of leastsq: {reduce_chi2}")
                print(f"second cost: {fit_cost}")
                print(f"Reduced chi2 of first minimize (mine): {reduce_chi2_mine}")
                print(f"Error scaled: {best_fit_error}")
                print(f"Error pcov: {best_fit_error2}")
        else: 
            best_fit_error = None
        # self.best_fit0 = dict(zip(self.parameters, first_result.x))
        # self.best_fit = dict(zip(self.parameters, popt))
        self.best_fit0 = first_result.x
        self.best_fit = popt
        if best_fit_error is not None:
            self.best_fit_error = best_fit_error
        else:
            self.best_fit_error = None

def minimize_vector_cost_old(theta, models, grid, data):
    """
    args: [theta, models, grid, data]
    """
    # var, data, data_err = args
    model_parameters = models.parameters
    # for key, value in model_parameters.items():
        # print(key, value.value)
    # print('theta', theta)
    dict_theta = dict(zip(model_parameters.keys(), theta))
    models.update_parameters(dict_theta)
    params_priors = models.get_priors(dict_theta)
    # if not np.isfinite(params_priors):
        # print('inf of priors')
        # return -np.inf 
    model = models.create_model(grid)
    var = data['var']
    try: var_err = data['var_err']
    except: var_err = 0.0
    obs = data['data']
    try: obs_err = data['data_err']
    except: obs_err = 0.0
    for m in models.models:
        if isinstance(m, ScaleUncertainty):
            log_f = m.factor.value
            err2_addition = model**2*np.exp(2*log_f)
        else:
            err2_addition = 0.0
    sigma2 = var_err**2 + obs_err**2 + err2_addition
    # if isinstance(sigma2, float):
        # if sigma2 < 1e-8:
            # sigma2 = 1
    # elif (sigma2 < 1e-8).all():
        # sigma2 = np.ones_like(sigma2)
    
    # interpolate the model grid to the observed coordinates
    if obs.ndim == 1:
        model_interp = interpolate.interpn(grid.grid, model, var)
        model_log_likelihood = (obs-model_interp)**2/sigma2 #+ np.log(sigma2) 
        print('sigma2', np.log(sigma2))
        print('diff/sigma2', (obs-model_interp)**2/sigma2)
    if obs.ndim == 2:
        model_interp = interpolate.interpn(
                grid.grid, model, 
                np.array([var[0].flatten(),var[1].flatten()]).T
                ).reshape(obs.shape)
        model_log_likelihood = ((obs-model_interp)**2/sigma2 + np.log(sigma2)).flatten()
        # model_log_likelihood = np.sum((obs-model_interp)**2/sigma2 + np.log(sigma2))
    if obs.ndim == 3:
        model_interp = interpolate.interpn(
                grid.grid, model, 
                np.array([var[0].flatten(),var[1].flatten(), var[2].flatten()]).T
                ).reshape(obs.shape)
        model_log_likelihood = ((obs-model_interp)**2/sigma2 + np.log(sigma2)).flatten()
    return model_log_likelihood - params_priors

def minimize_cost_old(*args):
    return -1.0*log_probability(*args)

def log_probability_old(theta, models, grid, data):
    """
    args: [theta, models, grid, data]
    """
    # var, data, data_err = args
    model_parameters = models.parameters
    # for key, value in model_parameters.items():
        # print(key, value.value)
    # print('theta', theta)
    dict_theta = dict(zip(model_parameters.keys(), theta))
    models.update_parameters(dict_theta)
    params_priors = models.get_priors(dict_theta)
    if not np.isfinite(params_priors):
        # print('inf of priors')
        return -np.inf 
    model = models.create_model(grid)
    var = data['var']
    try: var_err = data['var_err']
    except: var_err = 0
    obs = data['data']
    try: obs_err = data['data_err']
    except: obs_err = 0
    for m in models.models:
        if isinstance(m, ScaleUncertainty):
            log_f = m.factor.value
            err2_addition = model**2*np.exp(2*log_f)
        else:
            err2_addition = 0
    sigma2 = var_err**2 + obs_err**2 + err2_addition
    
    # interpolate the model grid to the observed coordinates
    if obs.ndim == 1:
        need_interpolation = True
        if grid.grid[0].size == np.array(var).size:
            if ((grid.grid[0] - var) < 1e-4).all():
                need_interpolation = False
        if need_interpolation:
            model_interp = interpolate.interpn(grid.grid, model, var)
        else:
            model_interp = model
    if obs.ndim == 2:
        model_interp = interpolate.interpn(
                grid.grid, model, np.array([var[0].flatten(),var[1].flatten()]).T).reshape(obs.shape)
    if obs.ndim == 3:
        # model_interp = interpolate.interpn(
                # grid.grid, model, 
                # np.array([var[0].flatten(),var[1].flatten(), var[2].flatten()
                                              # ]).T).reshape(obs.shape)
        model_interp = model
    # model_log_likelihood = -0.5*np.sum((data-model)**2)/data_err**2 + np.log(data_err**2)
    model_log_likelihood = -0.5*np.sum((obs-model_interp)**2/sigma2 + np.log(sigma2))
    # print('model_log_likelihood', model_log_likelihood + params_priors)
    return model_log_likelihood + params_priors


###############################################
#### helper functions
###############################################
def check_regular_grid(arr,):
    arr_diff = np.diff(arr)
    if np.all(np.abs(arr_diff - arr_diff[0]) <= np.finfo(arr_diff.dtype).eps):
        return True
    else:
        False

###############################################
#### test functions
###############################################
def test_gaussian1d_fitting(plot=False):
    grid1d = Grid1D(size=100, pixelsize=0.5)
    grid_orig = Grid1D(size=50, pixelsize=1)
    amp0, mean0, sigma0, cont0 = [1, 0, 2, 0]
    gaussian_model1 = Gaussian1D(amplitude=Parameter(amp0, limits=[0,2]), 
                                 mean=Parameter(mean0, limits=[-2,2]), 
                                 sigma=Parameter(sigma0, limits=[0,5]),
                                 cont=0, #Parameter(cont0, limits=[-1,1]), 
                                 name='gaussian1')
    models = Models([gaussian_model1])
    model_data = models.create_model(grid_orig)
    np.random.seed(1994)
    y_err = 0.2*(np.random.random(model_data.shape)-0.5)
    y_data = model_data + y_err#0.1*(np.random.random(model_data.shape)-0.5)
    y_data_err = np.abs(y_err)
    data = {'var':grid_orig.x, 'data':y_data, 'data_err':y_data_err}
    
    start = time.time()
    fitter_minimize = MinimizeFitter(models, grid1d, data)
    fitter_minimize.auto_run()
    end = time.time()
    print(f"MinimizeFitter used {end - start}s")

    start = time.time()
    fitter_mcmc = EmceeFitter(models, grid1d, data)
    max_steps = 10000
    # fitter_mcmc.run(progress=True, steps=steps)
    fitter_mcmc.auto_run(progress=True, max_steps=max_steps)
    end = time.time()
    print(f"EmceeFitter used {end - start}s")

    initial_guess = np.array([amp0, mean0, sigma0])
    best_fit_minimize = fitter_minimize.best_fit
    best_fit_minimize_error = fitter_minimize.best_fit_error
    samples = fitter_mcmc.samples(flat=True)
    best_fit_mcmc = np.percentile(samples, 50, axis=0)
    best_fit_mcmc_low = best_fit_mcmc - np.percentile(samples, 16, axis=0)
    best_fit_mcmc_up = np.percentile(samples, 84, axis=0) - best_fit_mcmc
    best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
    print("Noiseless Truth:", initial_guess)
    print('Best fit minimize:', best_fit_minimize)
    print('Best fit minimize (error):', best_fit_minimize_error)
    print('Best fit mcmc:', best_fit_mcmc)
    print('Best fit mcmc (error):', best_fit_mcmc_err)
    # print('Differences:', np.abs(initial_guess-best_fit))
    if plot:
        fig, ax = plt.subplots(1,2, figsize=(10,4))
        # fitter_minimize.plot(ax=ax[0])
        # fitter_mcmc.plot(ax=ax[1])
        models.update_parameters(dict(zip(models.get_parameters(), best_fit_minimize)))
        fit_data_minimize = models.create_model(grid1d)
        models.update_parameters(dict(zip(models.get_parameters(), best_fit_mcmc)))
        fit_data_mcmc = models.create_model(grid1d)
        for ax_i in ax:
            ax_i.errorbar(grid_orig.x, y_data, yerr=y_data_err, linestyle='none', 
                          marker='o', markersize=8, capsize=4)
        ax[0].text(.05,.9, 'minimize fitter', transform=ax[0].transAxes, fontsize=12)
        ax[0].plot(grid1d.x, fit_data_minimize)
        ax[1].text(.05,.9, 'mcmc fitter', transform=ax[1].transAxes, fontsize=12)
        ax[1].plot(grid1d.x, fit_data_mcmc)
        plt.show()
    assert (np.abs(initial_guess-best_fit_minimize) < 0.2).all()
    assert (np.abs(initial_guess-best_fit_mcmc) < 0.2).all()

def test_gaussian1d_fitting_minimize():
    # test fitting with minimize algorithm
    grid1d = Grid1D(size=100, pixelsize=1)
    grid_orig = Grid1D(size=40, pixelsize=1)
    amp0, mean0, sigma0, cont0 = [1, 0, 2, 0]
    gaussian_model1 = Gaussian1D(amplitude=Parameter(amp0, limits=[0,2]), 
                                 mean=Parameter(mean0, limits=[-2,2]), 
                                 sigma=Parameter(sigma0, limits=[0,5]),
                                 cont=0, #Parameter(cont0, limits=[-1,1]), 
                                 name='gaussian1')
    models = Models([gaussian_model1])
    model_data = models.create_model(grid_orig)
    np.random.seed(1994)
    y_data = model_data + 0.01*(np.random.random(model_data.shape)-0.5)
    y_data_err = 0.01 #np.random.random(y_data.shape)
    data = {'var':grid_orig.x, 'data':y_data, 'data_err':y_data_err}
    
    fitter = MinimizeFitter(models, grid1d, data)
    fitter.run()
    initial_guess = np.array([amp0, mean0, sigma0])
    best_fit = fitter.best_fit
    best_fit_error = fitter.best_fit_error
    print("Noiseless Truth:", initial_guess)
    print('Best fit:', best_fit)
    print('Best fit error', best_fit_error)
    # print(np.abs(initial_guess-best_fit))
    assert (np.abs(initial_guess-best_fit) < np.array([1,0.1,0.2])).all()

def test_double_gaussian_fitting():
    # test fitting with minimize algorithm
    grid1d = Grid1D(size=100, pixelsize=1)
    grid_orig = Grid1D(size=40, pixelsize=1)
    amp1, mean1, sigma1, cont1 = [1, 0, 2, 0]
    amp2, mean2, sigma2, cont2 = [0.5, 1, 2, 0]
    sigma_joint = Parameter(sigma1, limits=[-4,4])
    gaussian_model1 = Gaussian1D(amplitude=Parameter(amp1, limits=[0,2]), 
                                 mean=Parameter(mean1, limits=[-2,2]), 
                                 sigma=sigma_joint,
                                 cont=0, #Parameter(cont0, limits=[-1,1]), 
                                 name='gaussian1')
    gaussian_model2 = Gaussian1D(amplitude=Parameter(amp2, limits=[0,2]), 
                                 mean=Parameter(mean2, limits=[-2,2]), 
                                 sigma=sigma_joint,
                                 cont=0, #Parameter(cont0, limits=[-1,1]), 
                                 name='gaussian2')

    models = Models([gaussian_model1, gaussian_model2])
    model_data = models.create_model(grid_orig)
    np.random.seed(1994)
    y_data = model_data + 0.01*(np.random.random(model_data.shape)-0.5)
    y_data_err = 0.01 #np.random.random(y_data.shape)
    data = {'var':grid_orig.x, 'data':y_data, 'data_err':y_data_err}
    
    fitter = MinimizeFitter(models, grid1d, data)
    fitter.run()
    initial_guess = np.array([amp0, mean0, sigma0])
    best_fit = fitter.best_fit
    best_fit_error = fitter.best_fit_error
    print("Noiseless truths:", initial_guess)
    print('Best fit:', best_fit)
    print('Best fit error', best_fit_error)
    # print(np.abs(initial_guess-best_fit))
    assert (np.abs(initial_guess-best_fit) < np.array([1,0.1,0.2])).all()

def test_gaussian2d_curve_fit(plot=False):
    from scipy.optimize import curve_fit
    def func(grid, amplitude, y_sigma, x_sigma, y0, x0, theta):
        return gaussian_2d(grid, amplitude, y_sigma, x_sigma, y0, x0, theta).ravel()
    grid2d = Grid2D(size=20, pixelsize=1)
    amp0, x0, y0, xsig0, ysig0, theta0 = [1, 0, 2, 2, 3, 0]
    gaussian_2d_model = Gaussian2D(amplitude=Parameter(amp0, limits=[0, 2]),
                                   x0=Parameter(x0, limits=[-5,5]),
                                   y0=Parameter(y0, limits=[-5,5]),
                                   x_sigma=Parameter(xsig0, limits=[0,5]),
                                   y_sigma=Parameter(ysig0, limits=[0,5]),
                                   theta=Parameter(theta0, limits=[-0.5*np.pi, 0.5*np.pi]))
    models = Models(gaussian_2d_model)
    model_data = models.create_model(grid2d)
    np.random.seed(1994)
    obs_data = model_data + 0.1*(np.random.random(model_data.shape)-0.5)
    obs_data_err = np.random.random(obs_data.shape)
    data = {'var':grid2d.meshgrid, 'data':obs_data, 'data_err':obs_data_err}

    truth_values = np.array([amp0, ysig0, xsig0, y0, x0, theta0]) 
    print(models.parameters)
    print(truth_values)

    popt, pcov = curve_fit(func, grid2d.meshgrid, obs_data.ravel())
    print("Best fit", popt)
    print("Best fit error:", np.sqrt(np.diag(pcov)))

    if plot:
        fig, ax = plt.subplots(1,3, figsize=(12,3))
        data_shape = obs_data.shape
        ax[0].imshow(obs_data)
        ax[1].imshow(func(grid2d.meshgrid, *popt).reshape(data_shape))
        ax[2].imshow(obs_data - func(grid2d.meshgrid, *popt).reshape(data_shape))
        plt.show()
 
def test_gaussian2d_fitting(plot=False, debug=False):
    grid2d = Grid2D(size=20, pixelsize=1)
    amp0, x0, y0, xsig0, ysig0, theta0 = [1, 0, 2, 2, 3, 0]
    gaussian_2d_model = Gaussian2D(amplitude=Parameter(amp0, limits=[0, 2]),
                                   x0=Parameter(x0, limits=[-5,5]),
                                   y0=Parameter(y0, limits=[-5,5]),
                                   x_sigma=Parameter(xsig0, limits=[0,5]),
                                   y_sigma=Parameter(ysig0, limits=[0,5]),
                                   theta=Parameter(theta0, limits=[-0.5*np.pi, 0.5*np.pi]))
    models = Models(gaussian_2d_model)
    model_data = models.create_model(grid2d)
    np.random.seed(1994)
    obs_data = model_data + 0.1*(np.random.random(model_data.shape)-0.5)
    obs_data_err = np.random.random(obs_data.shape)
    data = {'var':grid2d.meshgrid, 'data':obs_data, 'data_err':obs_data_err}

    truths = list(models.parameters.values())
    initial_guess = np.array(truths) * (1 + 0.1*np.random.randn(len(truths)))
    
    models.update_parameters(initial_guess)
    print("Noiseless truths:", truths)
    print("Initial guess:", initial_guess)
    if 0:
        fitter = MinimizeFitter(models, grid2d, data)
        fitter.auto_run(debug=debug)
        # fitter.auto_run(debug=debug)
        print("Best fit:", fitter.best_fit)
        print('Best fit error', fitter.best_fit_error)
        best_fit = fitter.best_fit

        if plot:
            fig, ax = plt.subplots(1,3, figsize=(12,3))
            ax[0].imshow(obs_data)
            models.update_parameters(dict(zip(models.parameters.keys(), best_fit)))
            best_fit_model = models.create_model(grid2d)
            ax[1].imshow(best_fit_model)
            ax[2].imshow(obs_data - best_fit_model)
            plt.show()
        
    if 1:
        fitter = EmceeFitter(models, grid2d, data)
        # fitter.run(progress=True, steps=4000)
        fitter.auto_run(progress=True, max_steps=10000)
        if debug:
            fitter.plot()
            fitter.corner_plot(truths=truths)

        best_fit = np.percentile(fitter.samples(discard=None, flat=True), 50, axis=0)
        print('Best fit:', best_fit)

        ## another more complicated also show the images
        fig = plt.figure(figsize=(10, 3))
        ax_mcmc = [plt.subplot2grid((fitter.n_params,5), (i,1), colspan=4) for i in range(fitter.n_params)]
        fitter.plot(ax=ax_mcmc)
        rowspan = fitter.n_params//3
        ax1 = plt.subplot2grid((fitter.n_params,5), (0,0), rowspan=rowspan)
        ax2 = plt.subplot2grid((fitter.n_params,5), (rowspan,0), rowspan=rowspan)
        ax3 = plt.subplot2grid((fitter.n_params,5), (2*rowspan,0), rowspan=rowspan)
        ax1.imshow(obs_data, origin='lower')
        model_parameters = models.get_parameters()
        best_fit_dict = dict(zip(model_parameters.keys(), best_fit))
        models.update_parameters(best_fit_dict)
        best_fit_model = models.create_model(grid2d)
        ax2.imshow(best_fit_model, origin='lower')
        ax3.imshow(obs_data - best_fit_model, origin='lower')
        plt.show()
    assert (np.abs(truths-best_fit) < 0.5).all()

def test_sersic2d_fitting(plot=False, debug=False):
    grid2d = Grid2D(size=20, pixelsize=1)
    # the initial guess
    amplitude0, reff0, n0, x0, y0, ellip0, theta0 = [
            0.4, 10, 1, 0.2, -0.2, 0.4, np.pi/2]
    sersic_2d_model = Sersic2D(
            amplitude = Parameter(amplitude0, limits=[0,20]),
            reff = Parameter(reff0, limits=[0.1, 30]),
            n = Parameter(n0, limits=[0.1, 8]),
            x0 = Parameter(x0, limits=[-10, 10]), 
            y0 = Parameter(y0, limits=[-10, 10]),
            ellip = Parameter(ellip0, limits=[0, 1]),
            theta = Parameter(theta0, limits=[-0.5*np.pi, 0.5*np.pi])
            )
 
    models = Models(sersic_2d_model)
    model_data = models.create_model(grid2d)
    np.random.seed(1994)
    obs_data = model_data + 0.01*(np.random.random(model_data.shape)-0.5)
    obs_data_err = np.random.random(obs_data.shape)
    data = {'var':grid2d.meshgrid, 'data':obs_data, 'data_err':obs_data_err}

    truths = np.array(list(models.parameters.values()))
    initial_guess = np.array(truths) * (1 + 0.1*np.random.randn(len(truths)))
    print("Noiseless truths:", truths)
    print("Initial guess:", initial_guess)
    models.update_parameters(initial_guess)
    if 0:
        fitter = MinimizeFitter(models, grid2d, data)
        fitter.auto_run()
        print("Best fit", fitter.best_fit)
        print("Best fit error", fitter.best_fit_error)
        best_fit = fitter.best_fit
    if 1:
        fitter = EmceeFitter(models, grid2d, data)
        # fitter.run(progress=True, steps=4000)
        # fitter.auto_run(progress=True, max_steps=10000)
        fitter.multi_run(progress=True, steps=10000, ncores=4)
        if debug:
            fitter.plot()
            fitter.corner_plot(truths=truths)

        best_fit = np.percentile(fitter.samples(discard=None, flat=True), 50, axis=0)
        print('Best fit:', best_fit)
        ## another more complicated also show the images
        fig = plt.figure(figsize=(12, 4))
        ax_mcmc = [plt.subplot2grid((fitter.n_params,5), (i,1), colspan=4) for i in range(fitter.n_params)]
        fitter.plot(ax=ax_mcmc)
        rowspan = fitter.n_params//3
        ax1 = plt.subplot2grid((fitter.n_params,5), (0,0), rowspan=rowspan)
        ax2 = plt.subplot2grid((fitter.n_params,5), (rowspan,0), rowspan=rowspan)
        ax3 = plt.subplot2grid((fitter.n_params,5), (2*rowspan,0), rowspan=rowspan)
        ax1.imshow(obs_data, origin='lower')
        model_parameters = models.get_parameters()
        best_fit_dict = dict(zip(model_parameters.keys(), best_fit))
        models.update_parameters(best_fit_dict)
        best_fit_model = models.create_model(grid2d)
        ax2.imshow(best_fit_model, origin='lower')
        ax3.imshow(obs_data - best_fit_model, origin='lower')
        plt.show()
    assert (np.abs(initial_guess-best_fit)/initial_guess < 0.2).all()

def test_blackbody(plot=True, debug=False):
    grid1d = Grid1D()
    freq = np.array([100, 200, 400, 600, 800, 1000, 2000, 4000, 5000]) # GHz
    grid1d.x = freq 
    temperature = 40
    dustmass = 1e10
    z = 2
    mbb_model = MBlackbody(temperature=Parameter(temperature, limits=[10,100]),
                           z=2,
                           beta=2.0,
                           dustmass=Parameter(dustmass, limits=[1e9, 1e11]), #solarmass
                           area=1, #kpc2
                           name='test_mbb')
    models = Models([mbb_model])
    model_data = models.create_model(grid1d)
    np.random.seed(1994)
    y_err = np.array([0.01, -0.03, -0.2, 0.5, 0.8, -1., 3, -4.5, 4.5]) # mJy
    # y_err = 1e-2*np.zeros(model_data.size) #0.1*model_data#*np.random.randn(model_data.size)
    y_data = model_data + y_err
    y_data_err = np.abs(y_err)

    # define the data for fitters
    # initial_guess = np.array([temperature, dustmass])
    initial_guess = np.array([50, 9.2])
    data = {'var':grid1d.x, 'data':y_data, 'data_err':y_data_err}

    # initialize fitters
    fitter_minimize = MinimizeFitter(models, grid1d, data)
    fitter_minimize.auto_run()
    best_fit_minimize = fitter_minimize.best_fit
    best_fit_minimize_error = fitter_minimize.best_fit_error

    initial_guess = np.array([50, 9.2])
    models.update_parameters(dict(zip(['temperature','dustmass'], initial_guess)))
    fitter_mcmc = EmceeFitter(models, grid1d, data)
    steps = 5000
    fitter_mcmc.auto_run(progress=True, max_steps=steps)
    if debug:
        fitter_mcmc.plot()
        fitter_mcmc.corner_plot()

    samples = fitter_mcmc.samples(discard=int(steps*0.2), thin=1, flat=True)
    best_fit_mcmc = np.percentile(samples, 50, axis=0)
    best_fit_mcmc_low = np.percentile(samples, 16, axis=0)
    best_fit_mcmc_up = np.percentile(samples, 84, axis=0)
    best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
    print("Noiseless truths:", [temperature, dustmass])
    print('Best fit minimize:', best_fit_minimize)
    print('Best fit mcmc:', best_fit_mcmc)
 
    if plot:
        models.update_parameters(dict(zip(models.get_parameters(), best_fit_minimize)))
        fit_data_minimize = models.create_model(grid1d)
        models.update_parameters(dict(zip(models.get_parameters(), best_fit_mcmc)))
        fit_data_mcmc = models.create_model(grid1d)
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.errorbar(freq, y_data, yerr=y_data_err, linestyle='none', marker='o', label='data')
        ax.plot(freq, model_data, label='model')
        ax.plot(freq, fit_data_minimize, label='minimize fit')
        ax.plot(freq, fit_data_mcmc, label='mcmc fit')
        ax.invert_xaxis()
        ax.legend()
        plt.show()
 
def test_blackbody_log(plot=True, debug=False):
    grid1d = Grid1D()
    freq = np.array([100, 200, 400, 600, 800, 1000, 2000, 4000,5000]) # GHz
    grid1d.x = freq 
    temperature = 40
    dustmass = 10
    z = 2
    mbb_model = MBlackbody_log(temperature=Parameter(temperature, limits=[10,100]),
                           z=2,
                           beta=2.0,
                           dustmass=Parameter(dustmass, limits=[9, 11]), #solarmass
                           area=1, #kpc2
                           name='test_mbb')
    models = Models([mbb_model])
    model_data = models.create_model(grid1d)
    np.random.seed(1994)
    y_err = np.array([0.01, -0.03, -0.2, 0.5, 0.8, -1., 3, -4.5, 4.5]) # mJy
    # y_err = 1e-2*np.zeros(model_data.size) #0.1*model_data#*np.random.randn(model_data.size)
    y_data = model_data + y_err
    y_data_err = np.abs(y_err)

    # define the data for fitters
    initial_guess = np.array([temperature, dustmass])
    initial_guess = np.array([50, 9.2])
    data = {'var':grid1d.x, 'data':y_data, 'data_err':y_data_err}

    # initialize fitters
    fitter_minimize = MinimizeFitter(models, grid1d, data)
    fitter_minimize.run()
    best_fit_minimize = fitter_minimize.best_fit
    best_fit_minimize_error = fitter_minimize.best_fit_error

    initial_guess = np.array([50, 9.2])
    models.update_parameters(dict(zip(['temperature','dustmass'], initial_guess)))
    fitter_mcmc = EmceeFitter(models, grid1d, data)
    steps = 5000
    fitter_mcmc.auto_run(progress=True, max_steps=steps)
    if debug:
        fitter_mcmc.plot()
        fitter_mcmc.corner_plot()

    samples = fitter_mcmc.samples(discard=int(steps*0.2), thin=1, flat=True)
    best_fit_mcmc = np.percentile(samples, 50, axis=0)
    best_fit_mcmc_low = np.percentile(samples, 16, axis=0)
    best_fit_mcmc_up = np.percentile(samples, 84, axis=0)
    best_fit_mcmc_err = list(zip(best_fit_mcmc_low, best_fit_mcmc_up))
    print("Noiseless truths:", [temperature, dustmass])
    print('Best fit minimize:', best_fit_minimize)
    print('Best fit mcmc:', best_fit_mcmc)
 
    if plot:
        models.update_parameters(dict(zip(models.get_parameters(), best_fit_minimize)))
        fit_data_minimize = models.create_model(grid1d)
        models.update_parameters(dict(zip(models.get_parameters(), best_fit_mcmc)))
        fit_data_mcmc = models.create_model(grid1d)
        fig, ax = plt.subplots(1,1, figsize=(8,6))
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.errorbar(freq, y_data, yerr=y_data_err, linestyle='none', marker='o', label='data')
        ax.plot(freq, model_data, label='model')
        ax.plot(freq, fit_data_minimize, label='minimize fit')
        ax.plot(freq, fit_data_mcmc, label='mcmc fit')
        ax.invert_xaxis()
        ax.legend()
        plt.show()
     
