# a collection of function to help search and identify molecular and ion lines

# by Jianhang Chen
# cjhastro@gmail.com 
# History: 
#   2021.07.18, first release, v0.1
#   2024.01.22, add class Spectrum, separated from cube_utils.py, v0.2

__version__ = '0.2.1'

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as const
from astropy.modeling import fitting, models
import warnings

class Database(object):
    """A small database of frequently used spectral lines
    """
    def __init__(self):
        self.CO_family = OrderedDict([
            ('12C16O_1-0', 115.27120256*u.GHz), ('12C16O_2-1', 230.53800100*u.GHz),
            ('12C16O_3-2', 345.79599131*u.GHz), ('12C16O_4-3', 461.04076975*u.GHz),
            ('12C16O_5-4', 576.26793295*u.GHz), ('12C16O_6-5', 691.4730780*u.GHz),
            ('12C16O_7-6', 806.6518028*u.GHz),  ('12C16O_8-7', 921.7997056*u.GHz),
            ('12C16O_9-8', 1036.9123861*u.GHz), ('12C16O_10-9', 1151.9854444*u.GHz)
            ])
        self.CO_13C = OrderedDict([('13C16O_1-0', 110.20135487*u.GHz), ('13C16O_2-1', 220.39868527*u.GHz),
                              ('13C16O_3-2', 330.58796682*u.GHz), ('13C16O_4-3', 440.76517539*u.GHz),
                              ('13C16O_5-4', 550.92628510*u.GHz), ('13C16O_6-5', 661.06727660*u.GHz),
                              ('13C16O_7-6', 771.18412500*u.GHz), ('13C16O_8-7', 881.27280800*u.GHz)])
                              # ('13C18O_1-0', 104.711393*u.GHz), ('13C17O_1-0', 107.288945*u.GHz),
                              # ('13C17O_2-1', 214.574077*u.GHz)])
        self.CO_17O = OrderedDict([('12C17O_1-0_multi', 112.358988*u.GHz), ('12C17O_2-1_multi', 224.714187*u.GHz),])

        self.CO_18O = OrderedDict([('12C18O_1-0', 109.78217340*u.GHz), ('12C18O_2-1', 219.56035410*u.GHz),
                          ('12C18O_3-2', 329.33055250*u.GHz), ('12C18O_4-3', 439.08876580*u.GHz),
                          ('12C18O_5-4', 548.83100550*u.GHz), ('12C18O_6-5', 658.55327820*u.GHz),
                          ('12C18O_7-6', 768.25159330*u.GHz), ('12C18O_8-7', 877.92195530*u.GHz)])
        self.C_ion = OrderedDict([('CI_1-0', 492.16065100*u.GHz), ('CI_2-1', 809.34197*u.GHz), 
                         ('CII', 1900.5369*u.GHz)])
        self.H2O = OrderedDict([
                       ('H2O_414-321', 380.19735980*u.GHz), ('H2O_423-330', 448.00107750*u.GHz),
                       ('H2O_110-101', 556.93598770*u.GHz), ('H2O_532-441', 620.70095490*u.GHz),
                       ('H2O_211-202', 752.03314300*u.GHz), ('H2O_422-331', 916.17158000*u.GHz), 
                       ('H2O_202-111', 987.92675900*u.GHz), ('H2O_312-303', 1097.36479000*u.GHz),
                       ('H2O_111-000', 1113.34300700*u.GHz), ('H2O_321-312', 1162.91160200*u.GHz),
                       ('H2O_422-413', 1207.63873000*u.GHz), ('H2O_220-211', 1228.78871900*u.GHz),
                       ('H2O_523-514', 1410.61806900*u.GHz), ('H2O_413-404', 1602.21936900*u.GHz),
                       ('H2O_221-212', 1661.00763700*u.GHz), ('*H2O_212-101', 1669.90477500*u.GHz),
                       ('*H2O_303-212', 1716.76963300*u.GHz), ('H2O_532-523', 1867.74859400*u.GHz),
                       ('H2O_322-313', 1919.35953100*u.GHz), ('H2O_431-422', 2040.47681000*u.GHz),
                       ('H2O_413_322', 2074.43230500*u.GHz), ('*H2O_313_202', 2164.13198000*u.GHz),
                       ('H2O_330-321', 2196.34575600*u.GHz), ('H2O_514-505', 2221.75050000*u.GHz),
                       ('*H2O_423-414', 2264.14965000*u.GHz), ('H2O_725-716', 2344.25033500*u.GHz),
                       ('H2O_331-322', 2365.89965900*u.GHz), ('*H2O_404-313', 2391.57262800*u.GHz),
                       ])
        self.HCN = OrderedDict([('HCN_1-0', 88.63160230*u.GHz), ('HCN_2-1', 177.26111150*u.GHz), 
                       ('HCN_3-2', 265.8864343*u.GHz), ('HCN_4-3', 354.50547790*u.GHz),
                       ('HCN_5-4', 443.1161493*u.GHz), ('HCN_6-5', 531.71634790*u.GHz)])
        self.HNC = OrderedDict([('HNC_1-0', 90.66356800*u.GHz), ('HNC_2-1', 181.32475800*u.GHz), 
                       ('HNC_3-2', 271.98114200*u.GHz), ('HNC_4-3', 362.63030300*u.GHz),
                       ('HNC_5-4', 453.26992200*u.GHz), ('HNC_6-5', 543.89755400*u.GHz)])
        self.HCO_plus = OrderedDict([('HCO_plus_1-0', 89.1885247*u.GHz), ('HCO_plus_2-1', 178.3750563*u.GHz), 
                       ('HCO_plus_3-2', 267.5576259*u.GHz), ('HCO_plus_4-3', 356.734223*u.GHz),
                       ('HCO_plus_5-4', 445.9028721*u.GHz), ('HCO_plus_6-5', 535.061581*u.GHz)])
        self.Other = OrderedDict([('CN_1-0_multi', 113.490982*u.GHz), ('CS_2-1', 97.9809533*u.GHz),
                         ('CS_3-2', 146.9690287*u.GHz), ('CS_4-3', 195.9542109*u.GHz),
                         ('CS_5-4', 244.9355565*u.GHz), ('CS_6-5', 293.9120865*u.GHz),
                         ('CS_7-6', 342.8828503*u.GHz), ('CS_8-7', 391.8468898*u.GHz),
                         ('CS_9-8', 440.8032320*u.GHz), ('CS_10-9', 489.7509210*u.GHz)])
        self.Special = OrderedDict([('CH+', 835.08*u.GHz), ('OH+(1-0)', 1033.1*u.GHz)])

        # Optical lines, all wavelength in vaccum
        # for Hydrogen,Helium and Lithium lines: https://zenodo.org/records/1232309
        self.Lyman_series = OrderedDict([('Ly-alpha', 1215.7*u.AA), ('Ly-beta', 1025.7*u.AA), 
                                          ('Ly-gamma', 972.54*u.AA), ('Lyman-limit', 911.75*u.AA)])
        self.Balmer_series = OrderedDict([('H-alpha', 6564.603*u.AA), ('H-beta', 4862.708*u.AA), 
                                          ('H-gamma', 4341.692*u.AA), ('H-delta', 4102.891*u.AA), 
                                          ('Balmer-limit', 3646.04*u.AA)])
        self.Paschen_series = OrderedDict([('Pa-alpha', 1875.73*u.nm), ('Pa-beta', 1281.61*u.nm),
                                           ('Pa-gamma', 1094.15*u.nm), ('Pa-delta', 1005.32*u.nm),
                                           ('Paschen-limit', 820.1*u.nm)])
        self.Brackett_series = OrderedDict([('Br-alpha', 4052.28*u.nm), 
                                            ('Br-beta', 2625.92*u.nm), 
                                            ('Brackett-limit', 1458.0*u.nm)])
        self.Helium = OrderedDict([('HeI3889', 3889*u.AA), 
                                   ('HeII', 303.783*u.AA),
                                   ('HeII1640', 1640.41*u.AA), 
                                   ('HeII4686', 4685.7*u.AA),
                                   ('HeII5876', 5875.8*u.AA),])
        # optical line from SDSS: https://classic.sdss.org/dr6/algorithms/linestable.php
        self.Optical_lines = OrderedDict([
                                          ('Mg II]2795', 2795.528*u.AA),
                                          ('Mg II]2802', 2802.705*u.AA),
                                          ('[OII]3727', 3727.092*u.AA), 
                                          ('[OII]3730', 3729.875*u.AA), 
                                          ('[OIII4364]',4364.436*u.AA),
                                          ('[OIII4960]', 4960.295*u.AA), 
                                          ('[OIII5008]', 5008.239*u.AA),
                                          ('[OI]6302', 6302.046*u.AA), 
                                          ('[OI]6365', 6365.536*u.AA), 
                                          ('[NII]6549', 6549.86*u.AA), 
                                          ('[NII]6585', 6585.27*u.AA), 
                                          ('[SII]6718', 6718.29*u.AA), 
                                          ('[SII]6732', 6732.67*u.AA)])

        self.Optical_obsorption = OrderedDict([('-Na_D1', 5890*u.AA), ('-Na_D2', 5896*u.AA),
                                               ('-CaII8500', 8500.36*u.AA), ('-CaII8544', 8544.44*u.AA), 
                                               ('-CaII8664', 8664.52*u.AA)])
 
class Spectrum(object):
    """the data structure to handle spectrum
    """
    def __init__(self, specchan=None, specdata=None, reffreq=None, refwave=None, 
                 z=None):
        """ initialize the spectrum
        """
        self.default_frequency_unit = 'GHz'
        self.default_wavelength_unit = 'nm'
        self.specchan = specchan
        self.specdata = specdata
        self._reffreq = reffreq
        self._refwave = refwave
        self.z = z
    @property
    def reffreq(self):
        if self._reffreq is not None:
            return self._reffreq
        elif self._refwave is not None:
            return (const.c/self.refwave).to(self.default_frequency_unit)
    @reffreq.setter
    def reffreq(self, reffreq):
        self._reffreq = reffreq
    @property
    def refwave(self):
        if self._refwave is not None:
            return self._refwave
        elif self._reffreq is not None:
            return (const.c/self.reffreq).to(self.default_wavelength_unit)
    @refwave.setter
    def refwave(self, refwave):
        self._refwave = refwave

    @property
    def channels(self):
        return list(range(len(self.specchan)))
    @property
    def unit(self):
        if isinstance(self.specchan, u.Quantity):
            return self.specchan.unit
        else: 
            return None
    @unit.setter
    def unit(self, unit):
        self.specchan = self.specchan*unit

    def wavelength(self, units=u.um, reference=None):
        if reference is None:
            reference = self.reffreq
        return convert_spec(self.specchan, units, reference=reference)

    def frequency(self, units=u.GHz, reference=None):
        if reference is None:
            reference = self.reffreq
        return convert_spec(self.specchan, units, reference=reference)

    def to_restframe(self, z=None):
        if z is None:
            z = self.z
        if is_equivalent(self.unit, u.Hz):
            return convert_spec(self.specchan, 'GHz')*(z+1)
        elif is_equivalent(self.unit, u.m):
            return convert_spec(self.specchan, 'um')/(z+1)
        else: # should left velocity specchan, no need to convert
            return self.specchan

    def velocity(self, reference=None):
        if reference is None:
            reference = self.reffreq
        if reference is not None:
            return convert_spec(self.specchan, 'km/s', reference)
        else:
            warnings.warn("No valid reference found for the conversion!")
            return None

    def convert_specchan(self, unit_out, refwave=None, reffreq=None):
        """convert the units of the specdata 
        """
        if refwave is not None:
            self.refwave = refwave
            self.specchan = convert_spec(self.specchan, unit_out, reference=refwave,)
        if reffreq is not None:
            self.reffreq = reffreq
            self.specchan = convert_spec(self.specchan, unit_out, reference=reffreq)

    def to_channel(self, values):
        #convert the spectral axis
        if values.unit.is_equivalent('km/s'): # velocity to channel
            return array_mapping(values, self.velocity(), self.channels)
        elif values.unit.is_equivalent('um'): # velocity to channel
            return array_mapping(values, self.wavelength(values.unit), self.channels)
        elif values.unit.is_equivalent('GHz'): # velocity to channel
            return array_mapping(values, self.frequency(values.unit), self.channels)

    def integral(self):
        """integrate the spectrum
        """
        return integral_spectrum(self.specvel, self.specdata)

    def plot(self, ax=None, **kwargs):
        plot_spectra(self.specchan, self.specdata, ax=ax, **kwargs)

    def fit_gaussian(self, data_boundary=None,
                     channel_boundary=None, **kwargs):
        """fitting the single gaussian to the spectrum
        """
        specchan = self.specchan
        velocity = self.velocity()
        specdata = self.specdata
        if isinstance(specdata, u.Quantity):
            specdata = specdata.data
        if velocity is not None:
            print("fitting velocity space")
            return fit_gaussian1D(velocity.data, specdata, **kwargs)
        else:
            if isinstance(specchan, u.Quantity):
                specchan = specchan.data
            return fit_gaussian1D(specchan, specdata, **kwargs)

class Fitspectrum():
    """the data structure to store the fitted spectrum
    """
    __slots__ = ['specchan','specdata','fitdata','bestfit','params','chi2']
    def __init__(self):
        """
        Args:
            specchan: the spectrum axis
            specdata: the data axis
            params: the parameter names
            bestfit: the best-fit value parameters
            chi2: the fitted chi2
        """
        self.specchan = None
        self.specdata = None
        self.fitdata = None
        self.params = None
        self.bestfit = None
        self.chi2 = None
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.step(self.specchan, self.specdata, where='mid', color='k')
        ax.step(self.specchan, self.fitdata, where='mid', color='red', alpha=0.8)

########################################
###### stand alone functions ###########
########################################

def convert_spec(spec_in, unit_out, reference=None, mode='radio'):
    """convert the different spectral axis

    Args:
        spec_in (astropy.Quantity): any valid spectral data with units
        unit_out (str or astropy.Unit): valid units for output spectral axis
        reference: the reference frequency or wavelength, with units

    Return:
    """
    unit_in = spec_in.unit
    if not isinstance(unit_out, u.Unit):
        unit_out = u.Unit(unit_out)
    if spec_in.unit.is_equivalent(unit_out):
        return spec_in.to(unit_out)
    if reference is not None:
        if not isinstance(reference, u.Quantity):
            reference = reference * unit_in
        if reference.unit.is_equivalent('m'):
            refwave = reference
            reffreq = (const.c/refwave).to(u.GHz)
        elif reference.unit.is_equivalent('Hz'):
            reffreq = reference
            refwave = (const.c/reffreq).to(u.um)
    else:
        reffreq, refwave = None, None
    if spec_in.unit.is_equivalent('m'): # this is velocity
        if unit_out.is_equivalent('Hz'):
            return (const.c/spec_in).to(unit_out)
        elif unit_out.is_equivalent('km/s'):
            if mode=='radio':
                return ((spec_in-refwave)/spec_in*const.c).to(unit_out)
            elif mode=='optical':
                return ((spec_in-refwave)/refwave*const.c).to(unit_out)
    if spec_in.unit.is_equivalent('Hz'):
        if unit_out.is_equivalent('m'):
            return (const.c/spec_in).to(unit_out)
        elif unit_out.is_equivalent('km/s'):
            if mode=='radio':
                return ((reffreq-spec_in)/reffreq*const.c).to(unit_out)
            elif mode=='optical':
                return ((reffreq/spec_in-1)*const.c).to(unit_out)
    if spec_in.unit.is_equivalent('km/s'): # this is velocity
        if unit_out.is_equivalent('m'):
            if mode=='radio':
                return (refwave/(1-spec_in/const.c)).to(unit_out)
            elif mode=='optical':
                return (spec_in/const.c*refwave+refwave).to(unit_out)
        elif unit_out.is_equivalent('Hz'):
            if mode=='radio':
                return (reffreq*(1-spec_in/const.c)).to(unit_out)
            elif mode=='optical':
                return (reffreq/(spec_in/const.c+1)).to(unit_out)

def calculate_diff(v):
    """approximate the differential v but keep the same shape
    """
    nchan = len(v)
    dv_cropped = np.diff(v)
    if isinstance(v, u.quantity.Quantity):
        v_expanded = np.zeros(nchan+1) * v.unit
    else:
        v_expanded = np.zeros(nchan+1)
    v_expanded[:-2] = v[:-1]-0.5*dv_cropped
    v_expanded[2:] = v_expanded[2:] + v[1:]+0.5*dv_cropped
    v_expanded[2:-2] = 0.5*v_expanded[2:-2]
    return np.abs(np.diff(v_expanded))

def integrate_spectrum(x, y):
    """calculate the area covered by the spectrum

    Args:
        x (ndarray): the x axis of the curve, with units
        y (ndarray): the y axis of the curve, with units
    """
    return np.sum(calculate_diff(x)*y)

def plot_spectra(specchan, specdata, ax=None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    ax.step(specchan, specdata, where='mid', **kwargs)

def array_mapping(vals1, array1, array2):
    """mapping the values from array2 at the position of vals1 relative to array1
    """
    return (vals1 - array1[0])/(array1[-1]- array1[0]) * (array2[-1]-array2[0])

def print_lines(z=0.0, family=None, unit=None, limits=None, reference=None, 
                mode='radio'):
    '''
    Parameters
    ----------
    format : str
        output format, either in frequency (freq) or wavelength (wave)
    '''
    db = Database()
    if limits is not None:
        limits = np.array(limits)
    if family is None:
        print("Specify the spectral line family:")
        print("all, CO, CO_isotop, H2O, dense_gas, optical")
    if family == 'simple':
        family_select = [db.CO_family, dbC_ion]
        if unit is None:
            unit = u.Unit('GHz')
    if family == 'CO':
        family_select = [db.CO_family]
        if unit is None:
            unit = u.Unit('GHz')
    elif family == 'water':
        family_select = [db.H2O]
        if unit is None:
            unit = u.Unit('GHz')
    elif family == 'all' or family=='full':
        family_select = [db.CO_family, db.CO_13C, db.CO_18O, db.CO_17O, 
                         db.C_ion, db.H2O, db.HCN, db.HNC, db.HCO_plus, db.Special, db.Other]
        if unit is None:
            unit = u.Unit('GHz')
    else:
        family_select = []

    if family is not None:
        if 'optical' in family:
            family_select.append(db.Lyman_series)
            family_select.append(db.Balmer_series)
            family_select.append(db.Paschen_series)
            family_select.append(db.Brackett_series)
            family_select.append(db.Helium)
            family_select.append(db.Optical_lines)
            family_select.append(db.Optical_obsorption)
        if unit is None:
            unit = u.Unit('nm')

    for fs in family_select:
        for name, value in fs.items():
            value_converted = convert_spec(value, unit, reference=reference, mode=mode)
            if unit.is_equivalent(u.m):
                value_in_unit = (value_converted*(1+z)).value
            if unit.is_equivalent(u.Hz):
                value_in_unit = (value_converted/(1+z)).value
            if limits is not None:
                if (value_in_unit > limits[-1]) | (value_in_unit < limits[0]):
                    continue
            print("{}: {:.2f}{}".format(name, value_in_unit, unit.to_string()))
    return

def stacking(dlist, plot=True, norm=True, norm_func=np.mean):
    """
    """
    first_item = dlist[0]
    if isinstance(first_item, str):
        data = np.loadtxt(first_item)
    elif isinstance(dlist, (list, np.ndarray)):
        data = dlist
    dtype = type(first_item)
    stacked_flux = np.zeros(len(data))

    if dtype == str:
        data_list = []
        for df in dlist:
            if type(df) is not dtype:
                raise ValueError("Data type not consistant!")
            data_list.append(np.loadtxt(df))
    else:
        data_list = dlist
    for data in data_list:
        if norm == True:
            stacked_flux += data[:,1] / norm_func(data[:,1]) / np.std(data[:,1]/norm_func(data[:,1]))
        else:
            stacked_flux += data[:,1]

    if plot:
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        for i,data in enumerate(data_list):
            if norm:
                ax.step(data[:,0], data[:,1]/norm_func(data[:,1]), 
                        label="data{}".format(i), where='mid')
            else:
                ax.step(data[:,0], data[:,1], label='data{}'.format(i), where='mid')
        #plot the stacked data
        if norm:
            ax.step(data[:,0], stacked_flux/norm_func(stacked_flux), color='black',
                    label='Stacked', where='mid', lw=4, alpha=0.8)
        else:
            ax.step(data[:,0], stacked_flux, label='Stacked', where='mid', lw=4, color='black', alpha=0.8)
        plt.legend()
        plt.show()

def wave_convert(lam):
    """
    Ciddor 1996, Applied Optics 35, 1566 http://doi.org/10.1364/AO.35.001566

    Args:
        lam: wavelength in Angstroms or astropy.Quantity
    
    Return:
    conversion factor
    """
    if isinstance(lam, u.Quantity):
        lam = lam.to(u.AA).value
    lam = np.array(lam)
    sigma2 = (1e4/lam)**2
    fact = 1 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)
    return fact

def vac_to_air(lam_vac):
    """
    Convert vacuum to air wavelengths

    Args:
        lam_vac: wavelength in Angstroms
    Return:
        lam_air: wavelength in Angstroms
    """
    if isinstance(lam_vac, u.Quantity):
        lam_vac = lam_vac.to(u.AA).value
    return lam_vac/wave_convert(lam_vac)*u.AA

def air_to_vac(lam_air):
    """
    Convert air to vacuum wavelengths

    Args:
        lam_air: wavelength in Angstroms
    Return:
        lam_vac: wavelength in Angstroms
    """
    if isinstance(lam_air, u.Quantity):
        lam_air = lam_air.to(u.AA).value
    return lam_air*wave_convert(lam_air)*u.AA

def fit_gaussian1D(specchan, specdata, guess=None, bounds=None, 
                   plot=False, ax=None, debug=False):
    """fit a single gaussian, with the dimensionless data

    Args:
        specchan: channel data, can be wavelength, frequency, velocity
        specdata: data
        guess: the initial guess for [amplitude, mean, sigma]
        bounds: the boundary for [(amp_min, amp_max), (mean_min, mean_max), 
                                  (sigma_min, sigma_max)]
    """
    specchan = np.array(specchan)
    specdata = np.array(specdata)
    if guess is not None:
        amp0, mean0, sigma0 = guess
    else:
        amp0 = mean0 = sigma0 = None
    spec_selection = specdata > np.percentile(specdata, 85)
    if amp0 is None:
        amp0 = 1.5*np.mean(specdata[spec_selection])
    if mean0 is None:
        mean0 = np.median(specchan[spec_selection])
    if sigma0 is None:
        sigma0 = np.abs(np.median(np.diff(specchan)))
    if bounds is None:
        fit_p = fitting.LevMarLSQFitter()
        fit_bounds = {}
    else:
        fit_p = fitting.TRFLSQFitter()
        fit_bounds = {'amplitude':bounds[0], 'mean':bounds[1], 'stddev':bounds[2]}
    if debug:
        print(f"Initial guess: {amp0, mean0, sigma0}")
        print(f"Bounds: {bounds}")
    p_init = models.Gaussian1D(amplitude=amp0, mean=mean0, 
                               stddev=sigma0,
                               bounds=fit_bounds)
    p = fit_p(p_init, specchan, specdata)
    specfit = p(specchan)
    fitobj = Fitspectrum()
    fitobj.specchan = specchan
    fitobj.specdata = specdata
    fitobj.params = p.param_names
    fitobj.fitdata = specfit
    fitobj.bestfit = p.param_sets
    fitobj.chi2 = None #TODO #np.sum((bestfit - self.specdata)**2/std**2)
    if plot:
        fitobj.plot(ax=ax)
    return fitobj


