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
        self.Other = OrderedDict([('CN_1-0_multi', 113.490982*u.GHz), ('CS_2-1', 97.9809533*u.GHz),
                         ('CS_3-2', 146.9690287*u.GHz), ('CS_4-3', 195.9542109*u.GHz),
                         ('CS_5-4', 244.9355565*u.GHz), ('CS_6-5', 293.9120865*u.GHz),
                         ('CS_7-6', 342.8828503*u.GHz), ('CS_8-7', 391.8468898*u.GHz),
                         ('CS_9-8', 440.8032320*u.GHz), ('CS_10-9', 489.7509210*u.GHz)])
        self.Special = OrderedDict([('CH+', 835.08*u.GHz), ('OH+(1-0)', 1033.1*u.GHz)])

        # Optical lines, all wavelength in air, the CRC standard
        self.Balmer_series = OrderedDict([('H-alpha', 6564.61*u.AA), ('H-beta', 4862.68*u.AA), 
                                          ('H-gamma', 4341.69*u.AA), ('H-delta', 4102.89*u.AA), 
                                          ('Balmer-limit', 3646.04*u.AA)])
        # self.Paschen_series = OrderedDict([('Pa-alpha', 1874.5*u.nm), ('Pa-beta', 1281.4*u.nm),
                                           # ('Pa-gamma', 1093.5*u.nm), ('Paschen-limit', 820.1*u.nm)])
        # # self.Brackett_series = OrderedDict([('Br-alpha', 4052.5*u.nm), 
                                            # ('Br-beta', 2625.9*u.nm), 
                                            # ('Brackett-limit', 1458.0*u.nm)])
        # self.Helium = OrderedDict([('HeII', 303.783*u.AA),
                                   # ('HeII1640', 1640*u.AA), 
                                   # ('HeII4686', 4686*u.AA)])
        self.Optical_lines = OrderedDict([
                                          # ('[OII]', 3728*u.AA), 
                                          # ('[OIII4364]',4364*u.AA),
                                          ('[OIII4960]', 4960.295*u.AA), 
                                          ('[OIII5008]', 5008.239*u.AA),
                                          # ('[OI]', 6302*u.AA), 
                                          ('[NII]6549', 6549.86*u.AA), 
                                          ('[NII]6585', 6585.27*u.AA), 
                                          ('[SII]6718', 6718.29*u.AA), 
                                          ('[SII]6732', 6732.68*u.AA)])

        self.Optical_obsorption = OrderedDict([('Na_D1', 5890*u.AA), ('Na_D2', 5896*u.AA)])
 
class Spectrum(object):
    """the data structure to handle spectrum
    """
    def __init__(self, specchan=None, specdata=None, reffreq=None, refwave=None, z=None):
        """ initialize the spectrum
        """
        self.specchan = specchan
        self.specdata = specdata
        self.reffreq = reffreq
        self.refwave = refwave
        if self.reffreq is None:
            if self.refwave is not None:
                self.reffreq = (const.c/self.refwave).to(u.GHz)
        if self.refwave is None:
            if self.reffreq is not None:
                self.refwave = (const.c/self.reffreq).to(u.um)
        if self.reffreq is not None:
            self.specvel = self.velocity(reffreq=reffreq)
        else:
            self.specvel = None
        if z is not None:
            self.to_restframe(z)
    @property
    def channels(self):
        return list(range(len(self.specchan)))
    @property
    def unit(self):
        return self.specchan.unit

    def wavelength(self, units=u.um):
        return convert_spec(self.specchan, units)

    def frequency(self, units=u.GHz):
        return convert_spec(self.specchan, units)

    def to_restframe(self, z):
        self.restfreq = convert_spec(self.specchan, 'GHz')*(z+1)
        self.restwave = convert_spec(self.specchan, 'um')/(z+1)

    def velocity(self, reference=None):
        return convert_spec(self.specchan, 'km/s', reference)

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

    def fit_gaussian(self, plot=False, ax=None, **kwargs):
        """fitting the single gaussian to the spectrum
        """
        fit_p = fitting.LevMarLSQFitter()
        spec_selection = self.specdata > np.percentile(self.specdata, 85)
        amp0 = np.mean(self.specdata[spec_selection])
        if self.specvel is not None:
            # fit a gaussian at the vel=0
            ## central velocity
            vel0 = np.median(self.specvel[spec_selection])
            p_init = models.Gaussian1D(amplitude=amp0, mean=vel0, 
                                       stddev=50*u.km/u.s)
                    # bounds={"mean":np.array([-1000,1000])*u.km/u.s, 
                            # "stddev":np.array([0., 1000])*u.km/u.s,})
            specchan = self.specvel
        else:
            # will try to fit a gaussian in the centre of the spectrum
            mean0 = np.median(self.specchan[spec_selection])
            p_init = models.Gaussian1D(amplitude=amp0, mean=mean0, 
                                       stddev=5*np.median(np.diff(self.specchan)))
            specchan = self.specchan
        p = fit_p(p_init, specchan, self.specdata)
        specfit = p(self.specchan)
        if plot:
            if ax is None:
                fig = plt.figure(figsize=(7,6))
                ax = fig.add_subplot(111)
                ax.step(specchan, self.specdata, label='data')
                ax.plot(specchan, specfit, label='mode')
        fitobj = Fitspectrum()
        fitobj.fitparams = p.param_sets.flatten()
        fitobj.fitnames = p.param_names
        fitobj.bestfit = specfit
        fitobj.specchan = specchan
        fitobj.specdata = self.specdata
        fitobj.fitfunc = p
        fitobj.chi2 = None #TODO #np.sum((bestfit - self.specdata)**2/std**2)
        return fitobj

class Fitspectrum():
    """the data structure to store the fitted spectrum
    """
    def __init__(self):
        """
        Args:
            fitparams: the fitted parameters
            bestfit: the best fit
            specchan: the spectrum axis
            specdata: the data axis
            chi2: the fitted chi2
        """
        self.fitparams = None
        self.fitnames = None
        self.bestfit = None
        self.fitfunc = None
        self.specchan = None
        self.specdata = None
        self.chi2 = None
    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        ax.step(self.specchan, self.specdata, where='mid', color='k')
        ax.step(self.specchan, self.bestfit, where='mid', color='red', alpha=0.8)

def convert_spec(spec_in, unit_out, reference=None, mode='radio'):
    """convert the different spectral axis

    Args:
        spec_in (astropy.Quantity): any valid spectral data with units
        unit_out (str or astropy.Unit): valid units for output spectral axis
        reffreq: reference frequency, with units
        refwave: reference wavelength, with units
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

def print_lines(z=0.0, family=None, unit='GHz', limits=None, reference=None, 
                mode='radio'):
    '''
    Parameters
    ----------
    format : str
        output format, either in frequency (freq) or wavelength (wave)
    '''
    db = Database()
    unit = u.Unit(unit)
    if limits is not None:
        limits = np.array(limits)
    if family is None:
        print("Specify the spectral line family:")
        print("all, CO, CO_isotop, H2O, dense_gas, optical")
    if family == 'simple':
        family_select = [db.CO_family, dbC_ion]
    elif family == 'water':
        family_select = [db.H2O]
    elif family == 'all' or family=='full':
        family_select = [db.CO_family, db.CO_13C, db.CO_18O, db.CO_17O, 
                         db.C_ion, db.H2O, db.HCN, db.HNC, db.Special, db.Other]
    else:
        family_select = []

    if family is not None:
        if 'optical' in family:
            family_select.append(db.Balmer_series)
            family_select.append(db.Paschen_series)
            family_select.append(db.Brackett_series)
            family_select.append(db.Optical_lines)
            family_select.append(db.Optical_obsorption)

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
