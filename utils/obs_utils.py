#!/usr/bin/env python

"""This is a stand alone file to assistant the plan of astronomical observation

Requirements:
    python >3.5
    numpy 
    matplotlib >2.0
    astropy >1.0

Usage:

Author: Jianhang Chen
Email: cjhastro@gmail
History:
    2020.07.20: first release, version=0.0.1

"""

# keep track of your version
version = '0.0.1'

# import all the required packages
import numpy as np
import os
from matplotlib import pylab as plt

from astropy import units as u
from astropy.coordinates import get_sun,get_moon, AltAz, SkyCoord, EarthLocation
from astropy.time import Time
from astropy.cosmology import Planck18 as cosm
import astropy.constants as const
from scipy import special, interpolate
from astropy.modeling import models
from astropy.io import fits
from astropy.table import Table

def flux_to_luminosity(flux=None, luminosity=None, z=None, obsfreq=None, restfreq=None, 
                       obswave=None, restwave=None, luminosity_unit=None, flux_unit=None):
    """convert the integrated flux into luminosity
    ref: Solomon+2005, ARAA

    \nu_0 L(\nu_0) = 4\pi D_L^2 \nu_{obs} S(\nu_{obs})
    dv = d\nu * c / \nu0
    L_co = 4\pi/c D_L^2 \nu_{rest}/(1+z) S_co

    Args:
        flux: the integrated flux, in [Jy km/s] or [K km/s]
        luminosity: the integrated luminosity, in [Lsun] or [K km/s pc^2]
        z: the redshift, dimentionless
        restfreq: in equivalent with GHz

    Return: 
        unit.Quantity (in solar Luminosity by default)

    # for the 1.04e-3 factor in Solomon2005:
    factor = 4*np.pi/(const.c.to(u.km/u.s).value)*1e-23*1e9*(u.Mpc.to(u.cm))**2/(u.Lsun.to(u.erg/u.s))
    """
    if z is None:
        raise ValueError("redshift `z` is required for the flux<->luminosity calculation!")
    luminosity_dist = cosm.luminosity_distance(z)
    if restfreq is None:
        if obsfreq is not None:
            restfreq = obsfreq * (1+z)
        elif restwave is not None:
            restfreq = const.c / restwave
        elif obswave is not None:
            restfreq = const.c / obswave * (1+z)
        else:
            raise ValueError("rest frequency`restfreq` is required for the flux<->luminosity calculation!")
        if not restfreq.unit.is_equivalent(u.Hz):
            raise ValueError('The reference frequency/wavelength is not with correct units!')
    if flux is not None:
        if luminosity_unit is None:
            if flux.unit.is_equivalent(u.Jy*u.km/u.s):
                luminosity_unit = u.Lsun
        if luminosity_unit.is_equivalent(u.Lsun):
            luminosity = flux*4*np.pi/const.c*luminosity_dist**2*restfreq/(1+z)
            return luminosity.to(luminosity_unit)
        elif luminosity_unit.is_equivalent(u.K*u.km/u.s*u.pc**2):
            luminosity = flux*const.c**2/(2*const.k_B)*luminosity_dist**2/(1+z)/restfreq**2
            return luminosity.to(luminosity_unit)
    elif luminosity is not None:
        if flux_unit is None:
            flux_unit = u.Jy*u.km/u.s
        if luminosity.unit.is_equivalent(u.Lsun):
            flux = luminosity/4/np.pi*const.c/luminosity_dist**2/restfreq*(1+z) 
            return flux.to(flux_unit)
        elif luminosity.unit.is_equivalent(u.K*u.km/u.s*u.pc**2):
            flux = luminosity/const.c**2*(2*const.k_B)/luminosity_dist**2*(1+z)*restfreq**2
            return flux.to(flux_unit)

def brightness_to_luminosity(intensity=None, luminosity_prime=None, z=None, Omega_sb=None):
    """convert between the brightness temperature and brightness luminosity
    It is mainly used for single-dish telescope

    Args:
        intensity: the brightness temperature, in u.K*u.km/u.s
        luminosity_prime: the brightness luminosity, in u.K*u.km/u.s*u.pc**2
        Omega_sb: solid angle of the signal, which is source convolved with the beam.
                  in arcsec**2
    """
    if z is None:
        raise ValueError("redshift `z` is required for the flux<->luminosity calculation!")
    luminosity_dist = cosm.luminosity_distance(z).to(u.Mpc).value
    if Omega_sb is None:
        raise ValueError('Please specify the solid angle of the signal')
    if intensity is not None:
        if isinstance(intensity, u.Quantity):
            intensity = intensity.to(u.K*u.km/u.s).value
        luminosity_prime = 23.5 * intensity * Omega_sb * luminosity_dist**2/(1+z)**3
        return luminosity_prime*(u.K*u.km/u.s*u.pc**2)
    if luminosity_prime is not None:
        if isinstance(luminosity_prime, u.Quantity):
            luminosity_prime = luminosity_prime.to(u.K*u.km/u.s*u.pc**2).value
        intensity = luminosity_prime/23.5/Omega_sb/luminosity_dist**2*(1+z)**3
        return intensity*(u.K*u.km/u.s)

def flux_to_T(flux=None, temperature=None, Omega=None, bmaj=None, bmin=None, lamb=None, freq=None):
    """convert between brightness temperature and flux density
    ref: https://science.nrao.edu/facilities/vla/proposing/TBconv
    T = \frac{\lambda^2}{2k_B\Omega} S
    \Omega = \pi bmaj*bmin/(4\ln2)

    flux: in equivalent to Jy/beam
    """
    if Omega is None:
        Omega = np.pi*bmaj*bmin/(4*np.log(2))
    if freq is not None:
        lamb = (const.c/freq).to(u.mm)
    if lamb is not None:
        assert lamb.unit.is_equivalent(u.mm)
    flux2T = lamb**2/(2*Omega*const.k_B)
    if flux is not None:
        temperature =  flux * flux2T
        return temperature.to(u.K)
    if temperature is not None:
        flux = temperature / flux2T
        return flux.to(u.Jy)

def KS_law(sigma_SFR=None, sigma_gas=None, slope=1.4, factor=2.5e-4, with_units=None):
    """covert between SFR surface density and the gas surface density

    The default values come from Kennicutt1998, formula 7
    sigma_SFR: in Msun/year/kpc2
    sigma_gas: Msun/pc2
    """
    if sigma_SFR is None:
        if isinstance(sigma_gas, u.Quantity):
            sigma_gas = sigma_gas.to(u.Msun/u.pc**2).value
            if with_units is None:
                with_units = True
        sigma_SFR = factor * sigma_gas**slope
        if with_units:
            sigma_SFR = sigma_SFR *(u.Msun/u.yr/u.kpc**2)
        return sigma_SFR
    if sigma_gas is None:
        if isinstance(sigma_SFR, u.Quantity):
            sigma_SFR = sigma_SFR.to(u.Msun/u.yr/u.kpc**2).value
            if with_units is None:
                with_units = True
        sigma_gas = (sigma_SFR/factor)**(1/slope)
        if with_units:
            sigma_gas = sigma_gas *(u.Msun/u.pc**2)
        return sigma_gas

def KS_law_integrated(SFR=None, M_H2=None, R=None):
    """
    SFR: in Msun/yr
    M_H2: in Msun
    """
    if R is None:
        R = 5*u.kpc
    averaged_area = np.pi*R**2
    if M_H2 is None:
        if not isinstance(SFR, u.Quantity):
            SFR = SFR * u.Msun/u.yr
        sigma_gas = KS_law(sigma_SFR=SFR/averaged_area, with_units=True)
        M_H2 = (sigma_gas * averaged_area).to(u.Msun)
        return M_H2
    if SFR is None:
        if not isinstance(M_H2, u.Quantity):
            M_H2 = M_H2 * u.Msun
        sigma_SFR = KS_law(sigma_gas=M_H2/averaged_area, with_units=True)
        SFR = sigma_SFR * averaged_area
        return SFR

def KS_law_FIR(SFR=None, FIR=None):
    """convert between SFR and line flux
    
    FIR: SFR [Msun/yr] = 4.5x10^-44 L_FIR [erg/s]
    """
    if FIR is not None:
        return 4.5e-44*FIR

def dust_to_MH2(S_dust=None, freq_obs=None, z=None, beta=1.8, Td=25, alpha_dust=6.7e19, delta_gd=150, 
                convention='Tacconi2020', debug=False):
    """convert the measure dust flux to the gas mass
    """
    if convention == 'Tacconi2020':
        luminosity_dist = cosm.luminosity_distance(z)
        M_H2 = (S_dust*luminosity_dist**2).to(u.mJy*u.Gpc**2).value*(1+z)**(-3-beta)*(freq_obs/352/u.GHz)**(-2-beta)*(6.7e19/alpha_dust)*(delta_gd/150)
        return (M_H2*1e10*u.M_sun).to(u.M_sun)
    if convention == 'Scoville2016':
        alpha_dust_with_unit = alpha_dust * u.erg/u.s/u.Hz/u.M_sun
        luminosity_dist = cosm.luminosity_distance(z)
        freq_rest = freq_obs * (1+z)
        ref_T = (const.h*353*u.GHz/const.k_B).to(u.K)
        obs_T = (const.h*freq_rest/const.k_B).to(u.K)
        Td = Td*u.K
        k_corr = (353*u.GHz/freq_rest)**(3+beta)*(np.exp(obs_T/Td)-1)/(np.exp(ref_T/Td)-1)
        if debug:
            print('k_corr', k_corr)
        L_v_850um = (4*np.pi*S_dust*k_corr*luminosity_dist**2/(1+z)).to(u.erg/u.s/u.Hz)
        M_H2 = (L_v_850um / alpha_dust_with_unit).to(u.M_sun)
        return M_H2


def CO_to_MH2(flux=None, dv=1, freq_obs=None, z=None, alpha_co=4.36, 
              Rj1=None, transition='4-3', debug=False):
    """convert CO into gas mass

    Args:
        flux: the measured flux, [with units equivalent to Jy if dv is set]
        dv: the velocity of the channel width, if flux is integrated over
            the velocity (with units of Jy*km/s), the dv can be set to 1
    """
    alpha_co_with_unit = alpha_co*u.Msun/(u.K*u.km/u.s*u.pc**2)
    CO_ladder = [1,1.3, 1.8, 2.4]
    #restfreq = {'1-0':115.27*u.GHz, '2-1':230.54*u.GHz, '3-2': 345.80*u.GHz,
    #            '4-3':461.04*u.GHz, '5-4':576.27*u.GHz, '6-5': 691.47*u.GHz}
    luminosity_dist = cosm.luminosity_distance(z)
    flux2L_prime = const.c**2/(2*const.k_B) * dv * luminosity_dist**2 / (1+z)**3 / freq_obs**2
    if Rj1 is None:
        Rj1 = CO_ladder[int(transition[-1])]
    if debug:
        print('transition={transition}; Rj1={Rj1};')
    L_co_prime = flux * flux2L_prime
    L_co10_prime = L_co_prime * Rj1
    M_H2 = L_co10_prime * alpha_co_with_unit
    return M_H2.to(u.M_sun)

def CI_to_MH2(flux, dv=1, freq_obs=None, z=None, alpha_ci=18.7, Q_10=0.48, X_CI=1.6e-5,):
    """convert the CI(1-0) flux into molecular mass
    The equation is followed Dune et al. 2022.
    
    Args:
        flux * dv show be in the units of Jy km/s
    """
    luminosity_dist = cosm.luminosity_distance(z)
    flux2L_prime = const.c**2/(2*const.k_B) * dv * luminosity_dist**2 / (1+z)**3 / freq_obs**2
    L_prime = (flux * flux2L_prime).to(u.K*u.km/u.s*u.pc**2)
    if alpha_ci is not None:
        alpha_ci_with_unit = alpha_ci * u.M_sun/(u.K*u.km/u.s*u.pc**2)
        return (alpha_ci_with_unit * L_prime).to(u.M_sun)
    else:
        return 9.51e-5/X_CI/Q_10*L_prime.value*u.M_sun

    # luminosity_dist = cosm.luminosity_distance(z).to(u.Mpc).value
    # flux_value = (flux*dv).to(u.Jy*u.km/u.s).value
    # return 0.0127/X_CI/Q_10*(luminosity_dist**2/(1+z))*flux_value*u.M_sun


def flux_CO(M_H2=None, SFR=None, FIR=None, z=None, Re=None, alpha_CO=4.3, transition='1-0'):
    """return CO flux base on total molecular mass
    """
    alpha_CO_with_unit = alpha_CO*u.Msun/(u.K*u.km/u.s*u.pc**2)
    # CO_ladder = {'R21':1.3, 'R32':1.8, 'R43':2.4}
    CO_ladder = [1,1.3, 1.8, 2.4]
    # CO_ladder_cumproduct = np.cumprod(CO_ladder)
    # print(CO_ladder_cumproduct)
    restfreq = {'1-0':115.27*u.GHz, '2-1':230.54*u.GHz, '3-2': 345.80*u.GHz,
                '4-3':461.04*u.GHz, '5-4':576.27*u.GHz, '6-5': 691.47*u.GHz}
    co_restfreq = restfreq[transition]
    co_obsfreq = co_restfreq / (1+z)
    # print("obsfreq", co_obsfreq)

    if SFR is not None:
        # first convert SFR into FIR based on KS-law
        if transition > '4-3':
            # Kennicutt1998, ARAA, Formula 4, for starburst
            FIR = SFR.to(u.Msun/u.yr).value / 4.5e-44 *u.erg/u.s 
        else:
            # Kennicutt1998, ARAA, Formula 5
            # the maximun SFR, assuming tau_dep~100Myr
            # which will return the maximum required molecular gas
            M_H2 = 1e10 * (SFR.to(u.Msun/u.yr).value)/100
            # M_H2 = KS_law_integrated(SFR=SFR, R=Re).to(u.Msun).value
    if M_H2 is not None:
        if isinstance(M_H2, u.Quantity):
            mass_unit = M_H2.unit
        else: 
            mass_unit = u.Unit('solMass')
            M_H2 = M_H2 * mass_unit
            print(f"M_H2: {M_H2}")
    if FIR is not None:
        # adopting the Liu+2015, ApJL, 810, L14
        log10_FIR = np.log10(FIR.to(u.Lsun).value)
        if transition == '4-3':
            log10_Lco = (log10_FIR - 1.96)/1.06
        elif transition == '5-4':
            log10_Lco = (log10_FIR - 2.27)/1.07
        elif transition == '6-5':
            log10_Lco = (log10_FIR - 2.56)/1.10
        Lco = 10**(log10_Lco)*u.K*u.km/u.s*u.pc**2
        # convert luminosity to flux    
        flux = flux_to_luminosity(luminosity=Lco, z=z, restfreq=restfreq[transition])
    elif M_H2 is not None:
        luminosity_dist = cosm.luminosity_distance(z)
        L_co10 = M_H2 / alpha_CO_with_unit
        Rj1 = CO_ladder[int(transition[-1])]
        L_co = L_co10 * Rj1
        # print("L_CO", L_co.to(u.K*u.km/u.s*u.pc**2))
        # S_co = L_co / 1.04e-3 / dfreq / freq_rest * (1+z) / (dist_luminosity.to(u.Mpc).value)**2
        flux2L = const.c**2/(2*const.k_B) * luminosity_dist**2 / (1+z)**3 / co_obsfreq**2
        flux = L_co / flux2L
    return flux.to(u.Jy*u.km/u.s)

def sigma_CO(sigma_SFR=None, z=None, Omega_sb=None, alpha_CO=4.3, transition='1-0'):
    """return CO surface flux density base on star-formation rate surface density

    sigma_SFR: in Msun/year/kpc2
    Omega_sb in arcsec**2
    """
    if isinstance(Omega_sb, u.Quantity):
        Omega_sb = Omega_sb.to(u.arcsec**2).value
    alpha_CO_with_unit = alpha_CO*u.Msun/(u.K*u.km/u.s*u.pc**2)
    # CO_ladder = {'R21':1.3, 'R32':1.8, 'R43':2.4}
    CO_ladder = [1, 1.3, 1.8, 2.4]
    # print(CO_ladder_cumproduct)
    restfreq = {'1-0':115.27*u.GHz, '2-1':230.54*u.GHz, '3-2': 345.80*u.GHz,
                '4-3':461.04*u.GHz, '5-4':576.27*u.GHz, '6-5': 691.47*u.GHz}
    co_restfreq = restfreq[transition]
    co_obsfreq = co_restfreq / (1+z)
    luminosity_dist = cosm.luminosity_distance(z)
    # print("obsfreq", co_obsfreq)

    sigma_gas = KS_law(sigma_SFR=sigma_SFR, with_units=True) 
    print('sigma gas:', sigma_gas)
    sigma_L_co10 = sigma_gas / alpha_CO_with_unit # u.K*u.km/u.s
    print('sigma_L_co10', sigma_L_co10)
    Rj1 = CO_ladder[int(transition[-1])]
    print('Rij', Rj1)
    sigma_L_co = sigma_L_co10 * Rj1
    flux2L = const.c**2/(2*const.k_B) * luminosity_dist**2 / (1+z)**3 / co_obsfreq**2
    sigma_flux = sigma_L_co / flux2L
    ## another method (untested): need conversion between Temperature and flux
    ## flux2L = 23.5 * Omega_sb * (luminosity_dist.to(u.Mpc).value)**2 / (1+z)**3 * (u.K*u.km/u.s*u.pc**2)/(u.K*u.km/u.s)
    ## sigma_I = sigma_L_co / flux2L # in units u.K*u.km/u.s/u.pc**2
    ## print('sigma_I', sigma_I)
    ## sigma_flux = flux_to_T(temperature=sigma_I.to(u.K*u.km/u.s/u.pc**2).value*u.K, 
                              # Omega=Omega_sb, freq=co_obsfreq) * u.km/u.s/u.pc**2
    # convert the pc to arcsec
    sigma_flux_angular = sigma_flux*(cosm.kpc_proper_per_arcmin(z).to(u.pc/u.arcsec))**2
    # area_to_beam = Omega_sb*u.arcsec**2
    # return (sigma_flux_angular*area_to_beam).to(u.Jy*u.km/u.s/u.arcsec**2)
    return (sigma_flux_angular).to(u.Jy*u.km/u.s/u.arcsec**2)

def sersic_profile(r, n=None, Re=None, Ie=None, Itot=None):
    # from 2005PASA...22..118G
    if Itot is not None:
        G2n = special.gamma(2 * n)
        bn = special.gammaincinv(2*n, 0.5)
        Ie = Itot / ( Re**2*2*np.pi*n*np.exp(bn)/(bn**(2*n)) * G2n)
    sersic1D_model = models.Sersic1D(amplitude=Ie, r_eff=Re, n=n)
    return sersic1D_model(r)

def sersic_source(n=None, Re=None, Ie=None, ellip=0, theta=0, Itot=None,
                  pixel_scale=0.1, plot=False, 
                  savefile=None, ra=None, dec=None, overwrite=False):
    sersic2D_model = models.Sersic2D(amplitude=Ie, r_eff=Re, n=n, ellip=ellip, theta=theta)
    # r = np.arange(-5.*Re, 5.*Re+pixel_scale, pixel_scale)
    r = np.arange(-5.*Re, 5.*Re, pixel_scale)
    rx, ry = np.meshgrid(r, r)
    model = sersic2D_model(rx, ry)
    ny, nx = model.shape
    if plot:
        fig, ax = plt.subplots()
        extent = [rx.min(), rx.max(), ry.min(), ry.max()]
        im = ax.imshow(model, origin='lower', extent=extent)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Brightness', rotation=270, labelpad=15)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
    if savefile is not None:
        header = fits.Header()
        header['CDELT1'] = -1.0*pixel_scale/3600.0
        header['CDELT2'] = pixel_scale/3600.0
        header['CUNIT1'] = 'deg'
        header['CUNIT2'] = 'deg'
        header['CTYPE1'] = 'RA---SIN'
        header['CTYPE2'] = 'DEC--SIN'
        header['CRPIX1'] = nx/2.0
        header['CRPIX2'] = ny/2.0
        header['CRVAL1'] = ra
        header['CRVAL2'] = dec
        imagehdu = fits.PrimaryHDU(data=model, header=header)
        header.update({'history':'created by image_tools.Image',})
        imagehdu.writeto(savefile, overwrite=overwrite)
    else:
        return model

def plot_altitude(target_list={}, observatory=None, utc_offset=0, obs_time=None, show_sun=True, show_moon=True):
    """plot the position of science target during observation
    """
    
    # define the observation time
    delta_hours = np.linspace(-12, 12, 100)*u.hour
    obstimes = obs_time + delta_hours - utc_offset
    # observertory
    altaz_frames = AltAz(location=observatory, obstime=obstimes)

    target_altaz_list = {}
    for name, skcoord in target_list.items():
        target_altaz_list[name] = skcoord.transform_to(altaz_frames)


    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(121)
    for name, pos in target_altaz_list.items():
        ax.scatter(delta_hours, pos.alt, label=name, s=8)
    ax.set_title("Observed on {}".format(obs_time.fits))
    # get position of sun and moon
    sun = get_sun(obstimes).transform_to(altaz_frames)
    if show_sun:
        ax.plot(delta_hours, sun.alt, 'r', label='sun')
    if show_moon:
        moon = get_moon(obstimes).transform_to(altaz_frames)
        ax.plot(delta_hours, moon.alt, 'k--', label='moon')

    ax.fill_between(delta_hours.to('hr').value, -90, 90, sun.alt < -0*u.deg, 
                     color='0.5', zorder=0, alpha=0.5)
    ax.fill_between(delta_hours.to('hr').value, -90, 90, sun.alt < -18*u.deg, 
                     color='k', zorder=0, alpha=0.5)
    ax.set_xlabel('LST offset')
    ax.set_ylabel('Altitude')
    # ax.set_ylim(-10, 90)
    ax.legend(loc='upper right')
    # plt.tight_layout()
    
    ax = fig.add_subplot(122, projection='polar')
    for name, pos in target_altaz_list.items():
        ax.plot(pos.az/180*np.pi, np.cos(pos.alt), label=name, marker='.', ms=8)
    if show_sun:
        ax.plot(sun.az/180*np.pi, np.cos(sun.alt), 'r.', label='sun')
    if show_moon:
        moon = get_moon(obstimes).transform_to(altaz_frames)
        ax.plot(moon.az/180*np.pi, np.cos(moon.alt), 'k--', label='moon')
    ax.set_ylim(0, 1)

    plt.show()

def ABmag2flux(mag):
    "AB magnitude to flux in Jy"
    return 10**((8.9 - mag)/2.5)

#####################################################
#  plot functions

def read_alma_pwv(pwv=None):
    pwv_list = np.array([0.5,1.0,1.5,2.0,2.5])
    tab_pwv = Table.read(os.path.join(os.path.expanduser('~'),
                         'Data/telescopes/ALMA/pwv_0.5_1.0_1.5_2_2.5.dat'), 
                         names=['frequency','0.5','1.0','1.5','2.0','2.5'],
                         format='ascii')
    return tab_pwv

def get_alma_transmission(freq, pwv=0.5):
    """
    Args:
        freq: frequency, in GHz
        pwv: precipate water vapor level: 0.5,1.0,1.5,2.0,2.5
    """
    tab_pwv = read_alma_pwv()
    x_freq = tab_pwv['frequency']
    y_pwv = tab_pwv[str(pwv)]
    cubicinterpolator = interpolate.CubicSpline(x_freq, y_pwv)
    return cubicinterpolator(freq)


def plot_alma_spw(freq_range=[90, 1000], lines=None, lines_names=None,
                  pwv=1.0, show_alma_bands=True):
    tab_pwv = read_alma_pwv()
    freq_selection = (tab_pwv['frequency']>freq_range[0]) & (tab_pwv['frequency']<freq_range[1])
    fig, ax = plt.subplots(1,1)
    x_freq = tab_pwv['frequency'][freq_selection]
    y_pwv = tab_pwv[str(pwv)][freq_selection]
    ax.plot(x_freq, y_pwv, color='black', alpha=0.5)
    height = 1.2*np.max(y_pwv)
    if show_alma_bands:
        band_list = {'Band3':[84, 116], 'Band4':[125, 163], 'Band5':[163, 211], 
                     'Band6':[211, 275], 'Band7':[275, 373], 'Band8':[385, 500], \
                     'Band9':[602, 720], 'Band10':[787, 950]}
        for name, freq in band_list.items():
            if (freq[0] > freq_range[0]) | (freq[1] < freq_range[1]):
                ax.broken_barh(([freq[0], np.diff(freq)[0]],), 
                              np.array([0, 1])*height, 
                              facecolor='lightblue', edgecolor='grey', \
                              linewidth=1, alpha=0.2)
            if (freq[0] > freq_range[0]) & (freq[1] < freq_range[1]):
                ax.text(np.mean(freq), 0.9*height, name, fontsize=8, 
                        ha='center', va='center', alpha=0.5, )
    for line, name in zip(lines, lines_names):
        ax.text(line, 0.55*height, name, ha='center', va='center', rotation='vertical')
        ax.vlines(line, 0, 0.5*height, lw=4)
                
    ax.set_xlim(freq_range[0], freq_range[1])
    ax.set_ylim(0, height)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Transmission')
    plt.show()


if __name__ == '__main__':

    obs_time = Time('2020-12-01 00:00') # Observatory local time
    utc_offset = -7*u.hour #Observatory time zone

    # exmaple:
    # targets = {'virgo': SkyCoord('12h27m0s', '+12d43m0s', frame='icrs'),
               # 'manga': SkyCoord(215.229240711*u.deg, 40.1210273909*u.deg, frame='icrs')}
    targets = {'PGC38025': SkyCoord('12h2m37.08s', '+64d22m35.20s', frame='icrs'),
            'J1148+5924': SkyCoord('11h48m50.358s', '+59d24m56.382s', frame='icrs'),
            '3C286': SkyCoord('13h31m08.288s', '+30d30m32.959s', frame='icrs'),
            }

    # location of ZDJ Observatory of Nanjing University
    vla = EarthLocation(lon=-107.61828*u.deg, lat=34.07875*u.deg, height=2124*u.m)
    
    plot_altitude(target_list=targets, observatory=vla, obs_time=obs_time, utc_offset=utc_offset, show_sun=True)

