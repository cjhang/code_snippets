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
from matplotlib import pylab as plt

from astropy import units as u
from astropy.coordinates import get_sun
from astropy.coordinates import get_moon
from astropy.coordinates import AltAz
from astropy.coordinates import SkyCoord
from astropy.coordinates import EarthLocation
from astropy.time import Time


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

