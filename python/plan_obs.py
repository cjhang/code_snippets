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


def make_plot(target_list={}, observatory=None, utc_offset=0, obs_time=None, show_sun=True, show_moon=True):
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


    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
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
    ax.set_ylim(-10, 90)
    ax.legend(loc='upper right')
    # plt.tight_layout()
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

    # location of ZDJ Observatory
    vla = EarthLocation(lon=-107.61828*u.deg, lat=34.07875*u.deg, height=2124*u.m)
    
    make_plot(target_list=targets, observatory=vla, obs_time=obs_time, utc_offset=utc_offset, show_sun=False)

    if False:
        from astroplan import Observer, FixedTarget
        from astroplan.plots import plot_sky

        # define the coodinate of targets
        virgo = SkyCoord('12h27m0s', '+12d43m0s', frame='icrs')
        MaNGA_8335_6101 = SkyCoord(215.229240711*u.deg, 40.1210273909*u.deg, frame='icrs')
        plan1 = FixedTarget(name='Virgo Cluster', coord=virgo)
        plan2 = FixedTarget(name='MaNGA:8335-6101', coord=MaNGA_8335_6101)
        # observertory
        obs_loc = EarthLocation(lat=32.1218*u.deg, lon=118.96097*u.deg, height=39*u.m)
        plan_site = Observer(location=obs_loc, name="ZDJO", timezone="Asia/Shanghai")

        # define time window
        obs_time = Time('2020-12-01 0:00') - 8*u.hour # also need minus the time offset
        start = obs_time
        end = obs_time + 10*u.hour
        time_window = start + (end - start) * np.linspace(0, 1, 20)

        plot_sky(plan1, plan_site, time_window, style_kwargs={'color': 'r'})
        plot_sky(plan2, plan_site, time_window, style_kwargs={'color': 'g'})

        # uncomment below to get the position of moon. 
        # It seems the astroplan not support the NonFixedTarget, so there is something wrong the legend
        #moon = get_moon(time_window)
        #plot_sky(moon, plan_site, time_window, style_kwargs={'color': 'b', 'alpha': 0.2})

        plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))  
        plt.show()
