"""
This script is designed to conduct self-calibration
Author: Jianhang Chen
Contact: cjhastro@gmail.com

History:
    2023.06.20 first release, ver=1.0
    2023.06.21 better organised naming, ver=1.1
    2024.08.08 change output folder names and keep the flags, ver=1.2
"""
__version__ = '1.2.6'

import os

from casatasks import tclean, gaincal, applycal
from casatools import table
from casatasks import rmtables, split, mstransform, flagmanager

try:
    from plot_utils import check_gain
    has_plot_lib = True
except:
    has_plot_lib = False
    print("Warning: cannot import the plotting library!")


def selfcal(vis, basename=None, imagedir='./selfcal_images', 
            field='', spw='', refant='', combine='',
            suffix='.selfcal', spwmap=[],
            minsnr=5.0, overwrite_model=True, gaintype="G",
            solints=['60s', 'inf'], 
            calmodes=['p', 'ap'],
            imsize=512, cell='0.1arcsec', niter=10000, interactive=True, 
            skip_last_model=True,
            **kwargs):
    """
    The key of selfcal is to set optimal time interval and mark the true signals;
    In special case, if the source is week, there are possible ways to improve selfcal:
    1. Set the combine='scan' to calculate the solutions over a long interval;
    2. If the source is not varying too much with frequency, the combine can be set to 'spw', 
       or 'scan,spw' to boost SNR; If combine includes 'spw', then the spwmap should be
       set for the applycal
    3. `gaintype` is set to 'G' by default to solve the two polarisation independently.
       If the source is known to be unpolarised, it can be set to 'T' to average both 
       polarisation to boost the S/N
    4. selfcal involves multiple tclean steps, it is thus recommended to bin the channels
       first to speed up the processes

    If there are multiple phase-only calibration, it is safer to only apply one (the best)
    phase-only solution before the amplitude self-cal. As the applycal can easily build-in 
    errors.

    The calibrated data coloumn of the vis will be rewrite!
    It is recommended to make the input vis as the splitted data from calibrated data

    Args:
        skip_last_imaging (bool): set it to true to skip the final imaging after all the selfcal

    examples:

    selfcal(vis, basename='test', 
            specmode='mfs',
            deconvolver='hogbom',
            imsize=[250,250],
            cell=['0.1arcsec'],
            weighting='natural',
            threshold='0mJy',
            niter=5000,
            interactive=True,
    """
    # keep the origal flagging
    vis_origin = vis
    flagmanager(vis, mode='save', versionname='OriginBeforeSelfcal')

    # set the record
    gainfields = []
    gaintables = []
    if basename is None:
        basename = os.path.basename(vis)
    imagename = os.path.join(imagedir, basename)

    tb = table()
    tb.open(vis)
    colnames = tb.colnames()
    tb.close()
    if ('MODEL_DATA' not in colnames) or overwrite_model:
        # clean the first image and get the initial model
        os.system('rm -rf {}.*'.format(imagename))
        tclean(vis=vis, field=field, spw=spw, datacolumn='data',
               imagename=imagename,
               savemodel='modelcolumn',
               imsize=imsize, cell=cell,
               niter=niter, interactive=interactive,
               **kwargs)
    else:
        print('re-use existing models, use `overwrite_model=True` to force rewrite.')

    # start the self calibration cycles
    ncycle = len(solints)
    for i in range(ncycle):
        solint = solints[i]
        calmode = calmodes[i]

        cycle_suffix = 'c{}{}{}{}{}{}{}'.format(i, calmode, solint, gaintype, minsnr,
                                              combine.replace(',',''), suffix)
        cycle_solution = '{}.{}.gcal'.format(basename, cycle_suffix)
        calibrated_vis = basename+'.{}'.format(cycle_suffix)
        calibrated_imagename = imagename+'.{}'.format(cycle_suffix)

        os.system("rm -rf {}".format(cycle_solution))
        gaincal(vis=vis, field=field, refant=refant, spw=spw,
                minsnr=minsnr, combine=combine,
                solint=solint, calmode=calmode,
                caltable=cycle_solution,
                gaintype=gaintype, # solve for each polarisation independently
                )

        if 'a' in calmode: plot_amp = True
        else: plot_amp = False
        if has_plot_lib:
            check_gain(cycle_solution, plot_phase=True, plot_amp=plot_amp,
                       plotdir='selfcal_plots/{}_{}'.format(
                                basename,cycle_suffix))

        # apply the solution
        applycal(vis=vis, field=field,
                 gaintable=[cycle_solution],
                 spwmap=spwmap,
                 interp="linear")
        gainfields.append(field)
        gaintables.append(cycle_solution)

        # split the calibrated data
        rmtables(calibrated_vis)
        mstransform(vis, outputvis=calibrated_vis, datacolumn='corrected') 

        # assign the new vis
        vis = calibrated_vis

        # imaging the updated data
        savemodel = 'modelcolumn'
        if i == ncycle-1:
            if skip_last_model:
                savemodel = 'none'
        tclean(vis=vis, field=field, spw=spw, imagename=calibrated_imagename,
               savemodel=savemodel, 
               imsize=imsize, cell=cell,
               niter=niter, interactive=interactive,
               **kwargs)
    # restore the flagging after all the steps
    flagmanager(vis_origin, mode='restore', versionname='OriginBeforeSelfcal')
    return gainfields, gaintables
