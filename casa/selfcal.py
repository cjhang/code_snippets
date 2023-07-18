"""
This script is designed to conduct self-calibration

History:
    2023.06.20 first release, ver=1.0
    2023.06.21 better organised naming, ver=1.1
"""
__version__ = '1.1.1'

import os

from casatasks import tclean, gaincal, applycal
from casatools import table
from casatasks import rmtables, split

try:
    from plot_utils import check_gain
except:
    print("Warning: cannot import the plotting library!")


def selfcal(vis, basename=None, imagename=None, imagedir='./images', field='', refant='', combine='',
            minsnr=3.0, overwrite_model=False,
            solints=['60s', '20s', 'inf'], 
            calmodes=['p', 'p', 'ap'],
            **kwargs):
    """
    The key of selfcal is to set optimal time interval;
    In special case, if the source is week and known to did no vary with time and frequency,
    the combine can be set to 'spw', 'scan', or 'scan,spw' to boost SNR;

    The calibrated data coloumn of the vis will be rewrite!
    It is recommended to make the input vis as the splitted data from calibrated data

    examples:

    selfcal(vis, imagename='test', 
            specmode='mfs',
            deconvolver='hogbom',
            imsize=[250,250],
            cell=['0.1arcsec'],
            weighting='natural',
            threshold='0mJy',
            niter=5000,
            interactive=True,
    """
    # set the record
    gainfields = []
    gaintables = []
    if basename is None:
        basename = os.path.basename(vis)
    if imagename is None:
        imagename = os.path.join(imagedir,basename)

    tb = table()
    tb.open(vis)
    colnames = tb.colnames()
    tb.close()
    if ('MODEL_DATA' not in colnames) or overwrite_model:
        # clean the first image and get the initial model
        os.system('rm -rf {}.*'.format(imagename))
        tclean(vis=vis, field=field,
           imagename=imagename,
           savemodel='modelcolumn',
           **kwargs)
    else:
        print('re-use existing models, use `overwrite_model=True` to force rewrite.')


    # start the self calibration cycles
    ncycle = len(solints)
    for i in range(ncycle):
        solint = solints[i]
        calmode = calmodes[i]

        cycle_suffix = 'selfcal.c{}{}{}'.format(i, calmode, solint)
        cycle_solution = '{}.{}.gcal'.format(basename, cycle_suffix)
        calibrated_vis = basename+'.{}'.format(cycle_suffix)
        calibrated_imagename = imagename+'.{}'.format(cycle_suffix)

        os.system("rm -rf {}".format(cycle_solution))
        gaincal(vis=vis, field=field, refant=refant,
                minsnr=minsnr,
                solint=solints[0], calmode="p",
                caltable=cycle_solution,
                gaintype="G" # solve for each polarisation independently
                )

        if 'a' in calmode: plot_amp = True
        else: plot_amp = False
        check_gain(cycle_solution, plot_phase=True, plot_amp=plot_amp,
                   plotdir='plots/cycle{}'.format(i))

        # apply the solution
        applycal(vis=vis, field=field,
             gaintable=[cycle_solution],
             interp="linear")
        gainfields.append(field)
        gaintables.append(cycle_solution)

        # split the calibrated data
        rmtables(calibrated_vis)
        split(vis=vis, outputvis=calibrated_vis, datacolumn="corrected")

        # assign the new vis
        vis = calibrated_vis

        # imaging the updated data

        tclean(vis=vis, field=field, imagename=calibrated_imagename,
           savemodel='modelcolumn', **kwargs)

    return gainfields, gaintables
