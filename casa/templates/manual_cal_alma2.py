# Simple manual calibration script for EVLA data
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
# History:
#   2019.09.30 First release for learning casa
#   2020.04.11 Update for reduce Centaurus_A
#   2020.11.22 


################ Basic Information ###############
# CALIBRATE_BANDPASS: 
# CALIBRATE_GAIN: 
# CALIBRATE_FLUX:
# OBSERVE_TARGET: 

#######################################################
#                  Data Prepare
#######################################################

# The program to maunual calibrate the calibrator data

asdmfile = ''
msfile = ''
importasdm(asdm=asdmfile, vis=msfile)

myrefant = ''
myreffreq = ''
fcal = '0'
bcal = '0'
gcal = '1'
allcal = '0,1'

myspw = ''
science_spw = '17,19,21,23'
science_field = ''
spw_bpphase =''     # the spw window to get the time variation for bandpass calibrator
myrefant = '1'      # reference antenna
mysolint = 'int'    # integration time
mycalwt = False     # calculate the weight, must be false for EVLA

if plot_utils is None:
    try:
        import plot_utils
        plot_results = True
    except:
        plot_results = False

# Flagging the useless data
flagdata(vis=msfile, mode='unflag', flagbackup=False)
flagdata(vis=msfile, mode='manual', autocorr=True, flagbackup=False)
# flagdata(vis=msfile, mode='manual', 
        # intent='*POINTING*,*ATMOSPHERE*', flagbackup=False)
flagdata(vis=msfile, mode='shadow', flagbackup=False)
flagmanager(vis=msfile, mode='list')

# Flag edge channel
# flagdata(vis = msfile, spw='17:0~3,19:0~3,21:0~3,23:0~3')

flagmanager(vis=msfile, mode='save', versionname='Priori')

if True:
    # Fix the ASDM SYSCal table issue, data before 2015
    from recipes.almahelpers import fixsyscaltimes
    fixsyscaltimes(vis=msfile)


#######################################################
#                  Prior Calibration
#######################################################
print("\n==============> Start Prior Calibration <=============\n")

##### Tsys Calibration
# It gives the first-order correction for atmospheric opacity as a funcction of time and freq
os.system("rm -rf {}.tsys".format(msfile))
gencal(vis = msfile, caltable = msfile+'.tsys', caltype = 'tsys')
# plot the results
# plotms(vis=msfile+'.tsys', xaxis='freq', yaxis='tsys', spw='', iteraxis='antenna', coloraxis='scan', showgui=True)

# flag edge channel
flagdata(vis = msfile+'.tsys', spw='17:0~3,19:0~3,21:0~3,23:0~3')

##### Water Vaper Calibration
os.system("rm -rf {}.wvr".format(msfile))
wvrgcal(vis = msfile,
        caltable = msfile+'.wvr',
        )
#> Cycle0, the offset should be set -1


##### Antenna position caliration


# Apply the Tsys and WVR calibration
from recipes.almahelpers import tsysspwmap
tsysmap = tsysspwmap(vis=msfile, tsystable=msfile+'.tsys')
for field in np.unique([fcal, bcal, gcal, science_field]):
    applycal(vis = msfile,
             field = field,
             spw = science_spw,
             gaintable = [msfile+'.tsys', msfile+'.wvr'],
             gainfield = [field, field],
             interp = 'linear',
             spwmap = [tsysmap,[]],
             calwt = True,
             flagbackup = False)
#> Check the calibrated data


splitmsfile = msfile+'.split' # the msfile after priori calibration
os.system('rm -rf {0} {0}.flagversions'.format(splitmsfile))
split(vis=msfile, field='', outputvis=splitmsfile,
      datacolumn='corrected', spw=science_spw)

flagmanager(vis = splitmsfile, mode = 'save',
            versionname = 'BeforeBandpassCal')


#######################################################
#             Bandpass and Gain Calibration
#######################################################

# gain cal in short time variation

setjy(vis = splitmsfile, field = fcal, reffreq = myreffreq,)
# set the flux manually
# setjy(vis = splitmsfile, field = fcal, reffreq = '233.0GHz', 
      # standard='manual', fluxdensity=[1.41, 0, 0, 0])

os.system('rm -rf {}.bpphase_int.cal'.format(splitmsfile)) 
gaincal(vis = splitmsfile,
        caltable = splitmsfile+'.bpphase_int.cal',
        field = bcal, # bandpass calibrator
        spw = '',
        minsnr = 2.0,
        solint = 'int',
        refant = myrefant,
        calmode = 'p')

# bandpass calibration
os.system('rm -rf {}.bandpass.cal'.format(splitmsfile)) 
bandpass(vis = splitmsfile,
         caltable = splitmsfile+'.bandpass.cal',
         field = '0', # bandpass calibrator
         solint = 'inf',
         minsnr = 2.0,
         combine = 'scan,field',
         refant = myrefant,
         solnorm = True,
         bandtype = 'B',
         gaintable = splitmsfile+'.bpphase_int.cal')


# phase short-time variation calibration
os.system('rm -rf {}.phase_int.cal'.format(splitmsfile)) 
gaincal(vis = splitmsfile,
        caltable = splitmsfile+'.phase_int.cal',
        field = allcal,
        solint = 'int',
        minsnr = 2.0,
        refant = myrefant,
        calmode = 'p',
        gaintable = splitmsfile+'.bandpass.cal')
 
# phase long-time variation calibration, used for science target
os.system('rm -rf {}.phase_scan.cal'.format(splitmsfile)) 
gaincal(vis = splitmsfile,
        caltable = splitmsfile+'.phase_scan.cal',
        field = allcal,
        solint = 'inf',
        minsnr = 2.0,
        refant = myrefant,
        calmode = 'p',
        gaintable = splitmsfile+'.bandpass.cal')
 

# amplitude calibration
os.system('rm -rf {}.amp.cal'.format(splitmsfile)) 
gaincal(vis = splitmsfile,
        caltable = splitmsfile+'.amp.cal',
        field = allcal, 
        solint = 'inf',
        refant = myrefant,
        calmode = 'ap',
        gaintable = [splitmsfile+'.bandpass.cal', splitmsfile+'.phase_int.cal'])

# Flux calibration
os.system('rm -rf {}.flux.cal'.format(splitmsfile)) 
fluxscale(vis = splitmsfile,
          caltable = splitmsfile+'.amp.cal',
          fluxtable = splitmsfile+'.flux.cal',
          reference = fcal,
          transfer = '') 

# Apply the short time gain_cal and bandpass to the calibrator

for field in np.unique([fcal, bcal, gcal]): 
    applycal(vis = splitmsfile,
             field = field,
             gaintable = [splitmsfile+'.bandpass.cal', 
                          splitmsfile+'.phase_int.cal', 
                          splitmsfile+'.flux.cal'],
             gainfield = [bcal, field, field],
             interp = ['nearest', 'linear', 'linear'],
             calwt = True,
             flagbackup = False)

applycal(vis = splitmsfile,
         field = science_field,
         gaintable = [splitmsfile+'.bandpass.cal', 
                      splitmsfile+'.phase_scan.cal',
                      splitmsfile+'.flux.cal'],
         gainfield = [bcal, gcal, gcal],
         interp = ['nearest', 'linear', 'linear'],
         calwt = True,
         flagbackup = False)

# split the calibrated data
calmsfile = splitmsfile + '.calibrated'
os.system('rm -rf {}'.format(calmsfile))
split(vis=splitmsfile, outputvis=calmsfile, datacolumn='corrected')


# tclean and self-calibration
os.system('rm -rf tclean/cycle0.*')
tclean(vis=calmsfile,
       imagename='./tclean/cycle0',
       field='',
       spw='',
       specmode='mfs',
       deconvolver='hogbom',
       nterms=1,
       gridder='standard',
       imsize=320,
       cell='0.1arcsec',
       pblimit=0.1,
       weighting='natural',
       threshold='0mJy',
       niter=50000,
       interactive=True,
       savemodel='modelcolumn')


os.system("rm -rf cycle1_phase.cal")
gaincal(vis=calmsfile,
        caltable="cycle1_phase.cal",
        field="",
        solint="2min",
        calmode="p",
        refant=myrefant,
        gaintype="G")

applycal(vis=calmsfile,
         field="",
         gaintable=["cycle1_phase.cal"],
         interp="linear")

os.system("rm -rf {0} {0}.flagversions".format(calmsfile+'.selfcal'))
split(vis=calmsfile,
      outputvis=calmsfile+'.selfcal',
      datacolumn="corrected")

# make the cycle 1 image
os.system('rm -rf tclean/cycle1.*')
tclean(vis=calmsfile,
       imagename='./tclean/cycle1',
       field='',
       spw='',
       specmode='mfs',
       deconvolver='hogbom',
       nterms=1,
       gridder='standard',
       imsize=320,
       cell='0.1arcsec',
       pblimit=0.1,
       weighting='natural',
       threshold='0mJy',
       niter=50000,
       interactive=True,
       savemodel='modelcolumn')


# subtracting the point source
uvmodelfit(vis="J1924-2914.ms.cal.selfcal", niter=5, comptype="P", sourcepar=[1.0, 0.0, 0.0], outfile="J1924-2914.cl")
ft(vis="J1924-2914.ms.cal.selfcal", complist="J1924-2914.cl")
uvsub(vis="J1924-2914.ms.cal.selfcal",reverse=False)

# make the dirty image of the point source subtracted measurement
os.system('rm -rf tclean/J1924-2914.ms.cal.selfcal.pointsub.cont.auto.*')
tclean(vis=calmsfile,
       imagename='./tclean/J1924-2914.ms.cal.selfcal.pointsub.cont.auto',
       field='',
       spw='',
       specmode='mfs',
       deconvolver='hogbom',
       nterms=1,
       gridder='standard',
       imsize=320,
       cell='0.1arcsec',
       pblimit=0.1,
       weighting='natural',
       niter=0)

