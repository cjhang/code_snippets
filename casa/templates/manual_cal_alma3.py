# Simple manual calibration script for EVLA data
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
# History:
#   2019.09.30 First release for learning casa
#   2020.04.11 Update for reduce Centaurus_A
#   2023.06.23 Test with fullpol observation


################ Basic Information ###############
# CALIBRATE_BANDPASS: 
# CALIBRATE_GAIN: 
# CALIBRATE_FLUX:
# OBSERVE_TARGET: 
#
# Notes:
#

#######################################################
#                  Data Prepare
#######################################################

# The program to maunual calibrate the calibrator data
import numpy as np
import sys
sys.path.append('/home/jchen/work/projects/code_snippets/python_tools')

asdmfile = '/home/jchen/hdd/dust_pol/spt0346-52/2021.1.00458.S.MOUS.uid___A001_X1590_X22eb.SBNAME.SPT0346-_a_07_TM1/sg_ouss_id/group_ouss_id/member_ouss_id/raw/uid___A002_Xfd6eb8_X2845.asdm.sdm'
msfile = 'uid_X2845.ms'
if not os.path.isdir(msfile):
    importasdm(asdm=asdmfile, vis=msfile)
listobs(vis=msfile, listfile=msfile+'.listobs.txt')


myrefant = 'DA61'
myreffreq = ''
fcal = 'J0519-4546'
bcal = 'J0519-4546'
gcal = 'J0253-5441'
pcal = 'J0423-0120'
allcal = ','.join(np.unique([fcal, bcal, gcal, pcal]))
# check if all the targets have Tsys observation
fields_tied_Tsys = [] # [field1, field2], the first one has Tsys info,check CALIBRATE_ATMOSPHERE
# tsys_spw = '5,7,9,11'
# check if all the targets have wvr observation
fields_tied_wvr = [] # [field1, field2], the first one has wvr info, check CALIBRATE_WVR

#> mapping tsys spw to science spw, check CALIBRATE_ATMOSPHERE
# from recipes.almahelpers import tsysspwmap # for casa before version 6
# from casarecipes.almahelpers import tsysspwmap
# tsysmap = tsysspwmap(vis=msfile, tsystable=msfile+'.tsys')
# example:
# >> tsysmap = list(range(0,24)) # 24 is total number of spw
# >> tsysmap[17:25] = list(range(9,17)) # map science spw to tsys spw
tsysmap=[]

science_spw = '13,15,17,19'
science_field = 'SPT0346-52'
spw_bpphase ='*:5~60' # the spw window to get the time variation for bandpass calibrator
mysolint = 'int'    # integration time
mycalwt = True     # calculate the weight, must be false for EVLA

# for Bandpass and gain callibration
splitmsfile = msfile+'.split' # the msfile after priori calibration
splited_spw = '0,1,2,3'

try:
    import plot_utils
    plot_results = True
except:
    plot_results = False

################
######### start 

# Flagging the useless data
# flagdata(vis=msfile, mode='unflag', flagbackup=False)
flagdata(vis=msfile, mode='manual', autocorr=True, flagbackup=False)
flagdata(vis=msfile, mode='shadow', flagbackup=False)

# Flag edge channel
flagdata(vis=msfile, spw='13:0~6;59~63;15:0~6;59~63,17:0~6;59~63,19:0~6;59~63')

flagmanager(vis=msfile, mode='save', versionname='Prior')

if False:
    #>> specific flagging
    # flagdata(vis='uid___A002_Xdb7ab7_X11a58.ms.tsys', scan='15', antenna='DV02', flagbackup=False)
    flagmanager(vis=obs, mode='save', versionname='Manual_flag')


if False:
    # Fix the ASDM SYSCal table issue, data before 2015
    from recipes.almahelpers import fixsyscaltimes
    fixsyscaltimes(vis=msfile)
    
    if False: # No issue found
        es.fixForCSV2555(msfile)

    # Fix the 0,0 coordinates of planets (flux calibrator)
    if False:
        fixplanets(vis = msfile, field = fcal, fixuvw = True)

if plot_results:
        # to get general info about the quality of the data (only calibrators)
        plot_utils.check_info(vis=msfile, spw=science_spw, refant=myrefant, correlation='XX,YY',
                              show_fields=True, fields=allcal, plotdir='plots/rawdata_info')

#######################################################
#                  Prior Calibration
#######################################################
print("\n==============> Start Prior Calibration <=============\n")

##### Tsys Calibration
# It gives the first-order correction for atmospheric opacity as a funcction of time and freq
os.system("rm -rf {}.tsys".format(msfile))
gencal(vis=msfile, caltable=msfile+'.tsys', caltype='tsys')
# flag edge channel
flagdata(vis=msfile+'.tsys', spw='13:0~3,15:0~3,17:0~3,19:0~3')
flagmanager(vis=msfile, mode='save', versionname='Tsys')

if plot_results:
    plot_utils.check_tsys(tsystable=msfile+'.tsys', spws=science_spw, plotdir='./plots/tsys')


##### Water Vaper Calibration
os.system("rm -rf {}.wvr".format(msfile))
#> Cycle-0 up to Jan 2013, the offset should be set -1
wvrgcal(vis=msfile, caltable=msfile+'.wvr', toffset=0.0)
        #> tie phasecal and science target, share CALIBRATE_ATMOSPHERE 
        # tie = [','.join(fields_tied_wvr),], 
        # statsource = 'fieldname', # usually the bandpass
        # wvrflag=[], # add the antenna with bad WVR

##### Antenna position caliration
if False: # check QA2
    os.system('rm -rf {}.antpos'.format(msfile)) 
    gencal(vis=msfile,
           caltable='{}.antpos'.format(msfile),
           caltype='antpos',
           antenna='', # 'DV02,DV04,PM01'
           parameter=[],) # [0,0,0, 0,0,0, 0,0,0]
 

# Apply the Tsys and WVR calibration
for field in np.unique([fcal, bcal, gcal, science_field]):
    applycal(vis=msfile,
             field=field,
             spw=science_spw,
             gaintable=[msfile+'.tsys', msfile+'.wvr'],
             gainfield=[field, field],
             interp='linear',
             spwmap=[tsysmap,[]],
             calwt=True,
             flagbackup=False)
# for the tied field, without their own wvr
if fields_tied_Tsys != []:
    applycal(vis=msfile,
             field=fields_tied_Tsys[1],
             spw=science_spw,
             gaintable=[msfile+'.tsys', msfile+'.wvr'],
             gainfield=[fields_tied_Tsys[0], fields_tied_Tsys[1]],
             interp='linear',
             spwmap = [tsysmap,[]],
             calwt=True,
             flagbackup = False)


#> Check the calibrated data
if plot_results:
    # only test the calibration on the bandpass calibrator, and only amplitude
    plot_utils.check_cal(msfile, spw=science_spw, field=bcal, 
                         ydatacolumn='data', yaxis=['amplitude'],
                         correlation='XX,YY',
                         plotdir='plots/priorcal')
    plot_utils.check_cal(msfile, spw=science_spw, field=bcal, 
                         ydatacolumn='corrected', yaxis=['amplitude'],
                         correlation='XX,YY',
                         plotdir='plots/priorcal')

os.system('rm -rf {0} {0}.flagversions'.format(splitmsfile))
split(vis=msfile, field='', outputvis=splitmsfile,
      datacolumn='corrected', spw=science_spw)

flagmanager(vis=splitmsfile, mode='save',
            versionname = 'BeforeBPCal') # before bandpass and phase calibration


#######################################################
#             Bandpass and Gain Calibration
#######################################################

# additional flag after the prior calibration
if True:
    # flag useless data
    flagdata(vis=splitmsfile, mode='manual', flagbackup=False, 
             intent='*POINTING*,*SIDEBAND_RATIO*,*ATMOSPHERE*')

# gain cal in short time variation

# setjy(vis=splitmsfile, field=fcal,)
# set the flux manually
setjy(vis=splitmsfile, field=fcal, reffreq='343.5GHz', 
      standard='manual', fluxdensity=[0.98, 0, 0, 0])

# solve short time variation of bandpass calibrator
os.system('rm -rf {}.bpphase_int.cal'.format(splitmsfile)) 
gaincal(vis = splitmsfile,
        caltable = splitmsfile+'.bpphase_int.cal',
        field = bcal, # bandpass calibrator
        spw = spw_bpphase,
        minsnr = 2.0,
        solint = 'int',
        refant = myrefant,
        calmode = 'p')

# bandpass calibration
os.system('rm -rf {}.bandpass.cal'.format(splitmsfile)) 
bandpass(vis = splitmsfile,
         caltable = splitmsfile+'.bandpass.cal',
         field = bcal, # bandpass calibrator
         solint = 'inf',
         minsnr = 2.0,
         combine = '',
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
        gaintable = [splitmsfile+'.bandpass.cal', 
                     splitmsfile+'.phase_int.cal'])

# Flux calibration
os.system('rm -rf {}.flux.cal'.format(splitmsfile)) 
fluxscale(vis = splitmsfile,
          caltable = splitmsfile+'.amp.cal',
          fluxtable = splitmsfile+'.flux.cal',
          reference = fcal,
          transfer = '') 

# Apply the short time gain_cal and bandpass to the calibrator

for field in allcal.split(','): 
    applycal(vis = splitmsfile,
             field = field,
             gaintable = [splitmsfile+'.bandpass.cal', 
                          splitmsfile+'.phase_int.cal', 
                          splitmsfile+'.flux.cal'],
             gainfield = [bcal, field, field],
             interp = ['nearest', 'linear', 'linear'],
             calwt = True,
             flagbackup = False)
if plot_results:
    plot_utils.check_cal(splitmsfile, spw=splited_spw, field=allcal, 
                         ydatacolumn='corrected', yaxis=['amplitude','phase'], 
                         correlation='XX,YY',plotdir='plots/BPcal')

applycal(vis = splitmsfile,
         field = science_field,
         gaintable = [splitmsfile+'.bandpass.cal', 
                      splitmsfile+'.phase_scan.cal',
                      splitmsfile+'.flux.cal'],
         gainfield = [bcal, gcal, gcal],
         interp = ['nearest', 'linear', 'linear'],
         calwt = True,
         flagbackup = False)
if plot_results:
    plot_utils.check_cal(splitmsfile, spw=splited_spw, field=science_field, 
                         ydatacolumn='data', yaxis=['amplitude'],
                         correlation='XX,YY', plotdir='plots/science')
    plot_utils.check_cal(splitmsfile, spw=splited_spw, field=science_field, 
                         ydatacolumn='corrected', yaxis=['amplitude'], 
                         correlation='XX,YY', plotdir='plots/science')

# split the calibrated data
calmsfile = splitmsfile + '.calibrated'
os.system('rm -rf {}'.format(calmsfile))
split(vis=splitmsfile, outputvis=calmsfile, datacolumn='corrected')


if False:
    os.system('rm -rf tclean/cycle0.*')
    tclean(vis=calmsfile, imagename='./tclean/cycle0', field='',
           spw='', specmode='mfs', deconvolver='hogbom',
           imsize=320, cell='0.1arcsec',
           pblimit=0.2,
           weighting='natural',
           threshold='0mJy', niter=50000, interactive=True,
           
