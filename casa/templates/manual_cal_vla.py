# Simple manual calibration script for EVLA data
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
# History:
#   2020.04.11 First release after multiple testings
#   2020.06.01 Fixed several bugs ready for general usage
#   2020.06.04 Move the Prior Calibration after the Prior Flagging
#   2020.06.23 Introduce 'prior_caltable' for flexibility

#######################################################
#                  Data Prepare
#######################################################

# reduce the size
# split out the emission line spw
split(vis='', outputvis='', spw='', datacolumn='data')

# define the global variables
msfile = ''
fcal = '0'
bcal = '0'
gcal = '1'
allcal = fcal +','+ bcal +','+ gcal
target = '2'

myspw = ''
spw_bpphase =''        # the spw window with small variation
myrefant = '1'      # reference antenna
mysolint = 'int'       # integration time
myminsnr = 3.0  # minimal snr requirement for solutions
mycalwt = False        # calculate the weight, must be false for EVLA


cal_antpos = False  # update the antenna position online
cal_gceff = False
cal_opacity = False

# optional
# clearcal(vis=msfile)

## Validation checking
try:
    import plot_utils
    plot_results = True
except:
    plot_results = False

#######################################################
#                  Data Pre-inspection
#######################################################
# listobs
default(listobs)
listobs(vis=msfile, listfile=msfile+'.listobs.txt')

#######################################################
#                  Prior Flagging
#######################################################
print("start prior flagging...")
flagdata(vis=msfile, autocorr=True, flagbackup=False)
flagdata(vis=msfile, mode='shadow', flagbackup=False)
#> remove zero data
flagdata(vis=msfile, mode='clip', clipzeros=True, flagbackup=False)
#> remove the first 5s and the end 5s data (optional, see the amp_vs_time)
# flagdata(vis=msfile, mode='quack', quackinterval=3.0, quackmode='beg', flagbackup=False)
# flagdata(vis=msfile, mode='quack', quackinterval=3.0, quackmode='endb', flagbackup=False)
#> remove edge channels, 5% channels can be enough
# flagdata(vis=msfile, mode='manual', spw='*:0~100;900~1000',flagbackup=False)
#> saving prior flags 
flagmanager(vis=msfile, mode='save', versionname='Prior')

if plot_results:
    plot_utils.check_info(vis=msfile, spw='', bcal_field=bcal, gcal_field=gcal, target_field=target, refant=myrefant, plotdir='plots/Prior')


if False: #tfcrop RFI flagging
    pass
    #>> Testing
    flagdata(vis='vis', mode='tfcrop', spw='0', datacolumn='data', 
             action='calculate', antenna='ea01&ea02', display='both', 
             ntime='scan', combinescans=False, extendflags=False,
             timedevscale=5.0, freqdevscale=5.0, flagbackup=False)
    #>> Applying
    flagdata(vis=msfile, mode='tfcrop', spw='', field='', antenna='',
         datacolumn='data', action='apply', display='none', 
         ntime='scan', combinescans=False, extendflags=False,
         timedevscale=5.0, freqdevscale=5.0, flagbackup=False)
    flagmanager(vis=msfile, mode='save', versionname='AfterTFcrop')

if True:
    pass
    ## Additional manual flagging
    try:
        execfile('flag.py')
    except:
        pass

    # flagmanager(vis=msfile, mode='save', versionname='ManualFlag')


#######################################################
#                  Prior Calibration
#######################################################
prior_caltable = []
prior_calfield = []

##Atenna Poistion
if cal_antpos:
    # Correction for antenna position, automatic fetch from online database
    os.system('rm -rf antpos.cal')
    gencal(vis=msfile, caltable='antpos.cal', caltype='antpos', antenna='')
    if os.path.isdir('antpos.cal')
        prior_caltable.append('antpos.cal')
        prior_calfield.append('')

## Antenna efficiency and Gain curve (VLA only)
if cal_gceff:
    os.system('rm -rf gaincurve.cal')
    gencal(vis=msfile, caltable='gaincurve.cal', caltype='gceff')
    prior_caltable.append('gaincurve.cal')
    prior_calfield.append('')

### Opacity correction 
##> only for high frequency (e.g., Ku, K, Ka, and Q band)
if cal_opacity:
    myTau = plotweather(vis=msfile, doPlot=True) #it will generate the weather plot
    gencal(vis=msfile, caltable='opacity.cal', caltype='opac', parameter=myTau)
    prior_caltable.append('opacity.cal')
    prior_calfield.append('')
# Apply the Prior calibration if there are
for cal_field in allcal.split(','):
    applycal(vis=msfile, field=cal_field,
         gaintable=prior_caltable,
         gainfield=prior_calfield,  
         calwt=mycalwt, applymode='calflagstrict', flagbackup=False)
# split out the data after prior calibration
msfile2 = os.path.basename(msfile)[:-3]+'.afterprior.ms'
split(vis=msfile, outputvis=msfile2)


#######################################################
#             Set Flux Model
#######################################################

### set flux density
##> list all the avaible model
# setjy(vis=msfile ,listmodels=True)
# for resolved calibrators, one should specify the model
setjy(vis=msfile2, field=fcal) #, spw='0',scalebychan=True, model='3C286_L.im')


#######################################################
#             Bandpass and Gain Calibration
#######################################################
print("\n==============> Start Calibration <=============\n")
print("\n==============> Generating Bandpass Calibration <=============\n")
# delay calibration
os.system('rm -rf delays.cal')
gaincal(vis=msfile2, caltable='delays.cal', field=bcal, refant=myrefant, 
        gaintype='K', solint='inf', combine='scan', minsnr=myminsnr, 
        gaintable='')

# integration bandpass calibration
os.system('rm -rf bpphase.gcal')
default(gaincal)
gaincal(vis=msfile2, caltable="bpphase.gcal", field=bcal, spw=spw_bpphase, 
        solint=mysolint, refant=myrefant, minsnr=myminsnr, gaintype='G', calmode="p",
        gaintable='delays.cal')

# bandpass calinration
os.system('rm -rf bandpass.bcal')
default(bandpass)
bandpass(vis=msfile2, caltable='bandpass.bcal', field=bcal, spw='', refant=myrefant, 
         combine='scan', solint='inf', bandtype='B', minsnr=myminsnr,
         gaintable=['bpphase.gcal','delays.cal'])

# testing the bandpass calibration, applying the calibration to bandpass calibrator
# applycal(vis=msfile2, field=bcal, calwt=False,
        # gaintable=['gaincurve.cal', 'delays.cal', 'bandpass.bcal'],
        # gainfield=['' ,bcal, bcal])


print("\n==============> Generating Gain Calibration <=============\n")
# phase calibration for quick time variation
os.system('rm -rf phase_int.gcal')
default(gaincal)
gaincal(vis=msfile2, caltable='phase_int.gcal', field=allcal, refant=myrefant, 
        calmode='p', solint=mysolint, minsnr=myminsnr, spw='',
        gaintable=['bandpass.bcal','delays.cal'])

# phase calibration for long time variation
os.system('rm -rf phase_scan.gcal')
default(gaincal)
gaincal(vis=msfile2, caltable='phase_scan.gcal', field=allcal, refant=myrefant, 
        calmode='p', solint='inf', minsnr=myminsnr, spw='',
        gaintable=['bandpass.bcal', 'delays.cal'])

# amplitude calibration
os.system('rm -rf amp_scan.gcal')
default(gaincal)
gaincal(vis=msfile2, caltable='amp_scan.gcal', field=allcal, refant=myrefant, 
        calmode='ap', solint='inf', minsnr=myminsnr, spw='',
        gaintable=['bandpass.bcal','phase_int.gcal', 'delays.cal'])

# fluxscale
os.system('rm -rf flux.cal')
default(fluxscale)
myscale = fluxscale(vis=msfile2, caltable='amp_scan.gcal', fluxtable='flux.cal',
                    reference=fcal, incremental=True)
print(myscale)


print("\n==============> Applying the Calibration <=============\n")
# Applying the caltable to the calibrators
default(applycal)
for cal_field in allcal.split(','):
    applycal(vis=msfile2, field=cal_field,
         gaintable=['delays.cal', 'bandpass.bcal', 'phase_int.gcal', 
                    'amp_scan.gcal', 'flux.cal'],
         gainfield=[bcal, bcal, cal_field, cal_field, cal_field],  
         # interp = ['nearest', '', '', ''],
         calwt=mycalwt, applymode='calflagstrict', flagbackup=False)

if False: # RFI flagging by rflag
    pass
    #>> Testing
    flagdata(vis='vis', mode='rflag', spw='0', datacolumn='corrected', 
             action='calculate', antenna='ea01&ea02',
             ntime='scan', combinescans=False,
             display='both', timedevscale=5.0, freqdevscale=5.0, 
             flagbackup=False)
    #>> Applying, run after applying the calibration
    flagdata(vis=msfile2, mode='rflag', spw='', field=allcal, scan='', 
             datacolumn='corrected', action='apply', display='none',
             ntime='scan', combinescans=False,
             timedevscale=5.0, freqdevscale=3.0, flagbackup=False)
    flagmanager(vis=msfile2, mode='save', versionname='AfterRflag')

if plot_results:
    plot_utils.check_cal(vis=msfile2, spw='', field='0,1', refant='', plotdir='plots/antenna_cal')
    plot_utils.check_cal(vis=msfile2, spw='', field='0,1', refant='all', plotdir='plots/all_cal')

# apply the caltable to the target
default(applycal)
applycal(vis=msfile2, field=target,
         gaintable=['delays.cal', 'bandpass.bcal', 'phase_scan.gcal', 
                    'amp_scan.gcal','flux.cal'],
         gainfield=[bcal, bcal, gcal, gcal, gcal],  
         # interp = ['nearest', '', '', ''],
         calwt=mycalwt, applymode='calflagstrict', flagbackup=False)

flagmanager(vis=msfile2, mode='save', versionname='AfterApplycal')

if plot_results:
    plot_utils.check_cal(vis=msfile2, spw='', refant='all', field='2', plotdir='plots/target')

