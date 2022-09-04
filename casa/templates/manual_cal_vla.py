# Simple manual calibration script for E/J-VLA data
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
vis_raw = ''
listobs(vis_raw, listfile=os.path.basename(vis_raw)+'.listobs.txt')
basename = '' # basename for all the following analysis: shortname for msfile, calibration tables
msfile = basename+'.ms' 
split(vis=vis_raw, outputvis=msfile, spw='0,1', datacolumn='data', 
      correlation='RR,LL', field='1,3,4')
listobs(vis=msfile, listfile=msfile+'.listobs.txt')

# define the global variables
fcal = '0'
bcal = '0'
gcal = '1'
calibrators = '0,1'#fcal +','+ bcal +','+ gcal
target = '2'

myspw = '0,1'
myrefant = 'ea02'      # reference antenna
myreffreq = '36GHz'
myminsnr = 2.0  # minimal snr requirement for solutions
mycalwt = False        # calculate the weight, must be false for EVLA

mysolint = '10s'       # integration time
spw4bpphase = '*:10~50' # the channls with small variation in freq
                 # used for derive better bandpass solution 

cal_antpos = True  # update the antenna position online
cal_gceff = True   # important for high frequency band
cal_opacity = True # useful for high frequency, e.g. K, Ku, Ka, Q bands
cal_tec = False    # important for low frequency, e.g P band 

# optional
# clearcal(vis=msfile)

## Validation checking
try:
    import plot_utils
    plot_results = True
except:
    plot_results = False


#######################################################
#                  Prior Flagging
#######################################################
print("start prior flagging...")
flagdata(vis=msfile, autocorr=True, flagbackup=False)
flagdata(vis=msfile, mode='shadow', flagbackup=False)
#> remove zero data
flagdata(vis=msfile, mode='clip', clipzeros=True, flagbackup=False)
#flagdata(vis=msfile, intent='*POINTING*,*FOCUS*,*ATMOSPHERE*,*SIDEBAND_RATIO*, *UNKNOWN*, *SYSTEM_CONFIGURATION*, *UNSPECIFIED#UNSPECIFIED*')
#> remove the first 5s and the end 5s data (optional, see the amp_vs_time)
flagdata(vis=msfile, mode='quack', quackinterval=3.0, quackmode='beg', flagbackup=False)
flagdata(vis=msfile, mode='quack', quackinterval=2.0, quackmode='endb', flagbackup=False)
#> remove edge channels, 5% channels can be enough
flagdata(vis=msfile, mode='manual', spw='*:0~3;60~63',flagbackup=False)
#> saving prior flags 
flagmanager(vis=msfile, mode='save', versionname='Prior')

if plot_results:
    # to get general info about the quality of the data (only calibrators)
    plot_utils.check_info(vis=msfile, spw=myspw, refant=myrefant,
                          show_fields=True, fields=calibrators, plotdir='plots/rawdata_info')
    # check the staibility of baselines with freq and time, useful to setup the spw4bpphase and solint 
    plot_utils.check_cal(vis=msfile, refant=myrefant, ydatacolumn='data', 
                         field=calibrators, spw=myspw, plotdir='plots/rawdata_refant')

#tfcrop RFI flagging, 
if False:
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

## Additional manual flagging
if True:
    # flagdata(vis=msfile, mode='manual', antenna='ea04', correlation='RR', flagbackup=False)
    try: # another way to make flagging
        execfile('flag.py')
    except:
        pass
    flagmanager(vis=msfile, mode='save', versionname='ManualFlag')
    plot_utils.check_cal(vis=msfile, refant=myrefant, ydatacolumn='data', 
                         field=calibrators, spw=myspw, plotdir='plots/rawdata_refant_flagged')
    plot_utils.check_cal(vis=msfile, refant='', ydatacolumn='data', yaxis=['amplitude'],
                         field=calibrators, spw=myspw, plotdir='plots/rawdata_antenna_flagged')

#######################################################
#                  Prior Calibration
#######################################################
prior_caltable = []
prior_calfield = []

##Atenna Poistion
if cal_antpos:
    # Correction for antenna position, automatic fetch from online database
    antpos_cal = f'{basename}_antpos.cal'
    rmtables(antpos_cal)
    gencal(vis=msfile, caltable=antpos_cal, caltype='antpos', antenna='')
    if os.path.isdir(antpos_cal):
        prior_caltable.append(antpos_cal)
        prior_calfield.append('')

## Antenna efficiency and Gain curve
if cal_gceff:
    gceff_cal = f'{basename}_gaincurve.cal'
    rmtables(gceff_cal)
    gencal(vis=msfile, caltable=gceff_cal, caltype='gceff')
    prior_caltable.append(gceff_cal)
    prior_calfield.append('')

## delay due to Total Electron Content (TEC), inversely proportional to the square of the frequency
## important for low-frequency p band observation, it needs data from IGS website
if cal_tec:
    from recipes import tec_maps
    tec_image, tec_rms_image, plotname = tec_maps.create(vis='3C129_pband.ms', doplot=True)
    #NOTE: If you are using CASA 5.0.0, or earlier only tec_image, tec_rms_image will be returned
    tec_cal = f'{basename}_tec.cal'
    gencal(vis=msfile, caltable=tec_cal, caltype='tecim', infile=tec_image)
    prior_caltable.append(tec_cal)
    prior_calfield.append('')

## Opacity correction 
##> only for high frequency (e.g., Ku, K, Ka, and Q band)
if cal_opacity:
    myTau = plotweather(vis=msfile, doPlot=True) #it will generate the weather plot
    
    opacity_cal = f'{basename}_opacity.cal'
    rmtables(opacity_cal)
    gencal(vis=msfile, caltable=opacity_cal, caltype='opac', parameter=myTau, spw=myspw)
    prior_caltable.append(opacity_cal)
    prior_calfield.append('')

# Apply the Prior calibration if there are
for cal_field in calibrators.split(','):
    applycal(vis=msfile, field=cal_field,
         gaintable=prior_caltable,
         gainfield=prior_calfield,  
         calwt=mycalwt, flagbackup=False)
# split out the data after prior calibration
basename_aprior = basename+'.aprior'
msfile_aprior = basename_aprior + '.ms'
rmtables(msfile_aprior)
split(vis=msfile, outputvis=msfile_aprior)


#######################################################
#             Set Flux Model
#######################################################

### set flux density
##> list all the avaible model
# setjy(vis=msfile ,listmodels=True)
# for resolved calibrators, one should specify the model
setjy(vis=msfile_aprior, field=fcal, spw=myspw, scalebychan=True, )#model='3C286_K.im', reffreq=myreffreq)


#######################################################
#             Bandpass and Gain Calibration
#######################################################
print("\n==============> Start Calibration <=============\n")
print("\n==============> Generating Bandpass Calibration <=============\n")
# delay calibration
delays_cal = f'{basename_aprior}_delays.gcal'
rmtables(delays_cal)
gaincal(vis=msfile_aprior, caltable=delays_cal, field=bcal, refant=myrefant, 
        gaintype='K', solint='inf', combine='scan', minsnr=myminsnr, 
        gaintable='')

# integration bandpass calibration
bpphase_gcal = f'{basename_aprior}_bpphase.gcal'
rmtables('bpphase.gcal')
gaincal(vis=msfile_aprior, caltable=bpphase_gcal, field=bcal, spw=spw4bpphase, 
        solint=mysolint, refant=myrefant, minsnr=myminsnr, gaintype='G', calmode="p",
        gaintable=delays_cal)

# bandpass calinration
bandpass_bcal = f'{basename_aprior}_bandpass.bcal'
rmtables(bandpass_bcal)
bandpass(vis=msfile_aprior, caltable=bandpass_bcal, field=bcal, spw=myspw, refant=myrefant, 
         combine='scan', solint='inf', bandtype='B', minsnr=myminsnr,
         gaintable=[delays_cal, bpphase_gcal])

# testing the bandpass calibration, applying the calibration to bandpass calibrator
# applycal(vis=msfile_aprior, field=bcal, calwt=False,
        # gaintable=[gaincurve_cal, delays_cal, bandpass_bcal],
        # gainfield=['' ,bcal, bcal])


print("\n==============> Generating Gain Calibration <=============\n")
# phase calibration for quick time variation
phase_int_gcal = f'{basename_aprior}_phase_int.gcal'
rmtables(phase_int_gcal)
gaincal(vis=msfile_aprior, caltable=phase_int_gcal, field=calibrators, refant=myrefant, 
        calmode='p', solint=mysolint, minsnr=myminsnr, spw=myspw,
        gaintable=[delays_cal, bandpass_bcal])

# phase calibration for long time variation
phase_scan_gcal = f'{basename_aprior}_phase_scan.gcal'
rmtables(phase_scan_gcal)
gaincal(vis=msfile_aprior, caltable=phase_scan_gcal, field=calibrators, refant=myrefant, 
        calmode='p', solint='inf', minsnr=myminsnr, spw=myspw,
        gaintable=[delays_cal, bandpass_bcal])

# amplitude calibration
amp_scan_gcal = f'{basename_aprior}_amp_scan.gcal'
rmtables(amp_scan_gcal)
gaincal(vis=msfile_aprior, caltable=amp_scan_gcal, field=calibrators, refant=myrefant, 
        calmode='ap', solint='inf', minsnr=myminsnr, spw=myspw,
        gaintable=[delays_cal, bandpass_bcal, phase_int_gcal])

# fluxscale
flux_cal = f'{basename_aprior}_flux.cal'
rmtables(flux_cal)
myscale = fluxscale(vis=msfile_aprior, caltable=amp_scan_gcal, fluxtable=flux_cal,
                    reference=fcal, incremental=True)
with open(f'{basename_aprior}_flux.txt','w+') as fp:
    print(myscale, file=fp)


print("\n==============> Applying the Calibration <=============\n")
# Applying the caltable to the calibrators
for cal_field in calibrators.split(','):
    applycal(vis=msfile_aprior, field=cal_field,
         gaintable=[delays_cal, bandpass_bcal, phase_int_gcal, 
                    amp_scan_gcal, flux_cal],
         gainfield=[bcal, bcal, cal_field, cal_field, cal_field],  
         # interp = ['nearest', '', '', ''],
         calwt=mycalwt, flagbackup=False)

if False: # RFI flagging by rflag
    pass
    #>> Testing
    flagdata(vis='vis', mode='rflag', spw='0', datacolumn='corrected', 
             action='calculate', antenna='ea01&ea02',
             ntime='scan', combinescans=False,
             display='both', timedevscale=5.0, freqdevscale=5.0, 
             flagbackup=False)
    #>> Applying, run after applying the calibration
    flagdata(vis=msfile_aprior, mode='rflag', spw='', field=calibrators, scan='', 
             datacolumn='corrected', action='apply', display='none',
             ntime='scan', combinescans=False,
             timedevscale=5.0, freqdevscale=3.0, flagbackup=False)
    flagmanager(vis=msfile_aprior, mode='save', versionname='AfterRflag')

if plot_results:
    plot_utils.check_cal(vis=msfile_aprior, spw=myspw, field=calibrators, refant='', plotdir='plots/antenna_cal')
    plot_utils.check_cal(vis=msfile_aprior, spw=myspw, field=calibrators, refant='all', plotdir='plots/all_cal')

# apply the caltable to the target
applycal(vis=msfile_aprior, field=target,
         gaintable=[delays_cal, bandpass_bcal, phase_scan_gcal, 
                    amp_scan_gcal,flux_cal],
         gainfield=[bcal, bcal, gcal, gcal, gcal],  
         # interp = ['nearest', '', '', ''],
         calwt=mycalwt, applymode='calflagstrict', flagbackup=False)

flagmanager(vis=msfile_aprior, mode='save', versionname='AfterApplycal')


# flag the science target
if False:
    flagmanager(vis=msfile_aprior, mode='save', versionname='TargetManualFlag')

if plot_results:
    plot_utils.check_cal(vis=msfile_aprior, spw=myspw, refant=myrefant, yaxis=['amplitude'], field=target, plotdir='plots/target_refant')
    plot_utils.check_cal(vis=msfile_aprior, spw=myspw, refant='', yaxis=['amplitude'], field=target, plotdir='plots/target_antenna')
    plot_utils.check_cal(vis=msfile_aprior, spw=myspw, refant='all', yaxis=['amplitude'], field=target, plotdir='plots/target_all')

msfile_target = f'{basename}_target.ms'
rmtables(msfile_target)
split(vis=msfile_aprior, field=target, outputvis=msfile_target, datacolumn='corrected')


# simple code for quick imaging
# FoV = 42'/f ; f in GHz
cell = '0.4arcsec' # FoV/n : D = 4, A:B:C:D=3:1
imsize = 150 
for field in calibrators.split(','):
    tclean(vis=msfile_aprior, field=field,
           imagename=f'images/{basename}_field{field}',
           cell=cell, imsize=imsize, niter=100)
tclean(vis=msfile_target, imagename=f'images/{basename}_target_calibrated', datacolumn='data',
       cell=cell, imsize=imsize, specmode='cube', 
       restfreq=myreffreq,
       start='35.94GHz', width='4MHz', nchan=40, niter=10)

