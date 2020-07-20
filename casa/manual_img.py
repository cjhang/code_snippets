# Simple manual imaging script for VLA data
#
# Author: Jianhang Chen
# Email: cjhastro@gmail.com
# History:
#   2020.06.25 First release after multiple testings

# The reference clean program and notes

print('\nstart imaging....\n')

calfile = ''
myspw = '' 
myimsize = 500
mycell = '7arcsec'
myrestfreq = '1.420405752GHz'
myvstart = ''
mynchan = -1
mywidth = ''
allcal = ['0', '1']
target = '2'

imagedir = './tclean'
os.system('mkdir {}'.format(imagedir))

filename = os.path.basename(calfile)
basename = filename.split('.')[0]

#######################################################
#              Imaging the calibrators
#######################################################
if True:
    # os.system('cp {} .'.format(obs))
    # obs_local.append()
    # os.system('mkdir {}'.format(basename))
    
    # for gain calibrator
    for cal_field in allcal:
        mythreshold = '20mJy'
        
        split_file = imagedir +'/'+ basename + '_field{}.ms'.format(cal_field)
        split(vis=calfile, outputvis=split_file, field=cal_field, datacolumn='corrected')
        
        myimagename = split_file + '.cube.auto'
        rmtables(tablenames=myimagename + '.*')
        tclean(vis=split_file, spw=myspw,
               selectdata=True, datacolumn="data",
               imagename=myimagename,
               imsize=myimsize, cell=mycell, 
               restfreq=myrestfreq, phasecenter="", 
               specmode="cube", outframe="LSRK", veltype="optical", 
               nchan=mynchan, start=myvstart, width=mywidth,
               perchanweightdensity=True, restoringbeam="common",
               gridder="standard", pblimit=-0.0001,
               # weighting='natural',
               weighting="briggs", robust=1.5,
               niter=40000, gain=0.1, threshold=mythreshold,
               # usemask='user',
               usemask="auto-multithresh",
               interactive=False,
               savemodel="none",)



#######################################################
#              Imaging the calibrators
#######################################################

## Continuum subtraction for emission lines
# myfitspw = ''
# default(uvcontsub)
# uvcontsub(vis=calfile, fitspw=myfitspw, excludechans=True, want_cont=True)


# imaging the science target
split_file = imagedir +'/'+ basename + '_target.ms'
split(vis=calfile, outputvis=split_file, field=target, datacolumn='corrected')

if False:
    # mfs
    myimagename = split_file + basename+'mfs'
    rmtables(tablenames=myimagename + '.*')
    tclean(vis=split_file, spw=myspw, imagename=myimagename,
           imsize=myimsize, cell=mycell, specmode='mfs',
           weighting='briggs', robust=1.5, pblimit=-0.0001,
           niter=1000, interactive=False)

myimagename = split_file + '.cube.auto'
mythreshold = '3mJy'
rmtables(tablenames=myimagename + '.*')
tclean(vis=split_file, spw=myspw,
       selectdata=True, datacolumn="data",
       imagename=myimagename,
       imsize=myimsize, cell=mycell, 
       restfreq=myrestfreq, phasecenter="", 
       specmode="cube", outframe="LSRK", veltype="optical", 
       nchan=mynchan, start=myvstart, width=mywidth,
       perchanweightdensity=True, restoringbeam="common",
       gridder="standard", pblimit=-0.0001,
       # weighting='natural',
       weighting="briggs", robust=1.5,
       niter=40000, gain=0.1, threshold=mythreshold,
       usemask='user',
       # usemask="auto-multithresh",
       interactive=True,
       savemodel="none",)
# check modelcolumn is generated
#niter=0; calcres=False; calcpsf=False

# Moments
immoments(imagename=myimagename, moments=[0], 
          chans='11~40',
          box = "",
          mask = "target_cube.image > 0.045",
          outfile=myimagename+'.mom0')

immoments(imagename=myimagename, moments=[1],
          chans='11~40', 
          box = "",
          mask = "target_cube.image > 0.045",
          excludepix=[-100,0.01],
          outfile=myimagename+'.mom1')


#######################################################
#               Self-calibration
#######################################################
myimagename = 'self_cal1'
tclean(vis=calfile, spw=myspw, imagename=myimagename,
       imsize=myimsize, cell=mycell, specmode='mfs',
       weighting='briggs', robust=0, pblimit=-0.0001,
       niter=1000, interactive=False, savemodel='modelcolumn')

tclean(vis=calfile, spw=myspw, 
       imagename=myimagename,
       imsize=myimsize, cell=mycell, specmode='cube',
       start=myvstart, nchan=mynchan,width=mywidth,
       outframe='LSRK', restfreq=myrestfreq, veltype='optical',
       perchanweightdensity=True, pblimit=-0.0001, weighting='briggs', 
       robust=0.5, niter=10000, threshold=mythreshold, 
       interactive=False, savemodel='modelcolumn')

