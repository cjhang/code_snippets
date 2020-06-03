# The reference clean program and notes

calfile = ''
myspw = '' 
myimsize = 400
mycell = '7arcsec'
myrestfreq = '1.420405752GHz'
mythreshold = '5.0mJy'
myvstart = '1700km/s'
mynchan = 40
mywidth = ''


## Continuum subtraction for emission lines
# myfitspw = ''
# default(uvcontsub)
# uvcontsub(vis=calfile, fitspw=myfitspw, excludechans=True, want_cont=True)


#######################################################
#                     mfs
#######################################################
myimagename = calfile + '_mfs'
rmtables(tablenames=myimagename + '.*')
tclean(vis=calfile, spw=myspw, imagename=myimagename,
       imsize=myimsize, cell=mycell, specmode='mfs',
       weighting='briggs', robust=0, pblimit=-0.0001,
       niter=1000, interactive=False)


#######################################################
#                   datacube
#######################################################
myimagename = calfile + '_cube'
rmtables(tablenames=myimagename + '.*')
#> First run for dirty image 
#>> to determine the cell and threshold
tclean(vis=calfile, spw=myspw,
       selectdata=True, datacolumn="data",
       imagename=myimagename+'.dirty',
       imsize=myimsize, cell=mycell, 
       restfreq=myrestfreq, phasecenter="", 
       specmode="cube", outframe="LSRK", veltype="optical", 
       nchan=mynchan, start=myvstart, width=mywidth,
       perchanweightdensity=True, restoringbeam="common",
       gridder="standard", pblimit=-0.0001,
       weighting='natural',
       # weighting="briggs", robust=1.5,
       niter=0,)

#> start the full clean
tclean(vis=calfile, spw=myspw,
       selectdata=True, datacolumn="data",
       imagename=myimagename,
       imsize=myimsize, cell=mycell, 
       restfreq=myrestfreq, phasecenter="", 
       specmode="cube", outframe="LSRK", veltype="optical", 
       nchan=mynchan, start=myvstart, width=mywidth,
       perchanweightdensity=True, restoringbeam="common",
       gridder="standard", pblimit=-0.0001,
       weighting='natural',
       # weighting="briggs", robust=1.5,
       niter=10000, gain=0.1, threshold=mythreshold,
       # usemask='user',
       usemask="auto-multithresh",
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

