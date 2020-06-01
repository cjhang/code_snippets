msfile = './Dconfig_sb833.spw3.ms.calibrated'
myimagename = "tclean/Dconfig_sb833.spw3.briggs.cube"

tclean(vis=msfile,
        field="", spw="", datacolumn="data",
        imagename=myimagename,
        imsize=400, cell="7arcsec",
        specmode="cube", nchan=40, start="1700km/s",
        width="", outframe="LSRK", veltype="optical",
        restfreq="1.420405752GHz",
        perchanweightdensity=True, gridder="standard",
        pblimit=0.2, restoringbeam='common',
        weighting="briggs",robust=0.5,
        niter=10000, gain=0.1, threshold="3mJy",
        interactive=True, usemask="auto-multithresh", pbmask=0.2,
        savemodel="none")

