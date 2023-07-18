import numpy as np
import scipy.special as spsc

arcsec2rad = np.pi/(180.*3600.)
rad2arcsec =3600.*180./np.pi
deg2rad = np.pi/180.
rad2deg = 180./np.pi




class GaussSource(object):
      """
      Adopted from visilens, from Justin Spilker
      Class to hold parameters of a circularly symmetric Gaussian light
      profile, where all the variable parameters are dictionaries.
      
      Example format of each parameter:
      x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
      initial/current value of x, x should not be a fixed parameter during fitting,
      and the value of x must be between xmin and xmax.
      
      Parameters:
      z
            Source redshift. Can be made up, as long as it's higher than
            the lens redshift.
      lensed
            True/False flag determining whether this object is actually lensed
            (in which case it gets run through the lensing equations) or not (in
            which case it's simply added to the model of the field without lensing).
            This also determines the convention for the source position coordinates,
            see below.
      x, y
            Position of the source in arcseconds. If lensed is True, this position
            is relative to the position of the lens (or the first lens in a list of
            lenses). If lensed is False, this position is relative to the field
            center (or (0,0) coordinates). +x is west (sorry not sorry), +y is north.
      flux
            Total integrated flux density of the source (ie, NOT peak pixel value), in
            units of Jy.
      width
            The Gaussian width (sigma) of the light profile, in arcseconds.
      """

      def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None,width=None):
            # Do some input handling.
            if not isinstance(xoff,dict):
                  xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(yoff,dict):
                  yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
            if not isinstance(flux,dict):
                  flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source
            if not isinstance(width,dict):
                  width = {'value':width,'fixed':False,'prior':[0.,2.]} # arcsec

            if not all(['value' in d for d in [xoff,yoff,flux,width]]): 
                  raise KeyError("All parameter dicts must contain the key 'value'.")

            if not 'fixed' in xoff: xoff['fixed'] = False
            if not 'fixed' in yoff: yoff['fixed'] = False
            if not 'fixed' in flux: flux['fixed'] = False  
            if not 'fixed' in width: width['fixed'] = False
            
            if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
            if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
            if not 'prior' in flux: flux['prior'] = [1e-5,1.]
            if not 'prior' in width: width['prior'] = [0.,2.]

            self.z = z
            self.lensed = lensed
            self.xoff = xoff
            self.yoff = yoff
            self.flux = flux
            self.width = width
 
class SersicSource(object):
    """
    Class to hold parameters of an elliptical Sersic light profile, ie
    I(x,y) = A * exp(-bn*((r/reff)^(1/n)-1)),
    where bn makes reff enclose half the light (varies with Sersic index), 
    and all the variable parameters are dictionaries. This profile is 
    parameterized by the major axis and axis ratio; you can get the half-light
    radius with r_eff = majax * sqrt(axisratio).
    
    Example format of each parameter:
    x = {'value':x0,'fixed':False,'prior':[xmin,xmax]}, where x0 is the
    initial/current value of x, x should not be a fixed parameter during fitting,
    and the value of x must be between xmin and xmax.
    
    Parameters:
    z
          Source redshift. Can be made up, as long as it's higher than
          the lens redshift.
    lensed
          True/False flag determining whether this object is actually lensed
          (in which case it gets run through the lensing equations) or not (in
          which case it's simply added to the model of the field without lensing).
          This also determines the convention for the source position coordinates,
          see below.
    x, y
          Position of the source in arcseconds. If lensed is True, this position
          is relative to the position of the lens (or the first lens in a list of
          lenses). If lensed is False, this position is relative to the field
          center (or (0,0) coordinates). +x is west (sorry not sorry), +y is north.
    flux
          Total integrated flux density of the source (ie, NOT peak pixel value), in
          units of Jy.
    majax
          The source major axis in arcseconds.
    index
          The Sersic profile index n (0.5 is ~Gaussian, 1 is ~an exponential disk, 4
          is a de Vaucoleurs profile). 
    axisratio
          The source minor/major axis ratio, varying from 1 (circularly symmetric) to
          0 (highly elongated).
    PA
          Source position angle. If lensed is True, this is in degrees CCW from the
          lens major axis (or first lens in a list of them). If lensed is False, this
          is in degrees east of north.
    """

    def __init__(self,z,lensed=True,xoff=None,yoff=None,flux=None,majax=None,\
                index=None,axisratio=None,PA=None):
          # Do some input handling.
          if not isinstance(xoff,dict):
                xoff = {'value':xoff,'fixed':False,'prior':[-10.,10.]}
          if not isinstance(yoff,dict):
                yoff = {'value':yoff,'fixed':False,'prior':[-10.,10.]}
          if not isinstance(flux,dict):
                flux = {'value':flux,'fixed':False,'prior':[1e-5,1.]} # 0.01 to 1Jy source
          if not isinstance(majax,dict):
                majax = {'value':majax,'fixed':False,'prior':[0.,2.]} # arcsec
          if not isinstance(index,dict):
                index = {'value':index,'fixed':False,'prior':[0.3,4.]}
          if not isinstance(axisratio,dict):
                axisratio = {'value':axisratio,'fixed':False,'prior':[0.01,1.]}
          if not isinstance(PA,dict):
                PA = {'value':PA,'fixed':False,'prior':[0.,180.]}

          if not all(['value' in d for d in [xoff,yoff,flux,majax,index,axisratio,PA]]): 
                raise KeyError("All parameter dicts must contain the key 'value'.")

          if not 'fixed' in xoff: xoff['fixed'] = False
          if not 'fixed' in yoff: yoff['fixed'] = False
          if not 'fixed' in flux: flux['fixed'] = False  
          if not 'fixed' in majax: majax['fixed'] = False
          if not 'fixed' in index: index['fixed'] = False
          if not 'fixed' in axisratio: axisratio['fixed'] = False
          if not 'fixed' in PA: PA['fixed'] = False
          
          if not 'prior' in xoff: xoff['prior'] = [-10.,10.]
          if not 'prior' in yoff: yoff['prior'] = [-10.,10.]
          if not 'prior' in flux: flux['prior'] = [1e-5,1.]
          if not 'prior' in majax: majax['prior'] = [0.,2.]
          if not 'prior' in index: index['prior'] = [1/3.,10]
          if not 'prior' in axisratio: axisratio['prior'] = [0.01,1.]
          if not 'prior' in PA: PA['prior'] = [0.,180.]

          self.z = z
          self.lensed = lensed
          self.xoff = xoff
          self.yoff = yoff
          self.flux = flux
          self.majax = majax
          self.index = index
          self.axisratio = axisratio
          self.PA = PA

def SourceProfile(xsource, ysource, source, lens):
    """
    Creates the source-plane profile of the given Source.

    Inputs:
    xsource,ysource:
          Source-plane coordinates, in arcsec, on which to
          calculate the luminosity profile of the source
    
    Source:
          Any supported source-plane object, e.g. a GaussSource
          object. The object will contain all the necessary
          parameters to create the profile.

    Lens:
          Any supported Lens object, e.g. an SIELens. We only need
          this because, in the case of single lenses, the source
          position is defined as offset from the lens centroid. If
          there is more than one lens, or if the source is unlensed,
          the source position is defined **relative to the field 
          center, aka (0,0) coordinates**.
          

    Returns:
    I:
          The luminosity profile of the given Source. Has same
          shape as xsource and ysource. Note: returned image has
          units of flux / arcsec^2 (or whatever the x,y units are),
          so to properly normalize, must multiply by pixel area. This
          isn't done here since the lensing means the pixels likely
          aren't on a uniform grid.
    """
    
    lens = list(np.array([lens]).flatten())

    # First case: a circular Gaussian source.
    if source.__class__.__name__=='GaussSource':
          sigma = source.width['value']
          amp   = source.flux['value']/(2.*np.pi*sigma**2.)
          if source.lensed:# and len(lens)==1:
                xs = source.xoff['value'] + lens[0].x['value']
                ys = source.yoff['value'] + lens[0].y['value']
          else:
                xs = source.xoff['value']
                ys = source.yoff['value']
          
          return amp * np.exp(-0.5 * (np.sqrt((xsource-xs)**2.+(ysource-ys)**2.)/sigma)**2.)

    elif source.__class__.__name__=='SersicSource':
          if source.lensed:# and len(lens)==1:
                xs = source.xoff['value'] + lens[0].x['value']
                ys = source.yoff['value'] + lens[0].y['value']
          else:
                xs = source.xoff['value']
                ys = source.yoff['value']
          PA, ar = source.PA['value']*deg2rad, source.axisratio['value']
          majax, index = source.majax['value'], source.index['value']
          dX = (xsource-xs)*np.cos(PA) + (ysource-ys)*np.sin(PA)
          dY = (-(xsource-xs)*np.sin(PA) + (ysource-ys)*np.cos(PA))/ar
          R = np.sqrt(dX**2. + dY**2.)
          
          # Calculate b_n, to make reff enclose half the light; this approx from Ciotti&Bertin99
          # This approximation good to 1 in 10^4 for n > 0.36; for smaller n it gets worse rapidly!!
          #bn = 2*index - 1./3. + 4./(405*index) + 46./(25515*index**2) + 131./(1148175*index**3) - 2194697./(30690717750*index**4)
          # Note, now just calculating directly because everyone's scipy
          # should be sufficiently modern.
          bn = spsc.gammaincinv(2. * index, 0.5)
          
          # Backing out from the integral to R=inf of a general sersic profile
          Ieff = source.flux['value'] * bn**(2*index) / (2*np.pi*majax**2 * ar * np.exp(bn) * index * spsc.gamma(2*index))
          
          return Ieff * np.exp(-bn*((R/majax)**(1./index)-1.))
    
    elif source.__class__.__name__=='PointSource':
          if source.lensed:# and len(lens)==1:
                #xs = source.xoff['value'] + lens[0].x['value']
                #ys = source.yoff['value'] + lens[0].y['value']
                return ValueError("Lensed point sources not working yet... try a"\
                 "gaussian with small width instead...")
          else:
                xs = source.xoff['value']
                ys = source.yoff['value']
                
          yloc = np.abs(xsource[0,:] - xs).argmin()
          xloc = np.abs(ysource[:,0] - ys).argmin()
          
          m = np.zeros(xsource.shape)
          m[xloc,yloc] += source.flux['value']/(xsource[0,1]-xsource[0,0])**2.
          
          return m
          
    
    else: raise ValueError("So far only GaussSource, SersicSource, and "\
          "PointSource objects supported...")


