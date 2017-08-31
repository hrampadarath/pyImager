"""
An implementation of the Cotton-Schwab clean for Aprojection using 
Bojan Nikolic clean.py from the ska github crocodile page:
https://github.com/SKA-ScienceDataProcessor/crocodile/tree/master/crocodile

Hayden Rampadarath 2016
hayden.rampadarath@manchester.ac.uk or haydenrampadarath@gmail.com
"""
import sys
import numpy as np
from astropy.modeling import models, fitting
import warnings
import GriddingFunctions as GF
import GriddingKernels as GK
import FITSmodules as FM
import Imagefuncs as IF
import matplotlib.pyplot as plt


LIGHT_SPEED = 299792458.         # Speed of light
deg2rad = np.pi/180.

def majorcycle(vis,HA,uvw,image_params,obs_params,files,hduList,Array):

    nminor  = image_params['niter'] #number of minor iterations
    nmajor  = image_params['nmajor'] # number of major iterations
    loopgain    = image_params['loopGain'] #loop gain
    ref_freq = obs_params['ref_freq']/1e6
    ch_width = obs_params['ch_width']
    thresh = image_params['thresh']
    fov = (image_params['imageSize'] * image_params['cellsize'])/3600.
    lam = LIGHT_SPEED/(ref_freq*1e6)
    over_sampling = image_params['over_sampling']
    kernel_support = image_params['kernel_support']

    """
    Major cycle clean: The main code
    
    vis - input visibilities (XX,YY)
    HA	- Hour angle values of the visibility (hrs)
    uvw	- UVWs of the visibilities (m) 
    image_params    - paramters specific to the imaging 
    obs_params	    - parameters related to the observation

    """
    vis_res = vis

    uvw2,visx = IF.sortw(uvw,vis[0])
    uvw2,visy = IF.sortw(uvw,vis[1])
    vis = np.array([visx,visy])


    # calculate the pswf

    pswf = GK.StaleyKernel(kernel_support,over_sampling)

    if image_params['kernel'] == 'awkernel':
        #generate unrotated mterms at the reference freq and store in memory.
        #note, this would not be valid for wide-band A-projection
        #if (image_params['kernel'] == 'awkernel') or image_params['kernel'] == 'askapsoft':
        Mterms,Mterms_ij = GF.MuellerTerms(image_params,ref_freq,ch_width)
        dty_image,psf_image = GF.awGrid(vis_res,HA,uvw,image_params,obs_params,Mterms,Mterms_ij)
        #obtain the clean beam


    elif image_params['kernel'] == 'wkernel':
        dty_image,psf_image = GF.wGrid(vis_res,uvw,image_params,obs_params,pswf)
        #obtain the clean beam
        clBeam, psf_fit = FM.cleanBeam(psf_image,image_params)
        FM.writetoFITS_scratch('dirtypsf_real.fits',psf_image.real/np.amax(psf_image),
                               fov,Array,hduList,psf_fit, imageType='dirtyimage')

    elif image_params['kernel'] == 'pswf':
        dty_image,psf_image = GF.Gridding(vis,uvw,image_params,obs_params,pswf)
        clBeam, psf_fit = FM.cleanBeam(psf_image,image_params)

    else:
        print 'Kernel not recognised .... aborting'
        sys.exit()
    
    ccLIST = np.zeros((nmajor,nminor))
    cc_image = 0
    for i in range(nmajor):

        print '---------Doing Major cycle %s -----------------' % str(i)

        imagename = files['wdir']+files['imagename']+'_nmaj'+ str(i)

        #FM.writetoFITS_scratch(imagename+'_dirtyimage_preclean.fits',np.abs(dty_image)/np.amax(np.abs(psf_image)),fov,Array,hduList,psf_fit,imageType='dirtyimage')

        ##2. obtain the clean components (cc) and residuals via Hogbom clean
        ccImage, rres, cclist = minorCycleImage(dty_image, psf_image, nminor, gain = loopgain)
        #cc, rres = hogbom(dty_image, psf_image, True, gain, thresh, nminor)
        ccLIST[i] = cclist
        temp_cc_image = ccImage
        # Add to
        cc_image+=temp_cc_image.real

        #################################################################

        #3. FFT the clean components
        # i.e. convert from the model image to model visibilities
        modvis = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(IF.pad_fft(temp_cc_image))))

        #4. Predict the continuous visibilities from the model visibilities
        #from the A kernel i.e the forward step
        if image_params['kernel'] == 'awkernel':
            predVis,uvws = awDeGrid(modvis,uvw,HA,image_params,obs_params,Mterms,Mterms_ij)
            vis -= predVis # determine the residual visibilities
            dty_image,psf_image = GF.awGrid(vis,HA,uvws,image_params,obs_params,Mterms,Mterms_ij)

        elif image_params['kernel'] == 'wkernel':
            predVis,uvws = wDeGrid(modvis,uvw,image_params,obs_params,pswf)
            #test_vis_subtract(vis, predVis, uvw2, uvws, image_params, obs_params,pswf)
            dty_image_pred,psf_image = GF.wGrid(predVis,uvws,image_params,obs_params,pswf)

            #FM.writetoFITS_scratch(imagename+'_pred_image_clean.fits',dty_image_pred.real/np.amax(psf_image.real),
                                    # fov,Array,hduList,psf_fit,imageType='image')
        elif image_params['kernel'] == 'pswf':
            predVis,uvws = DeGrid(modvis,uvw,image_params,obs_params,pswf)

            #5. Subtract the predicted visibilities from the ungridded sorted visibilities
            vis = vis - predVis # determine the residual visibilities
            dty_image,psf_image = GF.Gridding(vis,uvws,image_params,obs_params,pswf)

            FM.writetoFITS_scratch('post_subt_dirtypsf_real.fits',psf_image.real/np.amax(psf_image),
                                   fov,Array,hduList,psf_fit, imageType='dirtyimage')
            #sys.exit()
            #FM.writetoFITS_scratch(imagename+'_res_image_clean.fits',dty_image.real/np.amax(psf_image.real),
                                    # fov,Array,hduList,psf_fit,imageType='image')


        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
        print 'flux cleaned in this cycle = %0.4f' % (np.sum(cclist))
        print 'total amplitude cleaned after major cycle %0d = %0.4f' %(i,np.sum(cc_image.real.ravel()))
        print 'peak amplitude after major cycle %0d = %0.4f' %(i,np.amax(temp_cc_image.real))
        print 'peak amplitude in FT[vis residuals] = %0.4f' % (np.amax(np.abs(dty_image.real))/np.amax(np.abs(psf_image.real)))
        print 'image rms after major cycle %0d = %0.10f ' %(i,np.std(temp_cc_image.real))
        print '%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'


    return vis, cc_image, dty_image.real, psf_image.real, ccLIST


def test_vis_subtract(vis, predVis, uvw, uvws, image_params, obs_params, pswf):

    """
    vis - original visibilities
    predVis - model visibilities degridded from model image
    """



    print '----- print test images -----'
    #	residual image from subtracted vis

    #uvw, vis = FM.conjugateSymetry(uvw,vis)
    #uvws, predVis = FM.conjugateSymetry(uvws,predVis)

    #1. subtract
    #uvw2,vis_resx = IF.sortw(uvw,vis[0])
    #uvw2,vis_resy = IF.sortw(uvw,vis[1])
    #vis_res = np.array([vis_resx,vis_resy])

    vis_res = vis - predVis

    #2. conjugate
    #uvws2, vis_res = FM.conjugateSymetry(uvws,vis_res)

    dty_image_vis_res,psf_image_vis_res = GF.wGrid(vis_res,uvws,image_params,obs_params,pswf) # vis residual image
    pmax = np.amax(psf_image_vis_res.real)

    print 'Peak flux vis subtracted resid image = ', np.amax(dty_image_vis_res.real/pmax)
    y,x = np.unravel_index(np.argmax(dty_image_vis_res.real), np.shape(dty_image_vis_res.real))
    print '@ (%s,%s)' % (str(x), str(y))

    # residual image from subtracted images

    dty_image_vis,psf_image_vis = GF.wGrid(vis,uvw,image_params,obs_params,pswf) #  FT of orig vis
    print 'Peak flux orig dirty image = ', np.amax(dty_image_vis.real/pmax)
    y,x = np.unravel_index(np.argmax(dty_image_vis.real), np.shape(dty_image_vis.real))
    print '@ (%s,%s)' % (str(x), str(y))



    dty_image_pred,psf_image_pred = GF.wGrid(predVis,uvws,image_params,obs_params,pswf) #FT of model vis
    print 'Peak flux model dirty image = ', np.amax(dty_image_pred.real/pmax)
    y,x = np.unravel_index(np.argmax(dty_image_pred.real), np.shape(dty_image_pred.real))
    print '@ (%s,%s)' % (str(x), str(y))


    res_image = dty_image_vis - dty_image_pred #residual image
    print 'Peak flux image subt dirty image = ', np.amax(res_image.real/pmax)
    y,x = np.unravel_index(np.argmax(res_image.real), np.shape(res_image.real))
    print '@ (%s,%s)' % (str(x), str(y))




    plt.clf()
    plt.subplot(221)
    plt.imshow(dty_image_vis.real/pmax)
    plt.title("FT of orig vis")

    plt.subplot(222)
    plt.imshow(dty_image_pred.real/pmax)
    plt.title("FT of model vis")
    #"""
    plt.subplot(223)
    plt.imshow(res_image.real/pmax)
    plt.title("residual image")

    #"""
    plt.subplot(224)
    plt.imshow(dty_image_vis_res.real/pmax)
    plt.title("vis residual image")

    plt.show()

    sys.exit()






def flip(array):

    size = np.size(array)
    #invarray = np.rot90(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))
    #invarray = np.flipud(np.rot90(array))
    invarray = np.fliplr(array)
    return invarray


def awDeGrid(modvis,uvw,HA,image_params,obs_params,Mterms,Mterms_ij):

    """
    degrid the gridded model visibilities into continuous visibilities
    using the Akernels
    modvis - Gridded model visibilities
    """

    ch_width 	= obs_params['ch_width']
    lam		= LIGHT_SPEED/obs_params['ref_freq']

    over_sampling 	= image_params['over_sampling']
    kernel_support 	= image_params['kernel_support']
    ff_size 	= image_params['ff_size']
    wplanes		= image_params['wplanes']
    tinc 		= image_params['tinc']
    N 		= image_params['imageSize']#number of pixels in the image
    T2 		= np.radians((image_params['imageSize'] * image_params['cellsize'])/3600)/2 #FOV [radians] of the image.
    #let the cell size define the max uv distance
    L2 = (N)/(T2*5)
    print 'L2:', L2

    #sort UVW, vis and HA in order of increasing w
    print 'vis size: ', np.shape(modvis)
    zs=np.argsort(uvw[:,2],None)
    uvw = uvw[zs]
    #vis = modvis[zs]
    HA  = HA[zs]
    print 'HA shape: ', np.shape(HA)

    #Divide visibilities into wplanes and iterate over each wplane
    ii=np.linspace(0,len(uvw),wplanes+1)#range(0, len(vis), wstep) #
    #print 'ii:', ii
    ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(uvw))]

    resXX = []
    resYY = []
    for j in range (len(ir)-1):

        ilow,ihigh=ir[j]
        #Obtain the mean w value for the current plane
        w=uvw[ilow:ihigh,2].mean()

        ############SORT VISIBILITIES IN TIME#########
        print 'sorting visibilities in time'
        uvw_temp = uvw[ilow:ihigh]
        #vis_temp = vis[ilow:ihigh]
        HA_temp  = HA[ilow:ihigh]
        ts=np.argsort(HA_temp)
        uvw = uvw_temp[ts]
        #vis = vis_temp[ts]
        times = HA_temp[ts]
        print times[0], times[-1]
        print np.amin(times), np.amax(times)

        #generate image plane wkernel
        wCF = GK.Wkernelimage(w,over_sampling,ff_size,T2/2)

        haSteps = int((np.amax(times) - np.amin(times))/(tinc/60.))+1
        print 'HA steps = ', haSteps
        for i in range (haSteps):
            #check if there are visibilities within this time range
            trange = (times[0]+(tinc/60.)*i, times[0]+(tinc/60.)*(i+1))
            targs = np.where(np.logical_and(times>=trange[0], times<=trange[1]))
            if np.size(targs) > 0:
                print 'number of visibilities within %s = %s ' %(str(trange),str(np.size(targs)))
                #obtain the Akernel
                aCFXX = GK.akernel(i,times,uvw,image_params,obs_params,Mterms[0],Mterms_ij[0])
                aCFYY = GK.akernel(i,times,uvw,image_params,obs_params,Mterms[1],Mterms_ij[1])

            else:
                continue

        #combine with the Wkernel
    wgXX = GK.GCF(wCF,aCFXX,over_sampling,kernel_support)
    wgYY = GK.GCF(wCF,aCFYY,over_sampling,kernel_support)

    #De grid - the crux of this code!!!!
    resXX.append(convdegrid(modvis, uvw[ilow:ihigh] / L2, wgXX))
    resYY.append(convdegrid(modvis, uvw[ilow:ihigh] / L2, wgYY))
    degridvisXX = sum(resXX)
    degridvisYY = sum(resYY)
    degridvis = np.array([degridvisXX,degridvisYY])
    uvw[:,2] *= -1

    return degridvis, uvw





def DeGrid(modvis,uvw,image_params,obs_params,pswf):


    """
    restructure the gridded visibilities to continuous visibilities
    using the wkernel
    """

    ch_width 		= obs_params['ch_width']
    lam				= LIGHT_SPEED/obs_params['ref_freq']

    over_sampling 	= image_params['over_sampling']
    kernel_support 	= image_params['kernel_support']
    ff_size 		= image_params['ff_size']
    wplanes			= image_params['wplanes']
    tinc 			= image_params['tinc']
    N 				= image_params['imageSize']#number of pixels in the image
    T2 				= np.radians((image_params['imageSize'] * image_params['cellsize'])/3600)/2 #FOV [radians] of the image.
    #let the cell size define the max uv distance
    L2 = N/(T2*4)

    uvw = IF.sortw(uvw,None)

    #Divide visibilities into wplanes and iterate over each wplane
    ii=np.linspace(0,len(uvw),wplanes+1).astype(int)#range(0, len(vis), wstep) #
    #print 'ii:', ii
    ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(uvw))]

    res = []
    #resYY = []
    for j in range (len(ir)-1):

        ilow,ihigh=ir[j]
        wg = pswf

        #De grid - the crux of this code!!!!
        res.append(convdegrid(modvis, uvw[ilow:ihigh] / L2, wg))
        degridvis_t = sum(res)
        degridvis = np.array([degridvis_t,degridvis_t])
        uvw[:,2] *= -1


    return degridvis, uvw









def wDeGrid(modvis,uvw,image_params,obs_params,pswf):


    """
    restructure the gridded visibilities to continuous visibilities
    using the wkernel
    """

    ch_width 		= obs_params['ch_width']
    lam				= LIGHT_SPEED/obs_params['ref_freq']

    over_sampling 	= image_params['over_sampling']
    kernel_support 	= image_params['kernel_support']
    ff_size 		= image_params['ff_size']
    wplanes			= image_params['wplanes']
    tinc 			= image_params['tinc']
    N 				= image_params['imageSize']#number of pixels in the image
    T2 				= np.radians((image_params['imageSize'] * image_params['cellsize'])/3600)/2 #FOV [radians] of the image.
    #let the cell size define the max uv distance
    L2 = N/(T2*4)

    uvw = IF.sortw(uvw,None)

    #Divide visibilities into wplanes and iterate over each wplane
    ii=np.linspace(0,len(uvw),wplanes+1).astype(int)#range(0, len(vis), wstep) #
    #print 'ii:', ii
    ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(uvw))]

    res = []
    #resYY = []
    for j in range (len(ir)-1):

        ilow,ihigh=ir[j]
        #Obtain the mean w value for the current plane
        w=uvw[ilow:ihigh,2].mean()
        #-- wkernel
        if pswf is not None:
            wgcf = GK.Wkernel(w,over_sampling,kernel_support,ff_size,T2/2) # w-kernel
            wg = GK.aaGCF(wgcf,pswf,over_sampling) # combine wkernel and the pswf
        else:
            wg = GK.Wkernel(w,over_sampling,kernel_support,ff_size,T2/2)

    #De grid - the crux of this code!!!!
    res.append(convdegrid(modvis, uvw[ilow:ihigh] / L2, wg))
    degridvis_t = sum(res)
    degridvis = np.array([degridvis_t,degridvis_t])
    uvw[:,2] *= -1


    return degridvis, uvw





def convdegrid(a, p, gcf):
    """Convolutional-degridding

    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :param gcf: List of convolution kernels

    :returns: Array of visibilities.
    """
    x, xf, y, yf = convcoords(a, p, len(gcf))
    v = []
    sx, sy = gcf[0][0].shape[0] // 2, gcf[0][0].shape[1] // 2
    #print 'sx,sy:', sx, sy
    for i in range(len(x)):
        pi = (x[i], y[i])
    #print 'pi:', pi
        v.append((a[pi[0] - sx: pi[0] + sx + 1, pi[1] - sy: pi[1] + sy + 1] * gcf[xf[i]][yf[i]]).sum())
    return np.array(v)




def hogbom(dirty,psf,window,gain,thresh,niter):

    """
    Hogbom CLEAN (1974A&AS...15..417H)
    
    :param dirty: The dirty Image, i.e., the Image to be deconvolved
    
    :param psf: The point spread-function
    
    :param window: Regions where clean components are allowed. If
      True, all of the dirty Image is assumed to be allowed for
      clean components
    
    :param gain: The "loop gain", i.e., the fraction of the brightest
      pixel that is removed in each iteration
    
    :param thresh: Cleaning stops when the maximum of the absolute
      deviation of the residual is less than this value
    
    :param niter: Maximum number of components to make if the
      threshold `thresh` is not hit
    
    :returns clean SkyComponent Image, residual Image
    """
    
    assert 0.0 < gain < 2.0
    assert niter > 0
    
    comps = np.zeros(dirty.shape) # empty array shape of the dirt image to store clean components
    res = np.array(dirty) # residuals
    pmax = psf.max()
    assert pmax > 0.0 #testing whether the psf max value is > 0
    psfpeak = argmax(np.abs(psf)) # the absolute values of psfmax element-wise
    
    if window is True:
        window = np.ones(dirty.shape, np.bool) # array size of the dirty image filled with Truth values
    
    for i in range(niter):
        mx, my = np.unravel_index((np.abs(res[window])).argmax(), dirty.shape)
        val = res[mx, my] * gain / pmax
        comps[mx, my] += mval
        a1o, a2o = overlapIndices(dirty, psf,
                      mx - psfpeak[0],
                      my - psfpeak[1])
        res[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval

        if np.abs(res).max() < thresh:
            print 'threshold reached: res max = %s < thresh %s' %(np.abs(res).max(),thresh)
            break

    return comps, res



def minorCycleImage(res_image2, psf, nmin, gain=0.1):

    """
    Simple implementation of Hogbom clean.
    outputs a model image of clean components
    """

    print '--------Doing %s minor cleaning cycles' % str(nmin)

    cc_image = np.zeros(res_image2.shape)
    #psf = FM.cleanBeam(psf)
    pmax = np.amax(psf.real)
    clean_threshold = threshold(res_image2,psf.real,cyclefactor=1.5)

    #peak_amp = np.amax(res_image2.real)
    #print '%%%%%%%%%%%%%%%%%%%%%%%MINOR CYCLE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'
    #print '-----peak flux before cleaning = %0.6f' %(peak_amp/pmax)
    assert pmax > 0.0 #testing whether the psf max value is > 0
    psfpeak = np.unravel_index(np.argmax(psf.real), np.shape(psf))
    ccList = np.zeros(nmin)
    for i in range (nmin):

        # 1 find peak flux
        peak_amp = np.amax(res_image2.real)/pmax
        #print '-----peak flux before cleaning = %0.6f' %(peak_amp)
        if peak_amp > clean_threshold:
            y,x = np.unravel_index(np.argmax(res_image2), np.shape(res_image2))

            mval = peak_amp * gain
            #2 subtract peak flux * gain * psf
            a1o, a2o = overlapIndices(res_image2.real, psf, y - psfpeak[0], x - psfpeak[1])

            res_image2.real[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf.real[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
            peak_amp = np.amax(res_image2.real)

            #3. record peak flux to clean image * clean psf
            cc_image[y,x] += mval
            ccList[i] = mval # save flux cleaned for the ith minor cycle
            #print '--------------------------------------------------'
            #print '-----After niter %s, flux clean = %0.6f @ (%s,%s)' % (str(i), mval, str(x), str(y))
            #print '--------------------------------------------------'
        else:
            print 'Threshold reached'
            print 'Clean using %s iterations to reach a threshold of %0.6f' %(str(i), clean_threshold)
            break


    #print 'totall flux clean after %s minor cycles = %0.6f' % (str(nmin),np.sum(ccList.real,axis=1))
    peak_amp = np.amax(res_image2.real)
    print '-----peak flux in residual after minor cleaning = %0.6f' %(peak_amp/pmax)
    #print '%%%%%%%%%%%%%%%%%%%%%%END MINOR CYCLE%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'


    return cc_image, res_image2.real, ccList



def threshold(residual,dirtypsf,cyclefactor=1.5):
    """
    determine the minor cycle threshold to begin
    the major cycle
    will use the CASA threshold definition:
    cycle threshold = cyclefactor * max sidelobe * max residual
    where cyclefactor ranges from 0.25 to 1.5. Here a defult of 1.5 will be used
    see https://casa.nrao.edu/Release3.4.0/docs/userman/UserMansu274.html for details
    """

    #1. find the peak residuals
    resPeak = np.amax(residual.real)/np.amax(dirtypsf)
    #2. find the peak psf sidelobe
    #2a. find the main lobe of psf
    psf_side_lobe = FitpsfMainLobe(dirtypsf)
    print 'The minor cycle threshold is peak residual * %0.6f' % (cyclefactor * psf_side_lobe)
    threshold = cyclefactor * resPeak * psf_side_lobe
    print 'Maximum residual = %0.6f, cleaning down to %0.6f' % (resPeak,threshold)

    return threshold


def FitpsfMainLobe(dirtypsf):

    """
    To simulate a clean psf (or clean beam)
    by fitting a 2D Gaussian to the main lobe of the dirty beam.
    """

    PSF = dirtypsf/np.amax(dirtypsf)
    peak_psf = np.amax(PSF)
    #print 'peak psf:', peak_psf
    cent = np.shape(PSF)[0]/2
    x = np.arange(0,len(PSF))
    y=x

    xx,yy = np.meshgrid(x,y)

    g_init = models.Gaussian2D(amplitude=peak_psf,x_mean=cent,y_mean=cent,x_stddev=1,y_stddev=1)
    fit_g = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')

    g = fit_g(g_init, xx, yy, PSF)
    #print 'psf fitting', g
    cleanpsf = g(xx,yy)/np.amax(g(xx,yy))
    side_lobes = PSF - cleanpsf
    max_side_lobe = np.amax(side_lobes)

    return max_side_lobe



def argmax(a):
    """ Return unravelled index of the maximum

    param: a: array to be searched
    """
    return np.unravel_index(a.argmax(), a.shape)




def overlapIndices(a1, a2,
                   shiftx, shifty):
    """ Find the indices where two arrays overlapIndices

    :param a1: First array
    :param a2: Second array
    :param shiftx: Shift in x applied to a1
    :param shifty: Shift in y applied to a2
    """
    if shiftx >= 0:
        a1xbeg = shiftx
        a2xbeg = 0
        a1xend = a1.shape[0]
        a2xend = a1.shape[0] - shiftx
    else:
        a1xbeg = 0
        a2xbeg = -shiftx
        a1xend = a1.shape[0] + shiftx
        a2xend = a1.shape[0]

    if shifty >= 0:
        a1ybeg = shifty
        a2ybeg = 0
        a1yend = a1.shape[1]
        a2yend = a1.shape[1] - shifty
    else:
        a1ybeg = 0
        a2ybeg = -shifty
        a1yend = a1.shape[1] + shifty
        a2yend = a1.shape[1]

    return (a1xbeg, a1xend, a1ybeg, a1yend), (a2xbeg, a2xend, a2ybeg, a2yend)






def convcoords(a, p, Qpx):
    """Compute grid coordinates and fractional values for convolutional
    gridding

    The fractional values are rounded to nearest 1/Qpx pixel value at
    fractional values greater than (Qpx-0.5)/Qpx are roundeded to next
    integer index
    """
    (x, xf), (y, yf) = [fraccoord(a.shape[i], p[:, i], Qpx) for i in [0, 1]]
    return x, xf, y, yf


def fraccoord(N, p, Qpx):
    """Compute whole and fractional parts of coordinates, rounded to Qpx-th fraction of pixel size

    :param N: Number of pixels in total
    :param p: coordinates in range [-.5,.5]
    :param Qpx: Fractional values to round to
    """
    #print 'N = %s, p = %s, Qpx = %s' %(str(N),str(p),str(Qpx))	
    x = (1. + p) * N/2
    flx = np.floor(x + 0.5 / Qpx)
    fracx = np.around(((x - flx) * Qpx))
    return flx.astype(int), fracx.astype(int)











