import numpy as np
import matplotlib.pyplot as pl
from numpy.random import normal
from astropy.modeling import models, fitting
import warnings
import sys, os
from scipy import signal
from convolutionFunctions import *


#######################
####Set some globals
#######################
wstep=2000
over_sampling=4
ff_size=128
kernel_support=15

T2 = 0.015  # half-width of FOV [radians]
L2 = 4000 # half-width of uv-plane [lambda]
N = int(T2*L2*4) # number of pixels 
#print "Making grids of side: ",N," pixels."
pix2rad = (0.015*2)/N

######################

def exmid2(a, s):
    """Extract a section from middle of a map, suitable for zero frequencies at N/2"""
    cx=a.shape[0]/2
    cy=a.shape[1]/2
    return a[cx-s-1:cx+s, cy-s-1:cy+s]

# works in the opposite sense to ASKAPsoft, which calculates an under-sampled kernel and 
# interpolates.
def wextract(a, i, j, Qpx, s):
    """Extract the (ith,jth) w-kernel from the oversampled parent and normalise
    The kernel is reversed in order to make the suitable for
    correcting the fractional coordinates
    """
    x=a[i::Qpx, j::Qpx] # decimate the kernel
    x=x[::-1,::-1] # reverse the kernel
    x*=1.0/x.sum() # normalise the kernel
    
    return exmid2(x,s)



def fraccoord(N, p, Qpx):
    """Compute whole and fractional parts of coordinates, rounded to Qpx-th fraction of pixel size
    :param N: Number of pixels in total 
    :param p: coordinates in range -1,1
    :param Qpx: Fractional values to round to
    """
    H=N/2
    x=(1+p)*H
    flx=np.floor(x + 0.5 /Qpx)
    fracx=np.around(((x-flx)*Qpx))    
    return (flx).astype(int), fracx.astype(int)



def convgridone(a, pi, fi, gcf, v):
    """Convolve and grid one visibility sample"""
    sx, sy= gcf[0][0].shape[0]/2, gcf[0][0].shape[1]/2
    
    # NB the order of fi below 
    a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] += gcf[fi[1],fi[0]]*v  
    return a


def gridone(a,p,v):
    """grid one visibility without convolution"""
    
    a[p[0],p[1]] += v
    
    return a


def rows2cols(array):
	
	size = np.size(array)	
	invarray = np.flipud(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray

def row2colslr(array):
	
	size = np.size(array)	
	invarray = np.rot90(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray

def gridding(vis,uvw):
	
	#######GRIDDING
	grid_uv=np.zeros([N, N], dtype=complex)
	grid_wt=np.zeros([N, N], dtype=complex)
	
	temp=uvw
	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]
	vis = vis[zs]
	
	ii=range(0, len(uvw), wstep) # Bojan, is this what you mean here? Every 2000 entries or every 2000 lambda..?
	
	ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(uvw))]
	ilow,ihigh=ir[0]
	
	#calculate the wkernel
	w=uvw[ilow:ihigh,2].mean()
	ff = T2*np.mgrid[-1:(1.0*(ff_size-2)/ff_size):(ff_size*1j),-1:(1.0*(ff_size-2)/ff_size):(ff_size*1j)]
	r2 = (ff**2).sum(axis=0)
	#calculate convolution kernel
	#wg = wkernel(w,r2)
	wg = anti_aliasing_uv(w,r2)
	uvw_sub = uvw[ilow:ihigh,:]/L2 # extract subarray for w-slice
	(x, xf), (y, yf) = [fraccoord(grid_uv.shape[i], uvw_sub[:,i], over_sampling) for i in [0,1]]

	vis_sub = vis[ilow:ihigh]
	for i in range(len(x)):
	    #gridone(grid_uv, (x[i],y[i]), vis_sub[i])
	    #gridone(grid_wt, (x[i],y[i]), 1.0+0j)	
	    convgridone(grid_uv,(x[i], y[i]), (xf[i], yf[i]), wg, vis_sub[i])
	    convgridone(grid_wt,(x[i], y[i]), (xf[i], yf[i]), wg, 1.0+0j)
	
	
	dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_uv)))
	psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_wt)))
	
	return dty_image, psf_image



def wkernel(w,r2):
	ph=w*(1.-np.sqrt(1.-r2)) 
	cp=(np.exp(2j*np.pi*ph))
	padff=np.pad(cp,
		pad_width=(int(ff_size*(over_sampling-1.)/2.),),
		mode='constant',
		constant_values=(0.0,))
	af=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padff)))
	wg=[[wextract(af, i, j, over_sampling, kernel_support) for i in range(over_sampling)] for j in range(over_sampling)]
	# Convert list to numpy array:
	wg = np.array(wg)
	wg = np.conj(wg)
	return wg
	



def anti_aliasing(w,r2):

	support = 3
	oversample = 128
	#compute prolate spheroidal
	pswf = make_oversampled_pswf(support,oversample) #pswf in uv plane
	pswf_im=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(pswf)))
	ACF=np.pad(pswf_im,
		pad_width=(int(ff_size*(256-1.)/2.),),
		mode='constant',
		constant_values=(0.0,))
	#compute wkernel
	ph=w*(1.-np.sqrt(1.-r2)) 
	cp=(np.exp(2j*np.pi*ph))

	WCF=np.pad(cp,
		pad_width=(int(ff_size*(over_sampling-1.)/2.),),
		mode='constant',
		constant_values=(0.0,))
		#combine

	if np.shape(ACF)[0] > np.shape(WCF)[0]:
		#Resize to match
		N = np.shape(WCF)[0]
		cent = (np.shape(ACF)[0]/2,np.shape(ACF)[1]/2)
		ACF  = ACF[cent[0]-N/2:cent[0]+N/2, cent[1]-N/2:cent[1]+N/2]
	elif np.shape(ACF)[0] < np.shape(WCF)[0]:
		#Resize to match
		N = np.shape(ACF)[0]
		cent = (np.shape(WCF)[0]/2,np.shape(WCF)[1]/2)
		WCF  = WCF[cent[0]-N/2:cent[0]+N/2, cent[1]-N/2:cent[1]+N/2]

	af = WCF * ACF
	wg =np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(af)))


	wg=[[wextract(af, i, j, over_sampling, kernel_support) for i in range(over_sampling)] for j in range(over_sampling)]
	# Convert list to numpy array:
	wg = np.array(wg)
	wg = np.conj(wg)
	return wg

def anti_aliasing_uv(w,r2):

	support = 3
	oversample = 128
	#compute prolate spheroidal
	pswf = make_oversampled_pswf(support,oversample) #pswf in uv plane

	#compute wkernel
	ph=w*(1.-np.sqrt(1.-r2)) 
	cp=(np.exp(2j*np.pi*ph))

	WCF=np.pad(cp,
		pad_width=(int(ff_size*(over_sampling-1.)/2.),),
		mode='constant',
		constant_values=(0.0,))
		#combine

	af=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(WCF)))

	wg=[[wextract(af, i, j, over_sampling, kernel_support) for i in range(over_sampling)] for j in range(over_sampling)]
	# Convert list to numpy array:
	wg = np.array(wg)
	wg = np.conj(wg)
	for i in range (0, over_sampling):
		wg[i,0,:,:] = signal.convolve2d(wg[i,0,:,:],pswf,mode='same',boundary='symm')	


	return wg
	



def convdegrid(a, p, gcf):
    """Convolutional-degridding

    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :param gcf: List of convolution kernels

    :returns: Array of visibilities.
    """
    x, xf, y, yf = convcoords(a, p, len(gcf))
    #print 'a.shape = ', a.shape
    #print 'x:', x
    #print 'y:', y
    v = []
    sx, sy = gcf[0][0].shape[0] // 2, gcf[0][0].shape[1] // 2
    #print 'sx,sy:', sx, sy
    for i in range(len(x)):
        pi = (x[i], y[i])
	#print 'pi:', pi
        v.append((a[pi[0] - sx: pi[0] + sx + 1, pi[1] - sy: pi[1] + sy + 1] * gcf[xf[i]][yf[i]]).sum())
    return np.array(v)


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

def deGridding(modvis,uvw):

	"""
	the forward step i.e. from visibilities on a regular grid 
	to visibilities on an irregular grid
	"""	
	#######DEGRIDDING
	
	uvw_orig=uvw
	uvw = sortw(uvw,None)
	
	ii=range(0, len(vis), wstep)	
	ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(vis))]
	ilow,ihigh=ir[0]
	
	#calculate the wkernel
	w=uvw[ilow:ihigh,2].mean()
	ff = T2*np.mgrid[-1:(1.0*(ff_size-2)/ff_size):(ff_size*1j),-1:(1.0*(ff_size-2)/ff_size):(ff_size*1j)]
	r2 = (ff**2).sum(axis=0)
	ph=w*(1.-np.sqrt(1.-r2)) 
	cp=(np.exp(2j*np.pi*ph))
	padff=np.pad(cp,
		pad_width=(int(ff_size*(over_sampling-1.)/2.),),
		mode='constant',
		constant_values=(0.0,))
	af=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padff)))
	wg=[[wextract(af, i, j, over_sampling, kernel_support) for i in range(over_sampling)] for j in range(over_sampling)]
	# Convert list to numpy array:
	wg = np.array(wg)
	wg = np.conj(wg)
	
	# degridding using the wkernel

	#vis_degrid = []
	vis_degrid = (convdegrid(modvis, uvw[ilow:ihigh] / L2, wg)) 
	#print uvw[ilow:ihigh]
	uvw[:,2] *= -1
	return vis_degrid,uvw







def overlapIndices(a1, a2,
                   shiftx, shifty):
    """ 
	Taken from B. Nikolic csclean.py from the SKA/crocodile github page
	Find the indices where two arrays overlapIndices

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



"""
def minorCycleImage(res_image2, psf, nmin, gain=0.1):

	#
	#Simple implementation of Hogbom clean.
	#outputs a model image of clean components
	#
	cc_image = np.zeros(res_image2.shape)
	psf = cleanBeam(psf)
	pmax = np.amax(psf.real)
	print 'psf max =', pmax
	
    	assert pmax > 0.0 #testing whether the psf max value is > 0
	psfpeak = np.unravel_index(np.argmax(psf.real), np.shape(psf))
	ccList = np.zeros(nmin)
	for i in range (nmin):
	
		# 1 find peak flux
		peak_amp = np.amax(res_image2.real)
		x,y = np.unravel_index(np.argmax(res_image2), np.shape(res_image2))

		mval = peak_amp * gain /pmax
		#2 subtract peak flux * gain * psf

		a1o, a2o = overlapIndices(res_image2.real, psf, x - psfpeak[0], y - psfpeak[1])	
		res_image2.real[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf.real[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
		peak_amp = np.amax(res_image2.real)

		#3. record peak flux to clean image * clean psf
		cc_image[x,y] += mval
		ccList[i] = mval # save flux cleaned for the ith minor cycle

	return cc_image, res_image2.real, ccList
"""

def minorCycleImage(res_image2, psf, nmin, gain=0.1):

    
    #Simple implementation of Hogbom clean.
    #outputs a model image of clean components
    
    
    pmax = np.amax(psf.real)
    print 'psf max =', pmax
 
    cc_image = np.zeros(res_image2.shape)
    #psf = FM.cleanBeam(psf)
    peak_amp = np.amax(res_image2.real)
    print '-----peak flux before cleaning = %0.6f' %(peak_amp/pmax)
    assert pmax > 0.0 #testing whether the psf max value is > 0
    psfpeak = np.unravel_index(np.argmax(psf.real), np.shape(psf))
    ccList = np.zeros(nmin)
    for i in range (nmin):
        # 1 find peak flux
        peak_amp = np.amax(res_image2.real)
        y,x = np.unravel_index(np.argmax(res_image2), np.shape(res_image2))

        mval = peak_amp * gain /pmax
        #2 subtract peak flux * gain * psf
        a1o, a2o = overlapIndices(res_image2.real, psf, y - psfpeak[0], x - psfpeak[1])	
        res_image2.real[a1o[0]:a1o[1], a1o[2]:a1o[3]] -= psf.real[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
                       

        #3. record peak flux to clean image * clean psf
        cc_image[y,x] += mval
        ccList[i] = mval # save flux cleaned for the ith minor cycle

        print 'After niter %s, flux clean = %0.6f @ (%s,%s)' % (str(i), mval, str(x), str(y))


    peak_amp = np.amax(res_image2.real)
    print '-----peak flux in residual after cleaning = %0.6f' %(peak_amp/pmax)
    peak_amp_2 = np.amax(cc_image)
    print '-----peak flux in clean model after cleaning = %0.6f' %(peak_amp_2)


    return cc_image, res_image2.real, ccList



def row2cols(array):
	
	size = np.size(array)	
	invarray = np.flipud(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray


def udrot90(array):
	
	size = np.size(array)	
	#invarray = np.rot90(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))
	invarray = np.flipud(np.rot90(array))
	return invarray


def sortw(uvw,vis):
	zs=np.argsort(uvw[:,2])
	if vis is not None:
		return uvw[zs], vis[zs]
	else:
		return uvw[zs]


def majorCycle(vis, uvw, dirty_psf, nmaj, nmin, gain):

	vis_res = vis
	uvw2,vis_res = sortw(uvw,vis_res)
	ccLIST = np.zeros((nmaj,nmin))
	#1. generate dirty image of residual dataset
	res_image2, psf = gridding(vis,uvw)
	cc_image = 0
	for i in range(nmaj):


		#2. obtain a model image clean components via minor cycle (i.e. Hogbom Clean)

		temp_cc_image, rres, cclist = minorCycleImage(res_image2, dirty_psf, nmin, gain)
		ccLIST[i] += cclist
		temp_cc_image = temp_cc_image
		cc_image+=temp_cc_image.real

		"""
		#clean_image = MakeCleanImage(temp_cc_image,rres,psf)
		pl.clf()
		pl.subplot(121)
		pl.imshow(temp_cc_image)
		pl.title("clean components after "+str(i)+" major cycle")

		"""
		print 'flux cleaned in this cycle = %0.4f' % (np.sum(cclist))
		print 'total amplitude cleaned after major cycle %0d = %0.4f' %(i,np.sum(cc_image.real.ravel()))
		print 'peak amplitude after major cycle %0d = %0.4f' %(i,np.amax(temp_cc_image.real))
		print 'image rms after major cycle %0d = %0.10f ' %(i,np.std(temp_cc_image.real))

		#3. obtain model gridded visibilities from the model image 
		#modvis = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(np.rot90(temp_cc_image))))
		modvis = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(temp_cc_image)))		

		#4. obtain the degridded/predicted model visibilities
		predvis,uvws = deGridding(modvis,uvw)
		print np.shape(predvis)
		#predvis= np.reshape(predvis,len(vis_res))
				

		test_vis_subtract(vis_res, predvis, uvw, uvws)
		"""
		res_image,psf = gridding(vis_res,uvw)
		pl.clf()
		pl.imshow(res_image.real/np.amax(psf.real))
		pl.show()
		"""
		#5. subtract the model degridded visibilities from the original visibilities to obtain the residuals
		vis_res -= predvis
		
		res_image2,psf = gridding(vis_res,uvws)
		"""
		pl.clf()
		pl.imshow(res_image2.real/np.amax(psf.real))
		pl.show()
		sys.exit()
		"""
		#vis_res = [vis_res[i] - predvis[i] for i in range (len(vis))]
		#advance counter
	print '------total amplitude after all major cycles = %0.4f' %(np.sum(ccLIST.real.ravel()))
	
	return cc_image, rres, res_image2, ccLIST



def test_vis_subtract(vis_res, predvis, uvw, uvws):

	print '----- print test images -----'
	#	residual image from subtracted vis
	#vis_res2 = np.array([vis_res[i] - predvis[i] for i in range (len(vis_res))])
	
	#predvis = predvis[::-1]
	vis = vis_res
	uvw2,vis_res = sortw(uvw,vis_res)
	vis_res = vis_res - predvis
	#print np.shape(vis_res2)



	# conjugate symmetry
	#uvws2, vis_res2 = conjugateSymetry(uvws,vis_res2)

	dty_image_vis_res,psf_image_vis_res = gridding(vis_res,uvws)

	# residual image from subtracted images
	dty_image_vis,psf_image_vis = gridding(vis,uvw)
	dty_image_pred,psf_image_pred = gridding(predvis,uvws)
	res_image = dty_image_vis - dty_image_pred
	#dty_image_pred = dty_image_pred


	

	pmax = np.amax(psf_image_vis.real)

	pl.clf()
	pl.subplot(221)
	pl.imshow(dty_image_vis.real/pmax)
	pl.title("ft of raw visibilities")
	pl.subplot(222)
	pl.imshow(dty_image_pred.real/pmax)
	pl.title("ft of predicted visibilities")
	pl.subplot(223)
	pl.imshow(res_image.real/pmax)
	pl.title("residual image")
	pl.subplot(224)
	pl.imshow(dty_image_vis_res.real/pmax)
	pl.title("vis residual image")

	pl.show()
	sys.exit()


def cleanBeam(dirtypsf):

    """
    To simulate a clean psf (or clean beam)
    by fitting a 2D Gaussian to the main lobe of the dirty beam.
    """

    PSF = dirtypsf
    cent = np.shape(PSF)[0]/2
    x = np.arange(0,len(PSF))
    y=x
    
    xx,yy = np.meshgrid(x,y)
    
    z = PSF
    
    g_init = models.Gaussian2D(amplitude=1,x_mean=cent,y_mean=cent,x_stddev=1,y_stddev=1)
    fit_g = fitting.LevMarLSQFitter()
    
    with warnings.catch_warnings():
    # Ignore model linearity warning from the fitter
	warnings.simplefilter('ignore')
	g = fit_g(g_init, xx, yy, z)
	#print 'psf fitting', g
    cleanpsf = g(xx,yy)
    pS = 30 # radius in pixel of clean beam

    sub_psf = cleanpsf[cent-pS:cent+pS,cent-pS:cent+pS]
    pad_width = np.shape(PSF)[0]/2 - pS
    sub_psf = np.pad(sub_psf,(pad_width,pad_width),mode='constant')

    return sub_psf


def MakeCleanImage(cc,rres,dirty_psf):


	#obtain the clean beam
	clBeam = cleanBeam(dirty_psf)

	#peak psf value
	psfpeak = argmax(np.abs(clBeam))
	clBeam = clBeam/np.amax(clBeam.real)
	#obtain the clean image
	ycc,xcc = np.nonzero(cc)
	clIMG = np.zeros(cc.shape)
	#"""
	for i in range (len(ycc)):
		
		mval = cc[ycc[i], xcc[i]]
		#print mval
		a1o, a2o = overlapIndices(cc, clBeam,
    			      ycc[i] - psfpeak[0],
    			      xcc[i] - psfpeak[1])

        	clIMG[a1o[0]:a1o[1], a1o[2]:a1o[3]] += clBeam[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
		
	
	cleanImage = clIMG #+ rres
	#"""

	return cleanImage

def argmax(a):
    """ Return unravelled index of the maximum

    param: a: array to be searched
    """
    return np.unravel_index(a.argmax(), a.shape)


def basel(ants_xyz,ha,dec):
	#UVW coordinates
	x,y,z=np.hsplit(ants_xyz,3)
	
	t=x*np.cos(ha) - y*np.sin(ha)
	u=x*np.sin(ha) + y*np.cos(ha)
	v=-1.*t*np.sin(dec)+ z*np.cos(dec)
	w=t*np.cos(dec)+ z*np.sin(dec)
	ants_uvw = np.hstack([u,v,w])
	#baseline distribution
	
	res=[]
	for i in range(ants_uvw.shape[0]):
	    for j in range(i+1, ants_uvw.shape[0]):
	        res.append(ants_uvw[j]-ants_uvw[i])
	
	basel_uvw = np.array(res)


	return basel_uvw

def Generate_Visibilities():
	
	ants_xyz=np.genfromtxt("./VLA_C_hor_xyz_v2.txt")
	print "Number of antennas in array:",ants_xyz.shape[0]
		
	ha = 0.0; dec = np.pi/4.  # Units are Radians pointing centre
		
	basel_uvw = basel(ants_xyz,ha,dec)
	print "Number of baselines in array:", basel_uvw.shape[0]
	
	
	
	#List of source positions and amplitude
	
	src= {'1':{'l':-0.0015,'m':-0.0058,'src_amp':200}}
	        #'2':{'l':-0.0044,'m':0.003,'src_amp':220},
	        #'3':{'l':-0.001,'m':0.003,'src_amp':230},
	        # '4':{'l':0.005,'m':-0.0065,'src_amp':190}}
	        #'5':{'l':0.003,'m':0.0065,'src_amp':110}}   # Units are Radians
	
	
	#simulate the visibilities
	vis = 0  
	for s in src:
	    l = src[s]['l']
	    m = src[s]['m']
	    src_pos=np.array([l, m , np.sqrt(1 - l**2 - m**2)])
	    src_amp = src[s]['src_amp']
	    vis+=(src_amp*np.exp(-2j*np.pi* np.dot(basel_uvw[:,0:2], src_pos[0:2])))
	    
	#vis=np.sum(vis,axis=0)      
	u = basel_uvw[:,0]
	v = basel_uvw[:,1]
	w = basel_uvw[:,2]
	
	
	#add noise
	mu = 0.
	sigma = 10.*np.sqrt(len(u))*np.sqrt(1) # 1 muJy
	#sigma = 0
	points = len(vis)
	noise = np.zeros((points), dtype=np.complex64)
	noise.real = normal(mu, sigma, points)
	noise.imag = normal(mu, sigma, points)
	
	vis_re = vis.real + noise.real
	vis_im = vis.imag + noise.imag
	vis = vis_re + 1j*vis_im
	
	uvw = np.column_stack((u,v,w))
	# conjugate symmetry
	uvw, vis = conjugateSymetry(uvw,vis)

	return vis, uvw 


def conjugateSymetry(uvw,vis):
	
	"""
	conjugate symmetry
	"""

	
	tmp_uvw = uvw*np.array([-1.,-1.,1.])
	uvw = np.vstack((uvw,tmp_uvw))#vstack - Stack arrays in sequence vertically (row wise).
	print 'uvw shape = ', np.shape(uvw)
	#zs=np.argsort(uvw[:,2])
	#uvw = uvw[zs]
	tmp_vis = np.conj(vis)
	vis = np.hstack((vis,tmp_vis))
	
	print 'vis shape = ', np.shape(vis)
	
	return uvw, vis


########################################################################


vis, uvw = Generate_Visibilities()



############################IMAGING: with sources
dirty_image,dirty_psf = gridding(vis,uvw)
pl.clf()
pl.imshow(dirty_image.real/np.amax(dirty_psf.real))
pl.title("Dirty Image")
pl.show()
sys.exit()
############################


nmaj = 1
nmin = 1
cc_image, rres, res_image2, ccLIST = majorCycle(vis, uvw, dirty_psf, nmaj, nmin, gain=1.0)

#gereate clean image once clean was completed
clean_image = MakeCleanImage(cc_image,res_image2,dirty_psf)

print '################################'
print '------Maximum peak of dirty image before cleaning: ', np.amax(dirty_image.real/np.amax(dirty_psf.real))

#print '------Maximum peak of final clean model image: ', np.amax(cc_image)

print '------Maximum peak of final clean image: ', np.amax(clean_image)


print '------std of image after cleaning: ', np.std(clean_image.real)

pl.clf()
pl.subplot(141)
pl.imshow(dirty_image.real/np.amax(dirty_psf.real))
pl.title("Dirty Image")
pl.subplot(142)
pl.imshow(clean_image)
pl.title("clean")
pl.subplot(143)
pl.imshow(res_image2.real/np.amax(dirty_psf.real))
pl.title("residual")
pl.subplot(144)
pl.imshow(clean_image+res_image2.real/np.amax(dirty_psf.real))
pl.title("clean + residual")

pl.show()

"""
#ccLIST = np.cumsum(np.sum(ccLIST.real,axis=1))
ccLIST = (np.sum(ccLIST.real,axis=1))

pl.plot(ccLIST,'r')
pl.xlabel('Number of major iterations')
pl.ylabel('Flux cleaned')
pl.show()
"""




