import numpy as np
import matplotlib.pyplot as pl
from numpy.random import normal
from astropy.modeling import models, fitting
import warnings
from synthesisImaging import *

#######################
####Set some globals
#######################
c0 		= 3.0e8

hastart 	= 0 # HA start of observation in hrs
haend   	= 0.0167  # HA end of observation in hrs
hastep  	= 1 # scan length of observation in mins
dec 		= 0.78539816339  # pi/4 Radians declination of source
lat 		= 34 #degrees - latitude of the array


######Frequency set up
start_freq 	= 660e6 	#in Hz              
nchan      	= 1 # number of freq channels
freq_inc   	= 1e6 # size of each channel in Hz

######################



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







def deGridding(modvis,uvw):

	"""
	the forward step i.e. from visibilities on a regular grid 
	to visibilities on an irregular grid
	"""	
	#######DEGRIDDING
	
	temp=uvw
	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]

	
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





def minorCycleImage(res_image2, psf, nmin, gain=0.1):

	"""
	Simple implementation of Hogbom clean.
	outputs a model image of clean components
	"""
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


def majorCycle(vis, uvw, dirty_psf, nmaj, nmin, gain):

	vis_res = vis
	res_image2, psf = gridding(vis_res,uvw)
	ccLIST = np.zeros((nmaj,nmin))
	cc_image = np.zeros(np.shape(res_image2))
	for i in range(nmaj):

		#1. generate dirty image of residual dataset
		#2. obtain a model image clean components via minor cycle (i.e. Hogbom Clean)

		temp_cc_image, rres, cclist = minorCycleImage(res_image2, dirty_psf, nmin, gain)
		ccLIST[i] += cclist
		temp_cc_image = udrot90(temp_cc_image)
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
		modvis = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift((temp_cc_image))))

		#4. obtain the degridded/predicted model visibilities
		predvis,uvw = deGridding(modvis,uvw)
		predvis= np.reshape(predvis,len(vis_res))

		"""
		mod_image, psf = gridding(predvis,uvw)
		pl.subplot(122)
		pl.imshow(mod_image.real)
		pl.title("Model Image after "+str(i)+" major cycle")
		pl.show()
		"""
		#5. subtract the model degridded visibilities from the original visibilities to obtain the residuals
		vis_res -= predvis
		res_image2,psf = gridding(vis_res,uvw)

		#vis_res = [vis_res[i] - predvis[i] for i in range (len(vis))]
		#advance counter
	print '------total amplitude after all major cycles = %0.4f' %(np.sum(ccLIST.real.ravel()))
	
	return cc_image, rres, predvis, ccLIST




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
    			      xcc[i] - psfpeak[0],
    			      ycc[i] - psfpeak[1])

        	clIMG[a1o[0]:a1o[1], a1o[2]:a1o[3]] += clBeam[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval
		
	
	cleanImage = clIMG + rres
	#"""

	return cleanImage

def conjugateSymmetry_notime(basel_uvw,vis):
	"""
	Apply the conjugate symmetry to the visibility data
	"""
	uu = basel_uvw[:,0]
	vv = basel_uvw[:,1]
	ww = basel_uvw[:,2]
	
	uvw = np.column_stack((uu,vv,ww))#column_stack - Stack 1-D arrays as columns into a 2-D array.
	#get conjugate symmetry
	tmp_uvw = uvw*np.array([-1.,-1.,1.])
	
	#sort the UVW data 
	uvw = np.vstack((uvw,tmp_uvw))#vstack - Stack arrays in sequence vertically (row wise).
		
	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]

	tmp_vis = np.conj(vis)
	vis = np.hstack((vis,tmp_vis))
	vis = vis[zs]
	
	return uvw, vis


"""
def GenerateVisibilities():

	ants_xyz=np.genfromtxt("./VLA_C_hor_xyz_v2.txt")
	print "Number of antennas in array:",ants_xyz.shape[0]

	ha = 0.0; dec = np.pi/4.  # Units are Radians pointing centre
	har = ha * 15 * (np.pi/180)#convert ha to radians
	hanum = 1 # number of time samples
	
	basel_uvw = basel_rot(ants_xyz,start_freq,har,dec,lat)
	visscan = make_vis(basel_uvw)
	vis = add_vis_noise(visscan,hanum)

	#basel_uvw = np.concatenate(basel_uvwall)
	#vis = np.concatenate(visall)
	#time = np.concatenate(timeall)

	uvw, vis = conjugateSymmetry_notime(basel_uvw,vis)
	return vis,uvw
"""



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
	tmp_uvw = uvw*np.array([-1.,-1.,1.])
	tmp_vis = vis_re - 1j*vis_im
	
	vis = np.hstack((vis,tmp_vis))
	uvw = np.vstack((uvw,tmp_uvw))


	return vis, uvw 






########################################################################

vis, uvw = GenerateVisibilities()
    
############################IMAGING: with sources
dirty_image,dirty_psf = gridding(vis,uvw)



############################


nmaj = 5
nmin = 30
cc_image, rres, predvis, ccLIST = majorCycle(vis, uvw, dirty_psf, nmaj, nmin, gain=0.05)

#gereate clean image once clean was completed
clean_image = MakeCleanImage(cc_image,rres,dirty_psf)

print '################################'
print '------Maximum peak of dirty image before cleaning: ', np.amax(dirty_image.real/np.amax(dirty_psf.real))

#print '------Maximum peak of final clean model image: ', np.amax(cc_image)

print '------Maximum peak of final clean image: ', np.amax(clean_image)


print '------std of image after cleaning: ', np.std(clean_image.real)

pl.clf()
pl.subplot(131)
pl.imshow(dirty_image.real/np.amax(dirty_psf.real))
pl.title("Dirty Image")
pl.subplot(132)
pl.imshow(clean_image)
pl.title("clean model after major cycles")
pl.subplot(133)
pl.imshow(rres.real)
pl.title("residual Image after major cycles")
pl.show()


#ccLIST = np.cumsum(np.sum(ccLIST.real,axis=1))
ccLIST = (np.sum(ccLIST.real,axis=1))

pl.plot(ccLIST,'r')
pl.xlabel('Number of major iterations')
pl.ylabel('Flux cleaned')
pl.show()
#"""




