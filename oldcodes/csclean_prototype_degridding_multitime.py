import numpy as np
import matplotlib.pyplot as pl
from numpy.random import normal
from astropy.modeling import models, fitting
import warnings
from synthesisImaging import *
import sys

#####GLOBALS
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

def majorCycle(vis, uvw, ha, dirty_psf, nmaj, nmin, gain):

	vis_res = vis
	res_image2, psf = gridding(vis_res, uvw, ha)
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
		print 'shape predvis = ', np.shape(predvis)
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







def GenerateVisibilities():


	###############


	###############
	#ha = 0.0; dec = np.pi/4.  # Units are Radians pointing centre
	hanum = int((haend-hastart)/(hastep/60.))
	harange = np.linspace(hastart,haend,hanum)
	print 'HA num = ', hanum

	arcmin2rad = np.pi / (180 * 60)

	ants_xyz=np.genfromtxt("./VLA_C_hor_xyz_v2.txt")
	print "Number of antennas in array:",ants_xyz.shape[0]

	
	basel_uvwall=[]
	visall = []
	timeall = []
	for ha in harange:
		har = ha * 15 * (np.pi/180)#convert ha to radians
        	# determine the antenna positions given the obs freq and HA
		basel_uvw_scan = basel_rot(ants_xyz,start_freq,har,dec,lat)	
    
        	# determine the XX pol visibilities. See the functions below for more details on the inputs       
        	visscan = make_vis(basel_uvw_scan)
		visscan = add_vis_noise(visscan,hanum)
        	#append the baselines and visibilities for each pol per HA scan
		basel_uvwall.append(basel_uvw_scan)
		visall.append(visscan)
           	#append the visibility times per HA scan
		timescan = np.zeros(len(visscan))+ha		
		timeall.append(timescan)
		

	basel_uvw = np.concatenate(basel_uvwall)
	vis = np.concatenate(visall)
	time = np.concatenate(timeall)

	uvw, vis, ha = conjugateSymmetry(basel_uvw,time,vis)
	return vis,uvw, ha

	
########################################################################
#generate the visinbilities
vis, uvw, ha = GenerateVisibilities()

#pl.plot(uvw[: ,0],uvw[: ,1], 'kx')
#pl.show()

#sys.exit()

############################IMAGING: with sources

#1. Make initial dirty image
#dirty_image,dirty_psf = gridding(vis, uvw, ha)
dirty_image, dirty_psf = alt_grid(vis, uvw, ha)
pl.imshow(dirty_image.real/np.amax(dirty_psf.real))
pl.title("Dirty Image")
pl.show()
sys.exit()

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
print '################################'


######Make plots

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




