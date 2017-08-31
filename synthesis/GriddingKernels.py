"""
Functions related to kernel generation and extraction

Hayden Rampadarath 2016
hayden.rampadarath@manchester.ac.uk or haydenrampadarath@gmail.com

"""
import sys,os,traceback, optparse
import time, ephem
import re, math
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv
from scipy import signal
from scipy.special import j1 #Bessel Function
import AntennaFunctions as AF
import Imagefuncs as IF
import GriddingFunctions as GF
from pswf import *

#sys.path.append("../staley_gridder")
from kernel_generation import Kernel
import conv_funcs as conv_funcs


LIGHT_SPEED = 299792458         # Speed of light


def writeCF(CF,filename):
        cf = np.ravel(CF)
        cf_re = cf.real
	#print np.amax(cf_re)
	#sys.exit()
        cf_im = cf.imag
        np.savetxt(filename,np.column_stack((cf_re,cf_im)),newline='\n')
        return

def readCF(filename):
        cf_re,cf_im = np.loadtxt(filename,unpack=True)
        cf = cf_re + 1j*cf_im
        CF = cf.reshape(np.sqrt(len(cf)),np.sqrt(len(cf)))
        return CF





def StaleyKernel(support,oversampling):

    """AA Gridding kernel based upon Tim Staley's Fast Imaging prototype"""

    #1. Specify the convolution function
    narrow_g_sinc = conv_funcs.GaussianSinc(trunc=support)
    #2. generate in the uv plane
    gs_kernel = Kernel(kernel_func=narrow_g_sinc, support=support, oversampling=oversampling,pad=True)
    #3. FT into the image plane
    af = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gs_kernel.array)))
    #4. pad to a common size

    print('Shape StaleyKernel \n{}'.format(af.shape))


    wg=[[GF.wextract(af, i, j, oversampling, support) for i in range(oversampling)] for j in range(oversampling)]

    wg = np.conj(np.array(wg))

    print('Shape Staley conv kernel \n{}'.format(wg.shape))
    #sys.exit()

    return wg








def Wkernel(wmean,over_sampling,kernel_support,ff_size,T2):

	"""
	determine the combine convolution kernel
	written for pAWIv3.py
	"""
	#print 'wmean= ', wmean
	WCF = Wkernelimage(wmean,over_sampling,ff_size,T2)
	
	af=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(WCF)))

	wgF = askapsoft_decimate_n_extract(af, over_sampling, kernel_support)

	wg = np.conj(wgF)

	print("Size of wkernel: {}".format(np.shape(wg)))
	sys.exit()
	return wg




def Wkernelimage(w,over_sampling,ff_size,T2):

	"""
	Obtain only the Wkernels
	Note: this determine only a single WKernel
	Ideally, Wkwernels should be determine per w-plane
	Use a nearest neighbour method to determine which 
	Wkernel to use to grid the visibilities -> this applies to determinng the w-phase
	wstep   - number of w plane steps
	ff_size - 
	T2      -
	num_vis - number of visibilities
	uvw     - the uvw coordinates
	over_sampling  -
	kernel_support -
	"""
	
	ff = T2*np.mgrid[-1:(1.0*(ff_size-2)/ff_size):(ff_size*1j),-1:(1.0*(ff_size-2)/ff_size):(ff_size*1j)]
	#print 'shape(ff) = ', np.shape(ff)
	r2 = (ff**2).sum(axis=0)
	ph=w*(1.-np.sqrt(1.-r2)) 

	cp=(np.exp(2j*np.pi*ph))
	#cp=(np.exp(1j*ph))


	padff=np.pad(cp, 
	    pad_width=int(ff_size*(over_sampling-1.)/2.,),
	    mode='constant',
	    constant_values=(0.0,))

	print('wkernel pad image shape:', np.shape(cp))


	return padff



def getAkernelimage(pangle,over_sampling,ff_size,Mterm):

	"""
	Determine the AW convolution function.

	pangle  - parallactic angle [radians]
	ff_size -
	over_sampling  -
	kernel_support -
	T2      - half-width of FOV [radians]
	"""

	#print 'Mterm shape:', np.shape(Mterm)
	print 'Computing Mueller terms with parallactic rotation %s degrees' %(str(pangle))
	beam = np.zeros((np.shape(Mterm)[0],np.shape(Mterm)[0]),dtype=complex)

	#rotate beam 
	beam.real = AF.rotatebeam(Mterm.real,pangle)
	beam.imag = AF.rotatebeam(Mterm.imag,pangle)
	#beam.real = Mterm.real
	#beam.imag = Mterm.imag

	padff=np.pad(beam, 
	    pad_width=int(ff_size*(over_sampling-1.)/2.,),
	    mode='constant',
	    constant_values=(0.0,))
	print 'padff shape: ', np.shape(padff)
	return padff




def aaGCF(wg,pswf,over_sampling):

	"""
	combine wkernel, akernel of awkernel with the anti-aliasing kernel: the prolate spheroidal 
	"""
	
	for i in range (0, over_sampling):
		wg[i,0,:,:] = signal.convolve2d(wg[i,0,:,:],pswf,mode='same',boundary='symm')	

	return wg


def GCF(WCF,ACF,over_sampling,kernel_support):

	"""
	determine the combine convolution kernel
	WCF - Wprojection convolution kernel (image plane)
	ACF - Aprojection convolution kernel (image plane)
	"""
	print('Shape StaleyKernel \n{}'.format(np.shape(ACF)))
	print('wkernel pad image shape:', np.shape(WCF))

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

	#print 'acf shape', np.shape(ACF)
		
	#padff = WCF * ACF
	padff = ACF
	print 'gcf image max = ', np.amax(padff.real)
	#padff = ACF
	#padff = np.conj(padff) * padff
	#ipadff = inv(padff)

	af=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(padff)))
	#iaf = inv(af)
	#wg = askapsoft_decimate_n_extract(af, over_sampling, kernel_support)
	wg=[[wextract(af, i, j, over_sampling, kernel_support) for i in range(over_sampling)] for j in range(over_sampling)]
	
	wg = np.conj(np.array(wg))
	
	return wg





def akernel(i,times,uvw,image_params,obs_params,Mterm,Mterms_ij):


	lat 	 	= obs_params['lat']
	dec 	 	= obs_params['DEC']
	lam		= LIGHT_SPEED/obs_params['ref_freq']
	
	over_sampling 	= image_params['over_sampling']
	kernel_support 	= image_params['kernel_support']
	ff_size 	= image_params['ff_size']
	tinc 		= image_params['tinc']
	Acache		= image_params['Acache']
	
	#all visibilities within the time range defined by (t[0], t[0]+tinc) will require the same A-kernel
	trange = (times[0]+(tinc/60.)*i, times[0]+(tinc/60.)*(i+1))	
	    
	print '----------Gridding Time Step ', i
	print '----------Gridding Hour Angle range ', trange
	

	#Obtain the AWkernel gridding convolution function
	

	print '----doing AWprojection----'
	#get mean hour angle
	hamean = '%2.2f' % ((trange[0]+trange[-1])/2.)
	print 'mean HA = ', hamean
	#1. determine parallactic angle
	panglea = IF.parangle(float(hamean),dec,lat)
	#panglea = panglea*-1
	pangle = '%2.2f' % panglea
	print 'Paralactic Angle', pangle
	#The name 00 for the ACF represents the first Mueller term or the pure XX term
	#need to change the name to correspond to the particular Mueller term
	AkernelName = 'ACF_%s_%s_hrs_%s_deg_inv' % (Mterms_ij,hamean,pangle)

	if os.path.isfile(Acache+AkernelName):
	   	#if Akernel exists in cache, read 
	   	 print 'reading Akernel (%s) from cache' % AkernelName
	   	 aCF = readCF(Acache+AkernelName)
	else:
	   	#if not, generate rotated Akernel and save to cache
	   	print 'generating new Akernel (%s)' % AkernelName
	   	aCF = getAkernelimage(float(pangle),over_sampling,ff_size,Mterm)
	   	IF.plot2Dimage(aCF,Acache+AkernelName)
	   	writeCF(aCF,Acache+AkernelName)

	return aCF


