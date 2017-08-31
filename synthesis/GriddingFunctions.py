"""
Functions related to gridding of visibilities

Hayden Rampadarath 2016
hayden.rampadarath@manchester.ac.uk or haydenrampadarath@gmail.com

"""


#Some common imported modules - uncomment as required

import sys,os,traceback, optparse
import time, ephem
import os.path
import re, math
import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.time import Time
from scipy import signal
import warnings


#######USER DEFINED FUNCTIONS
import Imagefuncs as IF
import FITSmodules as FM
import AntennaFunctions as AF
import GriddingKernels as GK


#Define global constants
global EARTH_RADIUS, LIGHT_SPEED 
EARTH_RADIUS = 6371 * 10**3     # Earth's radius
LIGHT_SPEED = 299792458         # Speed of light







def StaleyGridding(vis,uvw,image_params,obs_params):


	print '--------------Gridding X stokes--------------------'
	#xgrid_wt, xgrid_uv, N = gridOnePolWproj(vis[0],uvw,image_params,obs_params,pswf)

	kernel_support = 3
	kernel_func = GaussianSinc(trunc=kernel_support)
	vis_grid, sample_grid = convolve_to_grid(kernel_func,
                                             support=kernel_support,
                                             image_size=image_size_int,
                                             uv=uv_in_pixels,
                                             vis=vis,
                                             vis_weights=vis_weights,
                                             exact=kernel_exact,
                                             oversampling=kernel_oversampling,
                                             progress_bar=progress_bar)

	dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_uv))))
	psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_wt))))

	return dty_image, psf_image







def Gridding(vis,uvw,image_params,obs_params,pswf):
	"""
	grid and make dirty Stokes I image
	no Aprojection, only Wprojection
	note: not fully tested 
	"""	
	ref_freq = obs_params['ref_freq']/1e6
	#print 'ref freq =', ref_freq
	lat 	 = obs_params['lat']
	ch_width = obs_params['ch_width']
	DEC 	 = obs_params['DEC']
	Stokes = image_params['Stokes']
	
	print '--------------Gridding X stokes--------------------'
	xgrid_wt, xgrid_uv, N = gridder(vis[0],uvw,image_params,obs_params,pswf)
	print '--------------Gridding Y stokes--------------------'
	ygrid_wt, ygrid_uv, N  = gridder(vis[1],uvw,image_params,obs_params,pswf)

	N = np.shape(xgrid_wt)[0]
	grid_uv = np.zeros([N, N], dtype=complex)
	grid_wt = np.zeros([N, N], dtype=complex)
	
	if Stokes == 'I':
		#combine X and Y gridded vis to create the I pol gridded vis
		# I = (XX+YY)/2
		grid_uv.real = (ygrid_uv.real + xgrid_uv.real)/2
		grid_uv.imag = (ygrid_uv.imag + xgrid_uv.imag)/2

		#combine X and Y gridded wt to create the I pol gridded wt
		grid_wt.real = (ygrid_wt.real + xgrid_wt.real)/2
		grid_wt.imag = (ygrid_wt.imag + xgrid_wt.imag)/2

	elif Stokes == 'Q':
		#combine X and Y gridded vis to create the I pol gridded vis
		# Q = (XX-YY)/2
		grid_uv.real = (ygrid_uv.real - xgrid_uv.real)/2
		grid_uv.imag = (ygrid_uv.imag - xgrid_uv.imag)/2

		#combine X and Y gridded wt to create the I pol gridded wt
		grid_wt.real = (ygrid_wt.real - xgrid_wt.real)/2
		grid_wt.imag = (ygrid_wt.imag - xgrid_wt.imag)/2

	dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_uv))))
	psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_wt))))

	return dty_image, psf_image



def gridder(vis,uvw,image_params,obs_params,pswf):


	lat 	 = obs_params['lat']
	ch_width = obs_params['ch_width']
	dec 	 = obs_params['DEC']

	over_sampling = image_params['over_sampling']
	kernel_support = image_params['kernel_support']
	ff_size = image_params['ff_size']
	wplanes =  image_params['wplanes']

	N = image_params['imageSize']#number of pixels per side
	T2 = np.radians((N * image_params['cellsize'])/3600)/2 #half-width of FOV [radians] of the image.
	print 'fov (degrees):',np.degrees(T2)*2

	#let the cell size define the max uv distance
	L2 = int(1/np.radians((image_params['cellsize'])/3600.))
	#L2 = N/(T2*4)

	# remove all u/v values above L2 and the corresponding vis and w values. 
	uvw,vis = IF.uvTapering(uvw,vis,L2)

	num_vis = len(vis)
	#print 'num vis = ', num_vis

	#print "Making grids of side: ",N," pixels."
	grid_uv=np.zeros([N, N], dtype=complex)
	grid_wt=np.zeros([N, N], dtype=complex)

	#sort UVW in w
	uvw,vis = IF.sortw(uvw,vis)

	#Divide visibilities in terms of the w-term and get the mean w value.
	ii=np.linspace(0,len(vis),wplanes+1).astype(int)#range(0, len(vis), wstep) #
	#print 'ii:', ii
	ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(vis))]

	for j in range (len(ir)-1):

		ilow,ihigh=ir[j]

		#Obtain the Wkernel/gridding convolution function	
		w=uvw[ilow:ihigh,2].mean()

		wg = pswf

		uvw_sub = uvw[ilow:ihigh]/L2 # extract subarray for w-slice
		#print uvw_sub

		(x, xf), (y, yf) = [fraccoord(grid_uv.shape[i], uvw_sub[:,i], over_sampling) for i in [0,1]]

		grid_uvi=np.zeros([N, N], dtype=complex)
		grid_wti=np.zeros([N, N], dtype=complex)

		vis_sub = vis[ilow:ihigh]
		#print 'vis_sub:', np.shape(vis_sub)
		for i in range(len(x)):

			convgridone(grid_uvi,(x[i], y[i]), (xf[i], yf[i]), wg, vis_sub[i])
			convgridone(grid_wti,(x[i], y[i]), (xf[i], yf[i]), wg, 1.0+0j)
		#"""	
		grid_uv.real += grid_uvi.real
		grid_uv.imag += grid_uvi.imag

		grid_wt.real += grid_wti.real
		grid_wt.imag += grid_wti.imag		
		#"""


	return grid_wt, grid_uv, N





	
def MuellerTerms(image_params,ref_freq,ch_width):

	print '---generating the unrotated Mueller Terms and keep in memory----'

	#for multi-freq ... this needs rethinking
	#Note The Mueller Terms are dependednt on freq
	fov = np.radians((image_params['imageSize'] * image_params['cellsize'])/3600.)
	freq = ref_freq
	xyJones = AF.makeJonesMatrix(fov,freq)
	x = xyJones[0]
	y = xyJones[1]
	Jones = xyJones[2:]
	Jx = Jones[0]/np.amax(Jones[0])
	Jy = Jones[1]/np.amax(Jones[1])

	#return the complex Mterms
	MtermsXX = Jx*np.conj(Jx)
	MtermsYY = Jy*np.conj(Jy)
	

	#MtermsXX = MuellerMatrix(x,y,Jones,'XX')
	#MtermsYY = MuellerMatrix(x,y,Jones,'YY')
	#MtermsXY = MuellerMatrix(x,y,Jones,'XY')
	#MtermsYX = MuellerMatrix(x,y,Jones,'YX')	

	Mterms = [MtermsXX,MtermsYY]
	Mterms_ij = ['00','33']
	# kernel indexing: 4x4 matrix starting at 00 and ending at 33. 
	# 00 - XX; 11 - XY; 22 - YX; 33 - YY	

	return Mterms, Mterms_ij





def awGrid(vis,HA,uvw,image_params,obs_params,Mterms,Mterms_ij):
	"""
	grid and make dirty Stokes I 
	with awprojection
	vis = (xvis,yvis)
	Mterms = (MtermsXX,MtermsYY)
	"""	
	Stokes = image_params['Stokes']
	


	print '--------------Gridding X pol--------------------'
	xgrid_wt, xgrid_uv = gridOnePolAWproj(vis[0],HA,uvw,image_params,obs_params,Mterms[0],Mterms_ij[0])
	print '--------------Gridding Y pol--------------------'
	ygrid_wt, ygrid_uv = gridOnePolAWproj(vis[1],HA,uvw,image_params,obs_params,Mterms[1],Mterms_ij[1])

	N = np.shape(xgrid_wt)[0]
	grid_uv = np.zeros([N, N], dtype=complex)
	grid_wt = np.zeros([N, N], dtype=complex)
	
	if Stokes == 'I':
		#combine X and Y gridded vis to create the I pol gridded vis
		# I = (XX+YY)/2
		grid_uv.real = (ygrid_uv.real + xgrid_uv.real)/2
		grid_uv.imag = (ygrid_uv.imag + xgrid_uv.imag)/2

		#combine X and Y gridded wt to create the I pol gridded wt
		grid_wt.real = (ygrid_wt.real + xgrid_wt.real)/2
		grid_wt.imag = (ygrid_wt.imag + xgrid_wt.imag)/2

	elif Stokes == 'Q':
		#combine X and Y gridded vis to create the I pol gridded vis
		# Q = (XX-YY)/2
		grid_uv.real = (ygrid_uv.real - xgrid_uv.real)/2
		grid_uv.imag = (ygrid_uv.imag - xgrid_uv.imag)/2

		#combine X and Y gridded wt to create the I pol gridded wt
		grid_wt.real = (ygrid_wt.real - xgrid_wt.real)/2
		grid_wt.imag = (ygrid_wt.imag - xgrid_wt.imag)/2


	dty_image=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(grid_uv)))
	psf_image=np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(grid_wt)))

	return dty_image, psf_image






def gridOnePolAWproj(vis,HA,uvw,image_params,obs_params,Mterm,Mterms_ij):
	

	lat 	 	= obs_params['lat']
	ch_width 	= obs_params['ch_width']
	dec 	 	= obs_params['DEC']
	haLims 		= obs_params['haLims']

	over_sampling 	= image_params['over_sampling']
	kernel_support 	= image_params['kernel_support']
	ff_size 	= image_params['ff_size']
	wplanes		= image_params['wplanes']
	tinc 		= image_params['tinc']
	Acache		= image_params['Acache']


	N 		= image_params['imageSize']#number of pixels per side
	T2 		= np.radians((image_params['imageSize'] * image_params['cellsize'])/3600)/2 #half-width of FOV [radians] of the image.
	print 'fov (degrees):',np.degrees(T2)*2

	#let the cell size define the max uv distance
	#L2 = int(1/np.radians((image_params['cellsize'])/3600.))
	L2 = N/(T2*5)

	# remove all u/v values above L2 and the corresponding vis and w values. 
	uvw,vis = IF.uvTapering(uvw,vis,L2)

	num_vis = len(vis)
	#print 'num vis = ', num_vis
	
	print "Making grids of side: ",N," pixels."
	grid_uv=np.zeros([N, N], dtype=complex)
	grid_wt=np.zeros([N, N], dtype=complex)

	#sort UVW, vis and HA in order of increasing w

	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]
	vis = vis[zs]
	HA  = HA[zs]

	#Divide visibilities into wplanes and iterate over each wplane
	ii=np.linspace(0,len(vis),wplanes+1)#range(0, len(vis), wstep) #
	#print 'ii:', ii
	ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(vis))]
	

	for j in range (len(ir)-1):

		ilow,ihigh=ir[j]
		#Obtain the mean w value for the current plane
		w=uvw[ilow:ihigh,2].mean()

		############SORT VISIBILITIES IN TIME#########	
		print 'sorting visibilities in time'
		uvw_temp = uvw[ilow:ihigh]
		vis_temp = vis[ilow:ihigh]
		HA_temp  = HA[ilow:ihigh]
		ts=np.argsort(HA_temp)
		uvw = uvw_temp[ts]
		vis = vis_temp[ts]
		times = HA_temp[ts]
		print times[0], times[-1]
		print np.amin(times), np.amax(times)
		
		#generate image plane wkernel 
		wCF = GK.Wkernelimage(w,over_sampling,ff_size,T2)

		haSteps = int((np.amax(times) - np.amin(times))/(tinc/60.))+1 
		print 'HA steps = ', haSteps
		for i in range (haSteps):
			#check if there are visibilities within this time range	
			trange = (times[0]+(tinc/60.)*i, times[0]+(tinc/60.)*(i+1))				
			targs = np.where(np.logical_and(times>=trange[0], times<=trange[1]))
			if np.size(targs) > 0:
				print 'number of visibilities within %s = %s ' %(str(trange),str(np.size(targs)))
				#obtain the Akernel
				aCF = GK.akernel(i,times,uvw,image_params,obs_params,Mterm,Mterms_ij)				

			else:
				continue
 
			#combine with the Wkernel			
			wg = GK.GCF(wCF,aCF,over_sampling,kernel_support)

			uvw_sub = uvw[ilow:ihigh]/L2 # extract subarray for w-slice
			#print uvw_sub
	
			(x, xf), (y, yf) = [fraccoord(grid_uv.shape[i], uvw_sub[:,i], over_sampling) for i in [0,1]]
			
			grid_uvi=np.zeros([N, N], dtype=complex)
			grid_wti=np.zeros([N, N], dtype=complex)
			
			vis_sub = vis[ilow:ihigh]
			#print 'vis_sub:', np.shape(vis_sub)
			for i in range(len(x)):

				convgridone(grid_uvi,(x[i], y[i]), (xf[i], yf[i]), wg, vis_sub[i])
				convgridone(grid_wti,(x[i], y[i]), (xf[i], yf[i]), wg, 1.0+0j)
			
			grid_uv.real += grid_uvi.real
			grid_uv.imag += grid_uvi.imag
			
			grid_wt.real += grid_wti.real
			grid_wt.imag += grid_wti.imag		
		


	return grid_wt, grid_uv
	


def askapsoft_decimate_n_extract(af, over_sampling, kernel_support):

    """
    Extracted and translated from
    AWProjectVisGridder.cc by A. Scaife
    """

    # why is this normalization required..?
    rescale = over_sampling*over_sampling
    #rescale = 1

    cSize = 2 * kernel_support + 1
    itsConvFunc=np.zeros((over_sampling, over_sampling, cSize, cSize), dtype=complex)

    for fracu in range(0,over_sampling):
        for fracv in range(0,over_sampling):

            # Now cut out the inner part of the convolution function and
            # insert it into the convolution function
            for iy in range(-kernel_support,kernel_support+1):
                for ix in range(-kernel_support,kernel_support+1):

                    nx = af.shape[0]
                    ny = af.shape[1]

                    # assumes support is the same for all w-planes:
                    xval = (ix) * over_sampling + fracu + nx / 2
                    yval = (iy) * over_sampling + fracv + ny / 2

                    itsConvFunc[fracu, fracv, ix+cSize/2, iy+cSize/2] \
                            = rescale * af[xval, yval]

    return itsConvFunc[::-1,::-1]



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
	#print 'x b4 rever:', x, np.shape(x)
	#for i in range(np.shape(x)[0]): print i, np.abs(x[i,64])
	x=x[::-1,::-1] # reverse the kernel
	x*=1.0/x.sum() # normalise the kernel

	return exmid2(x,s)






def fraccoord(N, p, Qpx):
    """Compute whole and fractional parts of coordinates, rounded to Qpx-th fraction of pixel size
    :param N: Number of pixels in total
    :param p: coordinates in range -1,1
    :param Qpx: Fractional values to round to
    """
    #Qpx = 1
    H=N/2
    #print 'H:', H
    x=(1+p)*H
    flx=np.floor(x + 0.5 /Qpx)
    fracx=np.around(((x-flx)*Qpx))
    return (flx).astype(int), fracx.astype(int)


def convgridone(a, pi, fi, gcf, v):
    	"""Convolve and grid one visibility sample"""
    	sx, sy= gcf[0][0].shape[0]/2, gcf[0][0].shape[1]/2

	#print gcf[0][0].shape
	#print 'sx,sy:', sx, sy
	#print pi[0]-sx, pi[0]+sx+1
	#print pi[0]
	#print pi[1]-sy, pi[1]+sy+1
	#print pi[1]
	#print np.shape(a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ])

	#print np.shape(gcf[fi[1],fi[0]])
	a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] += gcf[fi[1],fi[0]]*v
    	return a

