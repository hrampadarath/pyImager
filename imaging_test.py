import sys,os,traceback, optparse
import time, ephem
import re, math
import numpy as np
import numpy.ma as ma
import warnings
import matplotlib.pyplot as plt
import os.path
import Imagefuncs as IF
import GriddingKernels as GK
import GriddingFunctions as GF
import FITSmodules as FM
from kernel_generation import Kernel
import conv_funcs as conv_funcs
from staley_gridder.gridder import convolve_to_grid
from staley_gridder.conv_funcs import GaussianSinc

import astropy.units as u


def makeDirtyImageFITS(vis,HA,uvw,image_params,obs_params,files,Array,hduList):
    """
    generate dirty image
    """
    ref_freq = obs_params['ref_freq']/1e6
    ch_width = obs_params['ch_width']

    over_sampling = image_params['over_sampling']
    kernel_support = image_params['kernel_support']
    ff_size = image_params['ff_size']

    dty_image,psf_image = grid(vis,uvw,image_params,obs_params)
    #dty_image,psf_image = Staley_Gridder(vis,HA,uvw,image_params,obs_params)

    fov = (image_params['imageSize'] * image_params['cellsize'])/3600.
    imagename = files['wdir']+files['imagename']


    psf_peak = np.amax((psf_image.real))
    corr_dirty_image = (dty_image.real)/psf_peak#/corrFunc

    #obtain the clean beam
    clBeam, psf_fit = FM.cleanBeam(psf_image,image_params)#/psf_peak

    #print psf_fit


    FM.writetoFITS_scratch(imagename +'_dirtyimage_real.fits', corr_dirty_image, fov, Array, hduList, psf_fit, imageType='dirtyimage')
    FM.writetoFITS_scratch(imagename +'_dirtypsf_real.fits',psf_image.real/psf_peak,fov,Array,hduList,psf_fit, imageType='dirtyimage')





def Staley_Gridder(vis,HA,uvw,image_params,obs_params):

    ref_freq = obs_params['ref_freq']/1e6
    ch_width = obs_params['ch_width']

    over_sampling = image_params['over_sampling']
    #kernel_support = image_params['kernel_support']
    ff_size = image_params['ff_size']
    kernel_support = 3
    kernel_func = GaussianSinc(trunc=kernel_support)
    image_size = image_params['imageSize']#
    cell_size= image_params['cellsize']

    #Size of a UV-grid pixel, in multiples of wavelength (lambda)

    grid_pixel_width_lambda = 1.0 / (np.radians(cell_size/3600.) * image_size)

    uvw_in_pixels = (uvw[:,0:2] / grid_pixel_width_lambda)

    grid_wt, grid_uv = convolve_to_grid(kernel_func,kernel_support,image_size,uvw_in_pixels,vis[0],exact=True,oversampling=0,raise_bounds=True)

    dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_uv))))
    psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_wt))))

    plt.imshow(dty_image.real,cmap='gray')
    plt.show()
    return dty_image, psf_image



def grid(vis,uvw,image_params,obs_params):
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
    grid_wt, grid_uv = gridOnePolWproj(vis[0],uvw,image_params,obs_params)


    dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_uv))))
    psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(IF.pad_fft(grid_wt))))

    return dty_image, psf_image



def gridOnePolWproj(vis,uvw,image_params,obs_params):


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
    #L2 = int(1/np.radians((image_params['cellsize'])/3600.))
    L2 = N/(T2*6)
    print('L2 = ', L2)

    print('max U = {} and max V = {}'.format(np.amax(uvw[:,0]),np.amax(uvw[:,1])))

    #sys.exit()
    # remove all u/v values above L2 and the corresponding vis and w values.
    uvw,vis = IF.uvTapering(uvw,vis,L2)

    num_vis = len(vis)
    #print 'num vis = ', num_vis

    #print "Making grids of side: ",N," pixels."
    grid_uv=np.zeros([N, N], dtype=complex)
    grid_wt=np.zeros([N, N], dtype=complex)

    #sort UVW in w

    zs=np.argsort(uvw[:,2])
    uvw = uvw[zs]
    vis = vis[zs]

    #Divide visibilities in terms of the w-term and get the mean w value.
    ii=np.linspace(0,len(vis),wplanes+1).astype(int)#range(0, len(vis), wstep) #
    #print 'ii:', ii
    ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(vis))]

    for j in range (len(ir)-1):

        ilow,ihigh=ir[j]
        #Obtain the Wkernel/gridding convolution function
        w=uvw[ilow:ihigh,2].mean()

        #wg = GK.Wkernel(w,over_sampling,kernel_support,ff_size,T2)

        wg = StaleyKernel(kernel_support,over_sampling)

        uvw_sub = uvw[ilow:ihigh]/L2 # extract subarray for w-slice
        #print uvw_sub

        (x, xf), (y, yf) = [GF.fraccoord(grid_uv.shape[i], uvw_sub[:,i], over_sampling) for i in [0,1]]

        grid_uvi=np.zeros([N, N], dtype=complex)
        grid_wti=np.zeros([N, N], dtype=complex)

        vis_sub = vis[ilow:ihigh]
        #print 'vis_sub:', np.shape(vis_sub)
        for i in range(len(x)):

            GF.convgridone(grid_uvi,(x[i], y[i]), (xf[i], yf[i]), wg, vis_sub[i])
            GF.convgridone(grid_wti,(x[i], y[i]), (xf[i], yf[i]), wg, 1.0+0j)
        #"""
        grid_uv.real += grid_uvi.real
        grid_uv.imag += grid_uvi.imag

        grid_wt.real += grid_wti.real
        grid_wt.imag += grid_wti.imag
        #"""


    return grid_wt, grid_uv


def StaleyKernel(support,oversampling):

    """Gridding kernel based upon Tim Staley's Fast Imaging prototype"""
    #1. Specify the convolution function
    narrow_g_sinc = conv_funcs.GaussianSinc(trunc=support)
    #2. generate in the uv plane
    gs_kernel = Kernel(kernel_func=narrow_g_sinc, support=support, oversampling=oversampling,pad=True)
    #3. FT into the image plane
    af = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(gs_kernel.array)))
    #4. pad to a common size


    wg=[[GF.wextract(af, i, j, oversampling, support) for i in range(oversampling)] for j in range(oversampling)]

    wg = np.conj(np.array(wg))

    print('Shape Staley conv kernel \n{}'.format(wg.shape))
    #sys.exit()

    return wg



