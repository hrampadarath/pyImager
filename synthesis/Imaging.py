import FITSmodules as FM
import GriddingKernels as GK
import GriddingFunctions as GF
import numpy as np








def makeDirtyImage(vis,HA,uvw,image_params,obs_params,files,Array,hduList):
	"""
	generate dirty image
	"""
	ref_freq = obs_params['ref_freq']/1e6
	ch_width = obs_params['ch_width']

	over_sampling = image_params['over_sampling']
	kernel_support = image_params['kernel_support']
	ff_size = image_params['ff_size']

	#pswf  = GK.anti_aliasing_uv()
	pswf = GK.StaleyKernel(kernel_support,over_sampling)

	if image_params['kernel'] == 'awkernel':
		Mterms,Mterms_ij = GF.MuellerTerms(image_params,ref_freq,ch_width)
		dty_image,psf_image = GF.awGrid(vis,HA,uvw,image_params,obs_params,Mterms,Mterms_ij)

	elif image_params['kernel'] == 'pswf':
		dty_image,psf_image = GF.Griddding(vis,uvw,image_params,obs_params,pswf)

	fov = (image_params['imageSize'] * image_params['cellsize'])/3600.
	imagename = files['wdir']+files['imagename']


	psf_peak = np.amax((psf_image.real))
	corr_dirty_image = (dty_image.real)/psf_peak#/corrFunc

	#obtain the clean beam
	clBeam, psf_fit = FM.cleanBeam(psf_image,image_params)#/psf_peak

	#print psf_fit


	FM.writetoFITS_scratch(imagename +'_dirtyimage_real.fits', corr_dirty_image, fov, Array, hduList, psf_fit, imageType='dirtyimage')
	FM.writetoFITS_scratch(imagename +'_dirtypsf_real.fits',psf_image.real/psf_peak,fov,Array,hduList,psf_fit, imageType='dirtyimage')

	#determine image peak and rms
	#image_peak = np.amax(np.abs(corr_dirty_image))
	#peak = image_peak/psf_peak
	#image rms
	#rms = np.std(np.abs(image))
	#print 'Dirty Image peak = %s, and rms = %s' % (str(peak), str(rms))
