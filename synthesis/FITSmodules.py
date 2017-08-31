"""
Python functions for reading and manipulating uvfits/fitstidi files
and writing to FITS image for pyImager

Hayden Rampadarath 2016
hayden.rampadarath@manchester.ac.uk or haydenrampadarath@gmail.com
"""


import sys,os,traceback, optparse
import time, ephem
import re, math
import numpy as np
import numpy.ma as ma
from astropy.io import fits
from astropy.time import Time
from scipy import signal
from astropy.modeling import models, fitting
import warnings
import matplotlib.pyplot as plt
import os.path
import Deconvolution as DC
import Imagefuncs as IF




def readFITS(files):
	
	"""
	read fits-idi file as hdulist
	"""
	
	uvfitsfile = files['uvfitsfile']
	hduList = fits.open(uvfitsfile)

	#FITSTYPE = detFITSTYPE(hduList)
		
	return hduList


def detFITSTYPE(hduList):

	if hduList[0].header.get('ORIGIN',"empty") == 'casacore':
		FITSTYPE = 'UVFITS' # AIPS and CASA uvfits
	if (hduList[0].header.get('OBSCODE',"empty") == 'DVA1_VLA_A01') or  (hduList[0].header.get('OBSCODE',"empty") == 'pySimulator'):
		FITSTYPE = 'FITSIDI' # fits-idi

	print 'Visibility data type = ', FITSTYPE

	return FITSTYPE


def getUVWData(hduList):
	
	"""
	Obtain from hdulist:
	uvw coordinates, time and frequency of all complex visibilities.	
	"""

	print 'Obtaining uvw coords, time and freq data'


	FITSTYPE = detFITSTYPE(hduList)

	if FITSTYPE == 'FITSIDI':

		#print hduList.info()
		UV_DATA = hduList[5].data
		UV_HEADER = hduList[5].header

		#Obtain Frequency Values
		#Note: this is fine, using a single freq value for the single channel case. 
		#For multiple chans, will need to obtain the freq of each visibility

		ref_freq = UV_HEADER['REF_FREQ']
		ch_width = UV_HEADER['CHAN_BW']
		num_chans = UV_HEADER['NO_CHAN']
		
		#Get UVW data. Note the U,V and W coordinates are in seconds, needs to be converted to m
 
		uu = UV_DATA.field(0) * ref_freq # in wavelengths
		vv = UV_DATA.field(1) * ref_freq # in wavelengths
		ww = UV_DATA.field(2) * ref_freq
		date = np.float64(UV_DATA.field(3)) # Julian date at 0 hours
		time = np.float64(UV_DATA.field(4)) # time elapsed since 0 hours in day fractions
		#TODO: determine the frequency of each visibility

		RA = hduList[4].data['RAEPO'][0]/15. # RA in hrs
		DEC = hduList[4].data['DECEPO'][0] # declination in degrees


	elif FITSTYPE == 'UVFITS':


		UV_DATA = hduList[0].data
		#UV_HEADER = hduList[0].header

		#Obtain Frequency Values
		#Note: this is fine, using a single freq value for the single channel case. 
		#For multiple chans, will need to obtain the freq of each visibility

		ref_freq = hduList[0].header['RESTFREQ']
		ch_width = hduList[1].data['CH WIDTH'][0]
		num_chans = ch_width/hduList[1].data['TOTAL BANDWIDTH'][0]
		
		#Get UVW data. Note the U,V and W coordinates are in seconds, needs to be converted to m
 
		uu = UV_DATA.field(0) * ref_freq # in wavelengths
		vv = UV_DATA.field(1) * ref_freq # in wavelengths
		ww = UV_DATA.field(2) * ref_freq
		date = np.float64(UV_DATA.field(3)) # Julian date at 0 hours
		time = np.float64(UV_DATA.field(4)) # time elapsed since 0 hours in day fractions
		#TODO: determine the frequency of each visibility

		RA = hduList[3].data['RAEPO'][0]/15. # RA in hrs
		DEC = hduList[3].data['DECEPO'][0] # declination in degrees


	return vv, uu, ww, time, date, ref_freq, ch_width, num_chans, RA, DEC





def getIVis(hduList):
	
	"""
	extract the polarisations that corresponds to the I polarisation
	if Linear feeds - will return XX and YY polarisations
	if Circular will return RR and LL
	"""

	FITSTYPE = detFITSTYPE(hduList)

	if FITSTYPE == 'FITSIDI':

		print 'extracting polarisation visibilities'
		UV_DATA = hduList[5].data
		UV_HEADER = hduList[5].header
		
		nStokes = UV_HEADER['NO_STKD']
		STK_1 = UV_HEADER['STK_1']
		# Numeric Codes for Stokes Parameters:
        	stk_params = {1:'I', 2:'Q', 3:'U', 4:'V', -1:'RR', -2:'LL', -3:'RL', -4:'LR', -5:'XX', -6:'YY', -7:'XY', -8:'YX'}
		#parallel hands
		stk_1 = np.zeros((len(UV_DATA)), dtype=np.complex64)
		stk_2 = np.zeros((len(UV_DATA)), dtype=np.complex64)
		stk1_Name = stk_params[STK_1]
		stk2_Name = stk_params[STK_1 - 1]

		stk_1.real = UV_DATA.field(10)[:,0]
		stk_1.imag = UV_DATA.field(10)[:,1]
		stk_2.real = UV_DATA.field(10)[:,2]
		stk_2.imag = UV_DATA.field(10)[:,3]
		"""
		#cross hands
		stk_3 = np.zeros((len(UV_DATA)), dtype=np.complex64)
		stk_4 = np.zeros((len(UV_DATA)), dtype=np.complex64)
		stk3_Name = stk_params[STK_1 - 2]
		stk4_Name = stk_params[STK_1 - 3]

		stk_3.real = UV_DATA.field(10)[:,4]
		stk_3.imag = UV_DATA.field(10)[:,5]
		stk_4.real = UV_DATA.field(10)[:,6]
		stk_4.imag = UV_DATA.field(10)[:,7]
		"""
		
		
	elif FITSTYPE == 'UVFITS':


		print 'extracting polarisation visibilities'
		UV_DATA = hduList[0].data
		DATA    = UV_DATA.field(9)[:,0][:,0][:,0][:,0]
		UV_HEADER = hduList[0].header
		
		nStokes = UV_HEADER['NAXIS3'] #number of stokes
		STK_1 = UV_HEADER['CRVAL3'] # 1st stokes number
		# Numeric Codes for Stokes Parameters:
        	stk_params = {1:'I', 2:'Q', 3:'U', 4:'V', -1:'RR', -2:'LL', -3:'RL', -4:'LR', -5:'XX', -6:'YY', -7:'XY', -8:'YX'}

		stk_1 = np.zeros((len(UV_DATA)), dtype=np.complex64)
		stk_2 = np.zeros((len(UV_DATA)), dtype=np.complex64)
		stk1_Name = stk_params[STK_1]
		stk2_Name = stk_params[STK_1 - 1]

		#for i in range(len(UV_DATA)):
		stk_1.real = DATA[:,0][:,0]
		stk_1.imag = DATA[:,0][:,1]
		stk_2.real = DATA[:,1][:,0]  
		stk_2.imag = DATA[:,1][:,1] 
		
	
                       
	return stk_1, stk_2, stk1_Name, stk2_Name			
	

def getFullPolarisations():
	"""
	Return all 4 polarisation visibilities
	Useful for full Stokes imaging
	TODO 
	"""
	pass




def Observer(hduList):

	"""
	Return the array as a pyephem object
	currenty supports only the JVLA 
	"""

	FITSTYPE = detFITSTYPE(hduList)
	print 'getting Array info'
	if FITSTYPE == 'FITSIDI':
		arrayHdu = hduList[1].header

	elif FITSTYPE == 'UVFITS':
		arrayHdu = hduList[2].header

	arrayName = arrayHdu['ARRNAM']
	obsDate = arrayHdu['RDATE'][0:10].replace("-","/")
	Array = ephem.Observer()
	
	arrayList = {'VLA':{'Latitude':'34:04:43.497', 'Longitude':'-107:37:05.819', 'Elevation':2124},
		     'EVLA':{'Latitude':'34:04:43.497', 'Longitude':'-107:37:05.819', 'Elevation':2124},
		     'JVLA':{'Latitude':'34:04:43.497', 'Longitude':'-107:37:05.819', 'Elevation':2124}}
	if arrayName in arrayList.keys():
		Array.lon = arrayList[arrayName]['Longitude']
		Array.lat = arrayList[arrayName]['Latitude']
		Array.elevation = arrayList[arrayName]['Elevation']
		Array.date = obsDate

	return Array



def timetoHA(vis_time, vis_date, Array, RA):
	"""
	Converting julian date to HA. 
	Copied from 
	http://uk.mathworks.com/matlabcentral/fileexchange/28233-convert-eci-to-ecef-coordinates/content/JD2GMST.m
	and http://www.csgnetwork.com/siderealjuliantimecalc.html
	and http://www.cv.nrao.edu/~rfisher/Ephemerides/times.html#LMST
	"""

	print 'converting normal time to HA'
	#print Array.lon
	Long = np.degrees(float((Array.lon))) # array longitude
	HA = []
	for i in range (len(vis_time)):
		JD = vis_date[i] + vis_time[i]
		#1. determine the Julian date of the previous midninght, JD0
		JD0_max = np.floor(JD)+0.5
		if JD > JD0_max:
			JD0 = JD0_max
		elif JD < JD0_max:
			JD0 = np.floor(JD)-0.5
		H = (JD-JD0)*24.      	#Time in hours past previous midnight
		D  = JD - 2451545.0    #Compute the number of days since J2000
		D0 = JD0 - 2451545.0   #Compute the number of days since J2000
		T  = D/36525.          #Compute the number of centuries since J2000
		#Calculate GMST in hours (0h to 24h)
		GMST = ((6.697374558 + 0.06570982441908*D0  + 1.00273790935*H + 0.000026*(T**2)) % 24. )
		LMST = GMST + (Long/15.) # determine LMST in hrs
		HA.append(LMST-RA) #calculate and append the HA in hrs
	return HA
	
	



def conjugateSymmetryUVW(uu,vv,ww,HA):
	"""
	Apply the conjugate symmetry to the visibility data
	"""
	
	uvw = np.column_stack((uu,vv,ww))#column_stack - Stack 1-D arrays as columns into a 2-D array.
	#get conjugate symmetry
	tmp_uvw = uvw*np.array([-1.,-1.,1.])
	uvw = np.vstack((uvw,tmp_uvw))#vstack - Stack arrays in sequence vertically (row wise).

	tmp_HA = HA
	HA = np.hstack((HA,tmp_HA))
	
	
	return uvw, HA
	

def conjugateSymmetryVis(vis):

	tmp_vis = np.conj(vis)
	vis = np.hstack((vis,tmp_vis))
	
	return vis


def conjugateSymetry(uvw,vis):
	
	"""
	first do X pol and then Y pol. stitch after!
	"""

	xvis = vis[0]
	yvis = vis[1]

	tmp_uvw = uvw*np.array([-1.,-1.,1.])
	uvw = np.vstack((uvw,tmp_uvw))#vstack - Stack arrays in sequence vertically (row wise).
	print 'uvw shape = ', np.shape(uvw)
	#zs=np.argsort(uvw[:,2])
	#uvw = uvw[zs]
	tmp_xvis = np.conj(xvis)
	xvis = np.hstack((xvis,tmp_xvis))
	tmp_yvis = np.conj(yvis)
	yvis = np.hstack((yvis,tmp_yvis))

	vis = np.array([xvis,yvis])
	
	print 'vis shape = ', np.shape(vis)
	
	return uvw, vis

def writetoFITS(fitsname,imageArrayA,FOV,Array,hduList,imageType='image', Stokes='I'):

	"""
	save numpy array image to fits image
	fitsname - ouput name 
	imageArrayA - 2d image array
	FOV - field of view in degrees
	Array - info about the array as an ephem object
	hduList - the visibility data (output from astropy pyfits)
	imageType - only image and psf supported
	"""
	#pass
	fitsname = checkfileexists(fitsname)



	#--------------Write data to FITS---------------------------
	if imageType == 'dirtyimage':
		imageArray = IF.row2colslr(np.abs(imageArrayA))
	else:
		imageArray = IF.row2cols(np.abs(imageArrayA)) # convert from rows to columns 

	hdu = fits.PrimaryHDU(imageArray)
	hdu.writeto(fitsname)
	
	#------------reload fits file to edit the header-------------
	data,header = fits.getdata(fitsname,header=True)
	
	#---------------Load image template--------------
	#easier to copy an exiting image header and chnage params, than to write one from scratch.
	#

	if (imageType == 'image') or (imageType == 'dirtyimage'):
	
		imgTemplate = "FITS/ImageTemplate.fits"
		templatedata,templateheader = fits.getdata(imgTemplate,header=True)
	if imageType == 'psf':
		imgTemplate = "FITS/psfTemplate.fits"
		templatedata,templateheader = fits.getdata(imgTemplate,header=True)

	
	#essentially copy the header from the template to the generated primary beam, however ... 
	#change a couple of entries that are specifc to the generated beam such as
	#NAXIS, NAXIS1, NAXIS2 
	#get info about the observation from the the visibility dataset
	XYZ, RA, DEC, freq, date, ch_width = getFITSheaderInfo(hduList)
	
	templateheader['CRPIX1'] = header['NAXIS1']/2. # change RA location of central pixel
	templateheader['CRPIX2'] = header['NAXIS2']/2. # change DEC location of central pixel
	
	DEG_PER_PIX = FOV/header['NAXIS1']
	templateheader['CDELT1']= -1 * DEG_PER_PIX
	templateheader['CDELT2']= DEG_PER_PIX

	#update info regarding the observation 
	#i.e. longitude, latitude, frequency, pointing centre, Stokes, date
	#
	templateheader['LONPOLE'] = -1 * np.degrees(float(repr(Array.lon))) # longitude of array
	templateheader['LATPOLE'] = np.degrees(float(repr(Array.lat))) # latitude of Array
	templateheader['CRVAL1']  = RA # RA of pt centre
	templateheader['CRVAL2']  = DEC # dec of pt centre
	templateheader['OBSRA']   = RA # RA of pt centre
	templateheader['OBSDEC']  = DEC # dec of pt centre
	templateheader['OBSGEO-X'] = XYZ[0] # 
	templateheader['OBSGEO-Y'] = XYZ[1] #
	templateheader['OBSGEO-Z'] = XYZ[2] #
	templateheader['DATE-OBS'] = date#Observation date
	templateheader['CRVAL3'] = freq # in Hz Central frequency
	templateheader['RESTFRQ'] = freq #Central frequency
	templateheader['HISTORY'] = 'FITSIMAGE made with pyImager'
	templateheader['DATE'] = time.asctime() #Date FITS file was written
	
	stk_params = {1:'I', 2:'Q', 3:'U', 4:'V', -1:'RR', -2:'LL', -3:'RL', -4:'LR', -5:'XX', -6:'YY', -7:'XY', -8:'YX'}
	for keys in stk_params:
		if stk_params[keys] == Stokes:
			templateheader['CRVAL4'] = keys #image stokes
	
	#TODO:get and update beam size
	
	#write the header to the fits file
	print 'Writing image file to: %s' % (fitsname)
	
	fits.writeto(fitsname, data, templateheader, clobber=True)







def writetoFITS_scratch(fitsname, imageArrayA, FOV, Array, hduList, psf_fit, imageType='image', Stokes='I'):

	"""
	save numpy array image to fits image
	build the header rom scratch
	fitsname - ouput name 
	imageArrayA - 2d image array
	FOV - field of view in degrees
	Array - info about the array as an ephem object
	hduList - the visibility data (output from astropy pyfits)
	imageType - only image and psf supported
	"""
	#pass
	fitsname = checkfileexists(fitsname)



	print('--------------Saving image to FITS---------------------------')
	#"""
	if imageType == 'dirtyimage':
		imageArrayA = IF.row2colslr(np.abs(imageArrayA))
	else:
		imageArrayA = IF.row2colslr(np.abs(imageArrayA)) # convert from rows to columns 
	#"""

	hdu = fits.PrimaryHDU(imageArrayA)
	hdu.writeto(fitsname)
	
	#------------reload fits file to edit the header-------------
	data,header = fits.getdata(fitsname,header=True)

	DEG_PER_PIX = FOV/header['NAXIS1']
	
	#get info about the observation from the the visibility dataset
	XYZ, RA, DEC, freq, date, ch_width = getFITSheaderInfo(hduList)
	#header['NAXIS'] = 4
	#header['NAXIS3'] = 1
	#header['NAXIS4'] = 1
	header['EXTEND'] = 'T'
	header['BSCALE'] = 1.000000000000E+00
	header['BZERO']  = 0.000000000000E+00
	####clean beam size get from the fitting
	header['BMAJ'] = psf_fit['bmaj']/3600
	header['BMIN'] = psf_fit['bmin']/3600
	header['BPA'] = psf_fit['bpa']
	header['BTYPE']  = 'Intensity'
	header['OBJECT'] = 'ImS     '
	header['BUNIT']  = 'Jy/beam '
	header['EQUINOX'] = 2.000000000000E+03
	header['RADESYS'] = 'FK5     '
	header['LONPOLE'] = 180
	header['LATPOLE'] = 45
	# Pixel Coordinate (PC) matrix to transform between the 
	#FITS array axes and axes in the direction of the physical coordinate system but on the array scale
	header['PC01_01'] = 1.000000000000E+00 
	header['PC02_01'] = 0.000000000000E+00
	header['PC01_02'] = 0.000000000000E+00
	header['PC02_02'] = 1.000000000000E+00

	header['CTYPE1'] = 'RA---SIN'
	header['CRVAL1'] = RA # RA of pt centre
	header['CDELT1'] = -1 * DEG_PER_PIX
	header['CRPIX1'] = header['NAXIS1']/2. # change RA location of central pixel
	header['CUNIT1'] = 'deg     '

	header['CTYPE2'] = 'DEC--SIN'
	header['CRVAL2'] = DEC # dec o0f pt centre
	header['CDELT2'] = DEG_PER_PIX
	header['CRPIX2'] = header['NAXIS2']/2. # change DEC location of central pixel
	header['CUNIT1'] = 'deg     '

	header['CTYPE3'] = 'FREQ    '
	header['CRVAL3'] = freq # in Hz Central frequency
	header['CDELT3'] = ch_width # freq increment in Hz
	header['CRPIX3'] = 1.000000000000E+00 
	header['CUNIT3'] = 'Hz      '

	header['CTYPE4'] = 'STOKES  '
	stk_params = {1.0:'I', 2:'Q', 3:'U', 4:'V', -1:'RR', -2:'LL', -3:'RL', -4:'LR', -5:'XX', -6:'YY', -7:'XY', -8:'YX'}
	for keys in stk_params:
		if stk_params[keys] == Stokes:
			header['CRVAL4'] = keys #image stokes
	header['CDELT4'] = 1.000000000000E+00
	header['CRPIX4'] = 1.000000000000E+00 
	header['CUNIT4'] = '        '

	header['RESTFRQ'] = freq #Central frequency
	header['SPECSYS'] = 'TOPOCENT'
	
	header['OBSRA']   = RA # RA of pt centre
	header['OBSDEC']  = DEC # dec of pt centre
	header['OBSGEO-X'] = XYZ[0] # 
	header['OBSGEO-Y'] = XYZ[1] #
	header['OBSGEO-Z'] = XYZ[2] #
	header['DATE-OBS'] = date#Observation date
	header['HISTORY'] = 'FITSIMAGE made with pyImager'
	header['DATE'] = time.asctime() #Date FITS file was written
	header['OBSERVER'] = 'HRAMPADARATH'
	header['ORIGIN']  = 'pyImager '

	
	#TODO:get and update beam size
	
	#write the header to the fits file
	print 'Writing image file to: %s' % (fitsname)
	
	fits.writeto(fitsname, data, header, clobber=True)





def getFITSheaderInfo(hduList):
	"""
	obtain info specific to this observation/FITS
	from the parent UVFITS/FITS-IDI data.
	"""	

	print 'getting fits header info'
	FITSTYPE = detFITSTYPE(hduList)
	if FITSTYPE == 'FITSIDI':
		arrayGEO = hduList[1].header
		freq = hduList[0].header['REF_FREQ']
		ch_width = hduList[5].header['CHAN_BW']
		RA = hduList[4].data['RAEPO'][0]
		DEC = hduList[4].data['DECEPO'][0]
		year,month,day = IF.jd_to_date(hduList[5].data[0][4]+hduList[5].data[0][3])
		#print 'year = %s, month = %s, day = %s' %(str(year),str(month),str(day))
		date = IF.fitsdate(year,month,day) 

	if FITSTYPE == 'UVFITS':
		arrayGEO = hduList[2].header
		freq = hduList[0].header['RESTFREQ']
		ch_width = hduList[1].data['CH WIDTH'][0]
		RA = hduList[3].data['RAEPO'][0]
		DEC = hduList[3].data['DECEPO'][0]
		date = hduList[2].header['RDATE']


	XYZ = (arrayGEO['ARRAYX'],arrayGEO['ARRAYY'],arrayGEO['ARRAYZ']) # array central XYZ coordinates



	return XYZ, RA, DEC, freq, date, ch_width





def makeFitsImages(cc,rres,dirty_psf,Array,hduList,image_params,files):



	fov = (image_params['imageSize'] * image_params['cellsize'])/3600.
	imagename = files['wdir']+files['imagename']

	Stokes = image_params['Stokes']

	#obtain the clean beam
	clBeam, psf_fit = cleanBeam(dirty_psf,image_params)#/psf_peak
	#peak psf value
	psfpeak = DC.argmax(np.abs(clBeam))
	peak_psf = np.amax(np.abs(clBeam))
	clBeam  = clBeam/peak_psf 
	#normalise the residuals
	rres = rres/peak_psf 

	#obtain the clean image
	ycc,xcc = np.nonzero(cc)
	clIMG = np.zeros(cc.shape)
	for i in range (len(ycc)):
		mval = cc[ycc[i], xcc[i]]
		#print 'cc flux = %0.6f @ x = %s, y = %s)' % (mval, str(xcc[i]),str(ycc[i])) 
		a1o, a2o = DC.overlapIndices(cc, clBeam, ycc[i] - psfpeak[0],xcc[i] - psfpeak[1])
		clIMG[a1o[0]:a1o[1], a1o[2]:a1o[3]] += clBeam[a2o[0]:a2o[1], a2o[2]:a2o[3]] * mval

		
	
	
	cleanImage = clIMG #+ rres
	"""
	plt.clf()
	plt.subplot(131)
	plt.imshow(cleanImage)
	plt.title("clean")
	plt.subplot(132)
	plt.imshow(rres.real)
	plt.title("residual")
	plt.subplot(133)
	plt.imshow(cleanImage+rres.real)
	plt.title("clean + residual")
	#plt.show()
	plt.savefig(imagename+'_images.png')	
	"""
	#save to fits
	
	writetoFITS_scratch(imagename +Stokes+'_image.fits',cleanImage,fov,Array,hduList,psf_fit,imageType='image')
	writetoFITS_scratch(imagename +Stokes+'_residuals.fits',rres,fov,Array,hduList,psf_fit,imageType='image')
	writetoFITS_scratch(imagename +Stokes+'_image_res.fits',cleanImage+rres,fov,Array,hduList,psf_fit,imageType='image')

	#writetoFITS(imagename +'_cc.fits',cc,fov,Array,hduList,imageType='image')
	writetoFITS_scratch(imagename +'_dirtypsf.fits',dirty_psf/peak_psf,fov,Array,hduList,psf_fit,imageType='image')
	#writetoFITS(imagename +'_cleanpsf.fits',np.abs(clBeam),fov,Array,hduList,imageType='image')

	image_peak,image_rms,peak_resid,resid_rms = IF.imStats(cleanImage,dirty_psf,rres)
	print 'Image peak = %s, and rms = %s' % (str(image_peak), str(image_rms))
	print 'Peak residuals = %s, and residuals rms = %s' % (str(peak_resid), str(resid_rms))





def cleanBeam(dirtypsf,image_params):

	"""
	To simulate a clean psf (or clean beam)
	by fitting a 2D Gaussian to the main lobe of the dirty beam.
	"""

	PSF = dirtypsf.real
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

	#plt.imshow(cleanpsf)
	#plt.show()

	pS = 100 # radius in pixel of clean beam

	sub_psf = cleanpsf[cent-pS:cent+pS,cent-pS:cent+pS]
	pad_width = np.shape(PSF)[0]/2 - pS
	sub_psf = np.pad(sub_psf,(pad_width,pad_width),mode='constant')

	######get clean beam size###
	cellsize = image_params['cellsize']
	bmaj = max(g.x_stddev*2*cellsize,g.y_stddev*2*cellsize)
	bmin = min(g.x_stddev*2*cellsize,g.y_stddev*2*cellsize)

	beam = {'bmaj':bmaj, 'bmin': bmin, 'bpa': g.theta*1}


	return sub_psf, beam




def cleanBeam2(dirtypsf):

    """
    To simulate a clean psf (or clean beam)
    """

    cent = np.shape(dirtypsf)[0]/2
    pS = 7 # radius in pixel of clean beam

    sub_psf = dirtypsf[cent-pS:cent+pS,cent-pS:cent+pS]
    pad_width = np.shape(dirtypsf)[0]/2 - pS
    sub_psf = np.pad(sub_psf,(pad_width,pad_width),mode='constant')

    return sub_psf



"""
def checkfileexists(fname):

	if os.path.isfile(fname):
		cpfname = fname+'_copy'
		print '%s exists, will make a copy to %s' %(fname,cpfname)
		os.system('rm ' + fname)
		return cpfname
	else:
		return fname

"""


def checkfileexists(fname):

	if os.path.isfile(fname):
		print '%s exists, deleting' %(fname)
		os.system('rm ' + fname)
	return fname

