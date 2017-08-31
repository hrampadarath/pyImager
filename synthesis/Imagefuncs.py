"""
Collection of important functions written for the pyImager and vis-simulator

Hayden Rampadarath 2016
hayden.rampadarath@manchester.ac.uk or haydenrampadarath@gmail.com
"""

import re,sys
import numpy as np
import scipy.io
from scipy import interpolate, ndimage
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import math
#from astronomyv1 import deltasep, alphasep,hmstora,dmstodec,deltapos
import numpy.ma as ma

#some globals
deg2rad    = np.pi / 180
rad2deg    = 1/deg2rad
arcmin2rad = np.pi / (180 * 60)
arcsec2rad = np.pi / (180 * 60 * 60)
rad2arcmin = 1/(arcmin2rad)
rad2arcsec = 1/(arcsec2rad)


def parse_inp(filename):
	''' Parse the list of inputs given in the specified file. (Modified from evn_funcs.py, taken from eMERLIN_pipeline.py)'''
	INPUTFILE = open(filename, "r")
	control = dict()

	# a few useful regular expressions
	newline = re.compile(r'\n')
	space = re.compile(r'\s')
	char = re.compile(r'\w')
	comment = re.compile(r'#.*')

	# parse the input file assuming '=' is used to separate names from values
	for line in INPUTFILE:
		if char.match(line):
			line = comment.sub(r'', line)
			line = line.replace("'", '')
			(param, value) = line.split('=')

			param = newline.sub(r'', param)
			param = param.strip()
			param = space.sub(r'', param)

			value = newline.sub(r'', value)
			value = value.strip()
			valuelist = value.split(', ')
			control[param] = valuelist

	return control






#############codes taken from astronomyv1.py################

# Find angular separation of 2 positions, in arcseconds



def pad_fft(inparray):
	"""return a zero-padded array"""

	padarray= np.pad(inparray,pad_width=(np.shape(inparray)[0]/4,np.shape(inparray)[1]/4),mode='constant')
	return padarray

def alphasep(ra1,ra2,dec1,dec2):
    	"""Find the angular separation of two sources in RA, in arcseconds

	Keyword arguments:
	ra1,dec1 - RA and Dec of the first source, in decimal degrees
	ra2,dec2 - RA and Dec of the second source, in decimal degrees

	Return value:
	angsep - Angular separation, in arcseconds

	"""

	return 3600*(ra1-ra2)*math.cos(math.radians((dec1+dec2)/2.0))


def deltasep(dec1,dec2):
	"""Find the angular separation of two sources in Dec, in arcseconds

	Keyword arguments:
	dec1 - Dec of the first source, in decimal degrees
	dec2 - Dec of the second source, in decimal degrees

	Return value:
	angsep - Angular separation, in arcseconds

	"""

	return 3600*(dec1-dec2)

# Find angular separation in Dec of 2 positions, in arcseconds


def hmstora(rah,ram,ras):
	"""Convert RA in hours, minutes, seconds format to decimal
	degrees format.

	Keyword arguments:
	rah,ram,ras -- RA values (h,m,s)

	Return value:
	radegs -- RA in decimal degrees

	"""
	radegs = 15*(float(rah)+(float(ram)/60)+(float(ras)/3600.0))
	
	return radegs
# Converts an hms format RA to decimal degrees

def dmstodec(decd,decm,decs):
	"""Convert Dec in degrees, minutes, seconds format to decimal
	degrees format.

	Keyword arguments:
	decd,decm,decs -- list of Dec values (d,m,s)

	Return value:
	decdegs -- Dec in decimal degrees

	"""
	sign = float(decd)
	if sign>0:
		decdegs = float(decd)+(float(decm)/60)+(float(decs)/3600.0)
	else:
		decdegs = float(decd)-(float(decm)/60)-(float(decs)/3600.0)
	
	return decdegs
# Converts a dms format Dec to decimal degrees



def deltapos(rah1,ram1,ras1,decd1,decm1,decs1,rah2,ram2,ras2,decd2,decm2,decs2):
	"""To calculate the difference between 2 positions given in the tradiational
	Ra and dec format
	"""
	#position 1 in degs
	ra1 = hmstora(rah1,ram1,ras1)
	dec1 = dmstodec(decd1,decm1,decs1)
	#position 2 in degs
	ra2 = hmstora(rah2,ram2,ras2)
	dec2 = dmstodec(decd2,decm2,decs2)
	#1 calculate the separation in delta in arcsecs
	deltadec = deltasep(dec1,dec2)
	#2 calculate separation in ra arcsecs
	deltara = alphasep(ra1,ra2,dec1,dec2)
	#3 calculate the positional difference arcsecs
	sourcesep = angsep(ra1,dec1,ra2,dec2)
	return deltara,deltadec,sourcesep


###########################################################




def parangle(HA, dec_d, lat_d):
    
    	"""
    	from p. 91 of  treatise on spherical astronomy" By Sir Robert Stawell Ball
    	this is how CASA does the pb rotation (thanks to Preshant)

	INPUTS:
        HA  - Hour angle of the object, in decimal hours (0,24)
        DEC - Declination of the object, in degrees
        lat - The latitude of the observer, in degrees

    	returns:
    	paralctic angle in degrees
    	"""
    	lat = np.radians(lat_d)
    	dec = np.radians(dec_d)   
    	HA = np.radians(HA*15.)
    	sin_eta_sin_z = np.cos(lat)*np.sin(HA)
    	cos_eta_sin_z = np.sin(lat)*np.cos(dec) - np.cos(lat)*np.sin(dec)*np.cos(HA)
    	
    	eta = np.arctan2(sin_eta_sin_z,cos_eta_sin_z)
    	return np.degrees(eta)







def rotate2d(degrees,point,origin):
	"""
	A rotation function that rotates a point around a point
	to rotate around the origin use [0,0]
	"""
	degrees = degrees * 1
	x = point[0] - origin[0]
	yorz = point[1] - origin[1]
	newx = (x*np.cos(np.radians(degrees))) - (yorz*np.sin(np.radians(degrees)))
	#print newx
	newyorz = (x*np.sin(np.radians(degrees))) + (yorz*np.cos(np.radians(degrees)))
	#print newyorz
	
	newx += origin[0]
	newyorz += origin[1]
	
	return newx,newyorz





def float2str(inarray):

	"""
	Converts a float array to an array of strings
	"""
	strs = []
	for n in inarray:
		strs.append(str(n))
	return strs




def jd_to_date(jd):
    	"""
    	Convert Julian Day to date. http://github.com/jiffyclub
    	
    	Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet', 
    	    4th ed., Duffet-Smith and Zwart, 2011.
    	
    	Parameters
    	----------
    	jd : float
    	    Julian Day
    	    
    	Returns
    	-------
    	year : int
    	    Year as integer. Years preceding 1 A.D. should be 0 or negative.
    	    The year before 1 A.D. is 0, 10 B.C. is year -9.
    	    
    	month : int
    	    Month as integer, Jan = 1, Feb. = 2, etc.
    	
    	day : float
    	    Day, may contain fractional part.
    	    
    	Examples
    	--------
    	Convert Julian Day 2446113.75 to year, month, and day.
    	
    	>>> jd_to_date(2446113.75)
    	(1985, 2, 17.25)
    	
    	"""
    	jd = jd + 0.5
    	
    	F, I = math.modf(jd)
    	I = int(I)
    	
    	A = math.trunc((I - 1867216.25)/36524.25)
    	
    	if I > 2299160:
    	    B = I + 1 + A - math.trunc(A / 4.)
    	else:
    	    B = I
    	    
    	C = B + 1524
    	
    	D = math.trunc((C - 122.1) / 365.25)
    	
    	E = math.trunc(365.25 * D)
    	
    	G = math.trunc((C - E) / 30.6001)
    	
    	day = C - E + F - math.trunc(30.6001 * G)
    	
    	if G < 13.5:
    	    month = G - 1
    	else:
    	    month = G - 13
    	    
    	if month > 2.5:
    	    year = D - 4716
    	else:
    	    year = D - 4715
    	    
    	return year, month, day


def dayfrac_to_time(dayfrac):

	"""
	determine the time from a day fraction
	"""
	hrs  = (dayfrac - np.floor(dayfrac)) * 24
	mins = (hrs - np.floor(hrs)) * 60
	sec  = (mins - np.floor(mins)) * 60

	return int(np.floor(hrs)), int(np.floor(mins)), int(np.floor(sec))


def row2cols(array):
	
	size = np.size(array)	
	invarray = np.flipud(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray


def row2colslr(array):
	
	size = np.size(array)	
	invarray = np.fliplr(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray



def fitsdate(yr,mo,d):

	hr = int(np.floor((d - np.floor(d)) * 24))
	mi = int(np.floor((((d - np.floor(d)) * 24) - hr)*60))
	#print 'month fitsdate = ', mo 
	s = (((((d - np.floor(d)) * 24) - hr)*60) - mi)*60
	d = int(np.floor(d))

	timefits = '%s-%s-%sT%s:%s:%0.4f' % (str(yr),str(mo),str(d),str(hr),str(mi),s)
	return timefits








def imStats(image,psf,residuals):

	#peak psf
	psf_peak = np.amax(np.abs(psf))
	#image peak
	image_peak = np.amax(np.abs(image))
	#peak residuals
	peak_resid = np.amax(np.abs(residuals))	
	#image rms
	image_rms = np.std(np.abs(image))
	#residuals rms
	resid_rms =  np.std(np.abs(residuals))

	return image_peak,image_rms,peak_resid,resid_rms
	




def imagestats(srclist,T2,imarray,psfarray):

	"""
	To estimate the rms in a simulated image given:
	srclist - list of sources used in the simulation, with
	l an m position (in radians) and amplitude
	Npix  - the total number of pixels on one side of an image
	T2    - half-width of FOV [radians]
	imagearray is the image as a numpy array
	"""
	#load srclist
	src = np.loadtxt(srclist,comments='#',dtype=float)
	
	Npix = (np.shape(imarray)[0])/2
	print 'Npix:', Npix
	rad2pix = Npix/T2
	print 'radians per pixel:', rad2pix

	#print 'sigma before mask = ', np.std(imarray)	
	pAmp = [] #peak source amplitudes
	#TAmp = [] #true source amplitudes
	mask = 10
	l = []
	m = []
	for i in range(len(src)):
		#find src position
		print src[i]
		#slm = RADEC2lmn(src[i])
		slm = offset(src[i])
		l.append(slm[0])
		m.append(slm[1])

		print 'l:', slm[0]
		print 'm:', slm[1]

		x = Npix + rad2pix * slm[0]  
		y = Npix + rad2pix * slm[1]

		print x,y
		#find pk amplitude of src
		subim = imarray[x-mask:x+mask, y-mask:y+mask]
		pAmp.append(np.amax(subim)/np.amax(np.abs(psfarray)))

		#TAmp.append(slm[2])
		imarray[x-mask:x+mask, y-mask:y+mask] = ma.masked
		plt.imshow((np.abs(imarray)))
		plt.show()
	#sigma = np.std(imarray)
	#print 'sigma after mask = ', sigma
	
	#plt.imshow(imarray)
	plt.imshow((np.abs(imarray)))
	plt.savefig('src_mask_dirtyimage.png')	

	plt.close('all')

	plt.plot(l,pAmp,'r*')
	plt.plot(m,pAmp,'bo')
	plt.xlabel('Phase Centre offset (radians)')
	plt.ylabel('Amplitude (Jy)')
	plt.savefig('Flux_vs_phase_offset.pdf')
	f = 'Flux_vs_phase offset_data.txt'
	np.savetxt(f,np.column_stack((pAmp,l,m)),newline='\n',fmt='%1.4e')
	
	#return sigma,pAmp,TAmp


def RADEC2lmn(src):

	"""
	convert RA & DEC to lmn coordinates
	src - source RA and DEC coordinate in the format:
	hh mm ss dd mm ss
	"""	
	#central corrdinates .... 
	RAc = np.array([5.,0.,0.])
	DECc = np.array([45,0.,0.])
	
	
	x = (np.cos((hmstora(src[0],src[1],src[2])*deg2rad) 
		- (hmstora(RAc[0],RAc[1],RAc[2])*deg2rad))
		* np.cos((dmstodec(src[3],src[4],src[5])*deg2rad)))
	
	y = (np.sin((hmstora(src[0],src[1],src[2])*deg2rad) 
		- (hmstora(RAc[0],RAc[1],RAc[2])*deg2rad))
		* np.cos((dmstodec(src[3],src[4],src[5])*deg2rad)))
	
	z = np.sin(dmstodec(src[3],src[4],src[5])*deg2rad)
	xyz = np.array([x,y,z])
	
	lmn = computeUVW(xyz,0,dmstodec(DECc[0],DECc[1],DECc[2])*deg2rad)
	l = lmn[0]
	m = lmn[1]
	n = lmn[2]
		
	return x,y


def offset(src):
	RAc = np.array([5.,0.,0.])
	DECc = np.array([45,0.,0.])
	
	dx,dy,sep = deltapos(RAc[0],RAc[1],RAc[2],DECc[0],DECc[1],DECc[2],src[0],src[1],src[2],src[3],src[4],src[5])
	# convert from arsecs to radians
	l = np.radians(dx/3600.)
	m = np.radians(dy/3600.)

	return l,m



def computeUVW(xyz,H,d):
	""" Converts X-Y-Z coordinates into U-V-W
	
	Uses the transform from Thompson Moran Swenson (4.1, pg86)
	
	Parameters
	----------
	xyz: should be a numpy array [x,y,z]
	H: float (degrees)
	  is the hour angle of the phase reference position
	d: float (degrees)
	  is the declination
	"""
	sin = np.sin
	cos = np.cos
	xyz = np.matrix(xyz) # Cast into a matrix

	
	trans= np.matrix([
	  [sin(H),         cos(H),        0],
	  [-sin(d)*cos(H), sin(d)*sin(H), cos(d)],
	  [cos(d)*cos(H), -cos(d)*sin(H), sin(d)]
	])
	
	uvw = trans * xyz.T
	
	uvw = np.array(uvw)
	
	return uvw[:,0]






def uvTapering(uvw,vis,L2):

	"""
	restrict uvw to a specified maximum value, L2
	"""
	
	u = uvw[:,0]
	v = uvw[:,1]
	w = uvw[:,2]
	#print uvw
		
	
	
        for n in reversed(range(len(u))):
                if np.abs(u[n]) > L2:
			#print u[n]
                        u = np.delete(u,n,axis=0)
                        v = np.delete(v,n,axis=0)
                        w = np.delete(w,n,axis=0)
			vis = np.delete(vis,n,axis=0)
	
	#"""
	for i in reversed(range(len(v))):
                if np.abs(v[i]) > L2:
                        v = np.delete(v,i,axis=0)
                        u = np.delete(u,i,axis=0)
                        w = np.delete(w,i,axis=0)
			vis = np.delete(vis,i,axis=0)
	#"""
	uvw = np.column_stack((u,v,w))
	#print uvw
	#sys.exit()

        return uvw, vis


def plot2Dimage(array2D,name):

	plt.imshow(np.abs(array2D))
	plt.savefig(name+'.png')


def sortw(uvw,vis):
	""" Sort the uvw and/or the visibilities according to the w-term """

	zs=np.argsort(uvw[:,2])
	if vis is not None:
		return uvw[zs], vis[zs]
	else:
		return uvw[zs]



