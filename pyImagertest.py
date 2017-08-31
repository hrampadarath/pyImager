#!/usr/bin/env python


"""
SYNOPSIS

    python pyImagerAproj.py [-h,--help] [-v,--verbose] [-f,--file] [--version]

DESCRIPTION
	
    Imager for radio interferometric data. The input data must be in FITS-IDI or UVFITS format
    This version builds upon pyImagerv2.py the currently working version) and introduces  
    the Akernel (for only the diagonals XX and YY)
    Dependencies (codes):
	makeImage.inputs 	- Image inputs 
	AntennaFunctions.py	- Functions relating to the primary beam and Mueller functions
	GriddingKernels.py	- related to gridding and kernel generation and extration
	FITSmodules.py		- reading and manipulating uvfits/fitstidi files and writing to FITS image for pyImager
	HogbomClean.py		- My implementation of the Hogbom clean Algorithm
	Imagefuncs.py		- Collection of important functions written for the pyImager and vis-simulator
    Dependencies :
	astropy, pyephem, numpy, scipy and matplotlib (not sure why)

EXAMPLES

    The imager requires an input file. The default is makeImage.inputs
    e.g.
    python pyImagerAproj.py -v -f makeImage.inputs
    python pyImagerAproj.py --verbose --file makeImage.inputs

EXIT STATUS

    TODO: List exit codes

AUTHOR

    Hayden Rampadarath <hayden.rampadarath@manchester.ac.uk && haydenrampadarath@gmail.com>

LICENSE

    This script is in the public domain, free from copyrights or restrictions.

VERSION

    3

NOTES
	This file was generated using the LoadPythonTemplate.py script using the pythonTemplate.py in ~/scripts

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
import time
import warnings


#######USER DEFINED FUNCTIONS
sys.path.append('./synthesis')
import Imagefuncs as IF
import FITSmodules as FM
from imaging_test import *

#Define global constants
global EARTH_RADIUS, LIGHT_SPEED 
EARTH_RADIUS = 6371 * 10**3     # Earth's radius
LIGHT_SPEED = 299792458         # Speed of light






def main(inFile):

	start = time.time()

	#####################################################################
	

	print '------Using inputs specified in', inFile
	if os.path.isfile(inFile) :
		control = IF.parse_inp(inFile)
	else :
		print "Error:" + inFile + "does not exist, quitting."
		sys.exit()


	#------Defining the Imaging Parameters----------#



	Stokes		= control['Stokes'][0]
	imageSize 	= int(control['imageSize'][0])
	cellsize  	= float(control['cellsize'][0])
	wplanes   	= int(control['wplanes'][0])
	niter 		= int(control['niter'][0])
	nmajor		= int(control['nmajor'][0])
	loopGain 	= float(control['loopGain'][0])
	tinc 		= float(control['tinc'][0])
	thresh		= float(control['threshold'][0])

	over_sampling 	= int(control['over_sampling'][0])
	ff_size 	= int(control['ff_size'][0])
	kernel_support 	= int(control['kernel_support'][0])
	kernel 		= control['kernel'][0]
	wdir 		= control['wdir'][0]
	"""
	if kernel == 'awkernel':
		imagename 	= '%s_%s_%s_mins_niter%s_nmaj%s' % (control['imagename'][0],kernel, str(tinc),str(niter), str(nmajor))
	elif kernel == 'wkernel':
		imagename 	= '%s_%s_niter%s_nmaj%s' % (control['imagename'][0],kernel, str(niter), str(nmajor))
	"""
	imagename 	= control['imagename'][0]
	uvfitsfile  = control['uvfitsfile'][0]
	Acache		= control['Acache'][0]




	##########################################################

	image_params = {'Stokes':Stokes, 
			'imageSize':imageSize, 
			'cellsize':cellsize, 
			'over_sampling':over_sampling, 
			'ff_size':ff_size, 
			'kernel_support':kernel_support, 
			'wplanes':wplanes, 
			'tinc':tinc, 
			'niter':niter,
			'thresh':thresh,
			'nmajor':nmajor, 
			'kernel':kernel,
			'Acache':Acache,
			'loopGain':loopGain}


	print '###################################################'
	print '--------------Imaging Parameters-------------------'
	print image_params
	print '###################################################'

	files = {'wdir':wdir,
		'uvfitsfile':uvfitsfile,
		'imagename':imagename}

	print '###################################################'
	print '--------------File Names-------------------'
	print files
	print '###################################################'


	#-------Starting the Imaging Algorithm-------------

	Imager(image_params,files)

	end = time.time()
	print '############# Time taken = %0.4f mins #####################' %((end-start)/60.)

def Imager(image_params,files):

	"""
	The main imaging code
	"""

	#########PART 1. Read the uvfits and do some bookkeeping##############
	#########Obtain visibilities, UVW, Hour Angle range, stokes etc#######
	#This section have been tested and rewritten many time over
	#current version has only been tested on a  single freq channel
	#but should work on multiple channel data

	#a. Read the uvfits
	hduList= FM.readFITS(files)

	#b. Get uvw, time, freq etc info
	vv, uu, ww, vis_time, vis_date, ref_freq, ch_width, num_chans, RA, DEC = FM.getUVWData(hduList)
	#print 'ref freq =', ref_freq

	
	#c. get the array info
	Array = FM.Observer(hduList) 
	
	#d. convert time to hour angle
	HA = FM.timetoHA(vis_time,vis_date, Array, RA)
	# ha range
	hastart = HA[0]
	haend   = HA[len(HA)-1]
	haLims = (hastart,haend)

	obs_params = {'ref_freq': ref_freq,
		      'ch_width': ch_width,
		      'num_chans': num_chans,
		      'RA': RA,
		      'DEC':DEC,
		      'lat':np.degrees(Array.lat),
		      'haLims':haLims}
	
	# e. extract the visibillities related to the Stokes

	#####extract the parallel hand polarisation visibilities
	#TODO: extract the cross-hands

	stk1_flux, stk2_flux, stk1_Name, stk2_Name = FM.getIVis(hduList)#

	#f. ----------conjugate symmetry-------#
	
	uvw, HA = FM.conjugateSymmetryUVW(uu,vv,ww,HA)
	xvis = FM.conjugateSymmetryVis(stk1_flux)
	yvis = FM.conjugateSymmetryVis(stk2_flux)
	
	
	vis = np.array([xvis,yvis])

	########PART 2 & 3: Gridding/deconvolution and generating the final image 
	#i.e. cleaning to obtain the final image
	#rewrote to incoporate a Cotton-Schwab style clean
	#current version only works for single polarisation and single channel

	Stokes = image_params['Stokes'] 
	if Stokes == 'I':

		if image_params['niter'] > 0:
			print 'Nope ...'
			sys.exit 

		elif image_params['niter'] == 0:
			print 'no cleaning, generating dirty image'
			makeDirtyImageFITS(vis,HA,uvw,image_params,obs_params,files,Array,hduList)

			#Staley_Gridder(vis,HA,uvw,image_params,obs_params)

			#dty_image, psf_image = grid(vis,uvw,image_params,obs_params)

		else:
			print 'no cleaning ... exiting'
			sys.exit()


	elif Stokes == 'Q':
		print 'Stokes %s not yet implemented 1' % (Stokes)
		sys.exit()
	elif (Stokes == 'U') or (Stokes == 'V'):
		print 'Stokes %s not yet implemented 2' % (Stokes)
		sys.exit()


	print '-------------PyImager appears to have ended successfully-----------------'




if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=globals()['__doc__'], version='$Id$')
        parser.add_option ('-v', '--verbose', action='store_true', default=False, help='verbose output')
	parser.add_option("-f", "--file", dest="filename",help="read data from FILENAME")
        (options, args) = parser.parse_args()
        if options.verbose: print time.asctime()

	if options.filename:
		print "reading %s..." % options.filename
        	main(options.filename)

        if options.verbose: print time.asctime()
        if options.verbose: print 'TOTAL TIME IN MINUTES:',
        if options.verbose: print (time.time() - start_time) / 60.0
        sys.exit(0)
    except KeyboardInterrupt, e: # Ctrl-C
        raise e
    except SystemExit, e: # sys.exit()
        raise e
    except Exception, e:
        print 'ERROR, UNEXPECTED EXCEPTION'
        print str(e)
        traceback.print_exc()
        os._exit(1)	


