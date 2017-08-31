#!/usr/bin/env python


"""
SYNOPSIS

    TODO helloworld [-h,--help] [-v,--verbose] [--version]

DESCRIPTION

    TODO This describes how to use this script. This docstring
    will be printed by the script if there is an error or
    if the user requests help (-h or --help).

EXAMPLES

    TODO: Show some examples of how to use this script.

EXIT STATUS

    TODO: List exit codes

AUTHOR

    Hayden Rampadarath <hayden.rampadarath@manchester.ac.uk && haydenrampadarath@gmail.com>

LICENSE

    This script is in the public domain, free from copyrights or restrictions.

VERSION

    $Id$

NOTES
	This file was generated using the LoadPythonTemplate.py script using the pythonTemplate.py in ~/scripts

"""




#Some common imported modules - uncomment as required

import sys,os,traceback, optparse
import time
import re
#import matplotlib.pyplot as plt
import numpy as np
from astronomyv1 import deltasep, alphasep,hmstora,dmstodec,deltapos

#Define global constants


#Some useful Global args
deg2rad    = np.pi / 180
rad2deg    = 1/deg2rad
#arcmin2rad = np.pi / (180 * 60)
#arcsec2rad = np.pi / (180 * 60 * 60)
#rad2arcmin = 1/(arcmin2rad)
#rad2arcsec = 1/(arcsec2rad)



def main():

	S = 'grid_nopol_3src_1Jy'
	src = np.loadtxt(S,comments='#')
	RAc = np.array([5.,0.,0.])
	DECc = np.array([45,0.,0.])
	


	Npix = 1500
	T2 = 0.0363610260832
	rad2pix = Npix/T2
	print 'radians per pixel:', rad2pix

	for i in range(len(src)):
		#find src position
		print src[i]
		slm = RADEC2lmn(src[i])
		#slm = offset(src[i])

		print 'l:', slm[0]
		print 'm:', slm[1]
		print 'xoffset:', slm[0] * rad2pix 
		print 'yoffset:', slm[1] * rad2pix 

		x = Npix + slm[0] * rad2pix  
		y = Npix + slm[1] * rad2pix  

		print x,y


		


def RADEC2lmn(src):

	"""
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
		
	return l,m




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



if __name__ == '__main__':
    try:
        start_time = time.time()
        parser = optparse.OptionParser(formatter=optparse.TitledHelpFormatter(), usage=globals()['__doc__'], version='$Id$')
        parser.add_option ('-v', '--verbose', action='store_true', default=False, help='verbose output')
        (options, args) = parser.parse_args()
        #if len(args) < 1:
        #    parser.error ('missing argument')
        if options.verbose: print time.asctime()
        main()
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


