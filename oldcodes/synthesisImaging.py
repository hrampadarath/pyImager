import numpy as np
import matplotlib.pyplot as pl
from numpy.random import normal
from astropy.modeling import models, fitting
import warnings



##############
##GLOBALS
##############
wplanes = 1
wstep = 2000
over_sampling=4
ff_size=256
kernel_support=15

tinc = 1
hastart 	= 0 # HA start of observation in hrs
haend   	= 0.0167  # HA end of observation in hrs
hastep  	= 1 # scan length of observation in mins

T2 = 0.015  # half-width of FOV [radians]
L2 = 4000 # half-width of uv-plane [lambda]
N = int(T2*L2*4) # number of pixels 
#print "Making grids of side: ",N," pixels."
pix2rad = (0.015*2)/N

c0 = 3.0e8

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
    x=x[::-1,::-1] # reverse the kernel
    x*=1.0/x.sum() # normalise the kernel
    
    return exmid2(x,s)



def fraccoord(N, p, Qpx):
    """Compute whole and fractional parts of coordinates, rounded to Qpx-th fraction of pixel size
    :param N: Number of pixels in total 
    :param p: coordinates in range -1,1
    :param Qpx: Fractional values to round to
    """
    H=N/2
    x=(1+p)*H
    flx=np.floor(x + 0.5 /Qpx)
    fracx=np.around(((x-flx)*Qpx))    
    return (flx).astype(int), fracx.astype(int)



def convgridone(a, pi, fi, gcf, v):
    	"""Convolve and grid one visibility sample"""
    	sx, sy= gcf[0][0].shape[0]/2, gcf[0][0].shape[1]/2
    
	print gcf[0][0].shape
	print 'sx,sy:', sx, sy
	print pi[0]-sx, pi[0]+sx+1
	print pi[0]
	print pi[1]-sy, pi[1]+sy+1 
	print pi[1]
	print np.shape(a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ])
	
	print np.shape(gcf[fi[1],fi[0]])
	

    	# NB the order of fi below 
    	a[ pi[0]-sx: pi[0]+sx+1,  pi[1]-sy: pi[1]+sy+1 ] += gcf[fi[1],fi[0]]*v  
    	return a


def gridone(a,p,v):
    """grid one visibility without convolution"""
    
    a[p[0],p[1]] += v
    
    return a


def rows2cols(array):
	
	size = np.size(array)	
	invarray = np.flipud(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray

def row2colslr(array):
	
	size = np.size(array)	
	invarray = np.rot90(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray


def wKernel(w):

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
	return wg

def gridding(vis, uvw, HA):

	tinc = 1 # ha step in minutes
	
	#######GRIDDING
	grid_uv=np.zeros([N, N], dtype=complex)
	grid_wt=np.zeros([N, N], dtype=complex)
	
	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]
	vis = vis[zs]
	# divide the visibilities into wplanes/
	ii=np.linspace(0, len(vis), wplanes+1) # 
	ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(vis))]

	for j in range (len(ir)-1):

		ilow,ihigh=ir[j]
		
		#calculate the wkernel
		w=uvw[ilow:ihigh,2].mean()
		wg = wKernel(w)

		uvw_sub = uvw[ilow:ihigh]/L2 # extract subarray for w-slice
		(x, xf), (y, yf) = [fraccoord(grid_uv.shape[i], uvw_sub[:,i], over_sampling) for i in [0,1]]

		grid_uvi=np.zeros([N, N], dtype=complex)
		grid_wti=np.zeros([N, N], dtype=complex)
		vis_sub = vis[ilow:ihigh]
		for i in range(len(x)):
			#gridone(grid_uv, (x[i],y[i]), vis_sub[i])
			#gridone(grid_wt, (x[i],y[i]), 1.0+0j)	
			convgridone(grid_uvi,(x[i], y[i]), (xf[i], yf[i]), wg, vis_sub[i])
			convgridone(grid_wti,(x[i], y[i]), (xf[i], yf[i]), wg, 1.0+0j)
		grid_uv.real += grid_uvi.real
		grid_uv.imag += grid_uvi.imag
		
		grid_wt.real += grid_wti.real
		grid_wt.imag += grid_wti.imag	
		
		dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_uv)))
		psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_wt)))
	
	return dty_image, psf_image




def alt_grid(vis, uvw, time):



	#uvw,vis,time,freq = readvis(inuvwfile)
	
	
	print "Making grids of side: ",N," pixels."
	grid_uv=np.zeros([N, N], dtype=complex)
	grid_wt=np.zeros([N, N], dtype=complex)
	
	num_vis = len(vis)
	
	
	#Divide visibilities according to w-planes
	ii=range(0, num_vis, wstep) 
	ir=zip(ii[:-1], ii[1:]) + [(ii[-1], num_vis)]
	
	
	hastep = (haend-hastart)/(tinc/60.)
	#hastep = 1
	for j in range (len(ir)):
	#for j in range (0,1):
	
		ilow,ihigh=ir[j]
		#calculate the mean w-value 
		uvw_temp  = uvw[ilow:ihigh]
		wmean = uvw[ilow:ihigh,2].mean()
	
		vis_temp  = vis[ilow:ihigh]
		time_temp = time[ilow:ihigh]
		#freq_temp = freq[ilow:ihigh]
		#sort the w-plane by time
		ts=np.argsort(time_temp)
		uvws = uvw_temp[ts]
		viss = vis_temp[ts]
		times = time_temp[ts]
		#freqs = freq_temp[ts]
		#determine time-planes given the time increment
		for i in range (int(hastep)):
			
			#all visibilities within the same time-plane will require the same A-kernel
			trange = (hastart+(tinc/60.)*i, hastart+(tinc/60.)*(i+1))
			#print trange
			targs = np.where(np.logical_and(times>=trange[0], times<=trange[1]))
			if np.size(targs) == 0:
				continue
			hamean = times[targs].mean()
						
			#determine combined Gcf
			wg = wKernel(wmean)
				     
			print wg[0][0].shape[0]/2
			print wg[0][0].shape[1]/2
	
			#Grid
			grid_uvi=np.zeros([N, N], dtype=complex)
			grid_wti=np.zeros([N, N], dtype=complex)
			
			uvw_sub = uvws[targs]/L2
			print grid_uvi.shape[0]
			print grid_uvi.shape[1]
			print '----------'
			(x, xf), (y, yf) = [fraccoord(grid_uvi.shape[i], uvw_sub[:,i], over_sampling) for i in [0,1]]
			print x
			print y	
	
			vis_sub = viss[targs]
			print np.shape(vis_sub)
			#sys.exit()	
			for j in range(len(x)):
				print x[j],
				print y[j]	
				convgridone(grid_uvi,(x[j], y[j]), (xf[j], yf[j]), wg, vis_sub[j])
				convgridone(grid_wti,(x[j], y[j]), (xf[j], yf[j]), wg, 1.0+0j)
				#print 'no CF'
				#gridone(grid_uv, (x[j],y[j]),vis_sub[j])#no CF

			grid_uv.real += grid_uvi.real
			grid_uv.imag += grid_uvi.imag
		
			grid_wt.real += grid_wti.real
			grid_wt.imag += grid_wti.imag	
	
	
	dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_uv)))
	psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_wt)))

	return dty_image, psf_image





def convdegrid(a, p, gcf):
    """Convolutional-degridding

    :param a:   The uv plane to de-grid from
    :param p:   The coordinates to degrid at.
    :param gcf: List of convolution kernels

    :returns: Array of visibilities.
    """
    x, xf, y, yf = convcoords(a, p, len(gcf))
    #print 'a.shape = ', a.shape
    #print 'x:', x
    #print 'y:', y
    v = []
    sx, sy = gcf[0][0].shape[0] // 2, gcf[0][0].shape[1] // 2
    #print 'sx,sy:', sx, sy
    for i in range(len(x)):
        pi = (x[i], y[i])
	#print 'pi:', pi
        v.append((a[pi[0] - sx: pi[0] + sx + 1, pi[1] - sy: pi[1] + sy + 1] * gcf[xf[i]][yf[i]]).sum())
    return np.array(v)


def convcoords(a, p, Qpx):
    """Compute grid coordinates and fractional values for convolutional
    gridding

    The fractional values are rounded to nearest 1/Qpx pixel value at
    fractional values greater than (Qpx-0.5)/Qpx are roundeded to next
    integer index
    """
    (x, xf), (y, yf) = [fraccoord(a.shape[i], p[:, i], Qpx) for i in [0, 1]]
    return x, xf, y, yf


def fraccoord(N, p, Qpx):
    """Compute whole and fractional parts of coordinates, rounded to Qpx-th fraction of pixel size

    :param N: Number of pixels in total
    :param p: coordinates in range [-.5,.5]
    :param Qpx: Fractional values to round to
    """
    #print 'N = %s, p = %s, Qpx = %s' %(str(N),str(p),str(Qpx))	
    x = (1. + p) * N/2
    flx = np.floor(x + 0.5 / Qpx)
    fracx = np.around(((x - flx) * Qpx))
    return flx.astype(int), fracx.astype(int)

def deGridding(modvis,uvw):

	"""
	the forward step i.e. from visibilities on a regular grid 
	to visibilities on an irregular grid
	"""
	
	#######DEGRIDDING
	
	temp=uvw
	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]

	
	ii=range(0, len(uvw), wstep)	
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
	#wg = np.conj(wg)
	
	# degridding using the wkernel

	#vis_degrid = []
	vis_degrid = (convdegrid(modvis, uvw[ilow:ihigh] / L2, wg)) 
	#print uvw[ilow:ihigh]
	uvw[:,2] *= -1
	return vis_degrid,uvw







def overlapIndices(a1, a2,
                   shiftx, shifty):
    """ 
	Taken from B. Nikolic csclean.py from the SKA/crocodile github page
	Find the indices where two arrays overlapIndices

    :param a1: First array
    :param a2: Second array
    :param shiftx: Shift in x applied to a1
    :param shifty: Shift in y applied to a2
    """
    if shiftx >= 0:
        a1xbeg = shiftx
        a2xbeg = 0
        a1xend = a1.shape[0]
        a2xend = a1.shape[0] - shiftx
    else:
        a1xbeg = 0
        a2xbeg = -shiftx
        a1xend = a1.shape[0] + shiftx
        a2xend = a1.shape[0]

    if shifty >= 0:
        a1ybeg = shifty
        a2ybeg = 0
        a1yend = a1.shape[1]
        a2yend = a1.shape[1] - shifty
    else:
        a1ybeg = 0
        a2ybeg = -shifty
        a1yend = a1.shape[1] + shifty
        a2yend = a1.shape[1]

    return (a1xbeg, a1xend, a1ybeg, a1yend), (a2xbeg, a2xend, a2ybeg, a2yend)


def row2cols(array):
	
	size = np.size(array)	
	invarray = np.flipud(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))

	return invarray


def udrot90(array):
	
	size = np.size(array)	
	#invarray = np.rot90(np.reshape(array,size,order='F').reshape(np.shape(array)[0],np.shape(array)[1]))
	invarray = np.flipud(np.rot90(array))
	return invarray

def cleanBeam(dirtypsf):

    """
    To simulate a clean psf (or clean beam)
    by fitting a 2D Gaussian to the main lobe of the dirty beam.
    """

    PSF = dirtypsf
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
    pS = 30 # radius in pixel of clean beam

    sub_psf = cleanpsf[cent-pS:cent+pS,cent-pS:cent+pS]
    pad_width = np.shape(PSF)[0]/2 - pS
    sub_psf = np.pad(sub_psf,(pad_width,pad_width),mode='constant')

    return sub_psf



def argmax(a):
    """ Return unravelled index of the maximum

    param: a: array to be searched
    """
    return np.unravel_index(a.argmax(), a.shape)



def basel_rot(ants_xyz,start_freq,ha,dec,lat):
    
    	"""
    	Determine the uvw and observing frequency per HA scan
    	"""
	x,y,z=np.hsplit(ants_xyz,3)
	freq = start_freq 
	freq_scale = freq / c0
	
	t=x*np.cos(ha) - y*np.sin(ha)
	u= (x*np.sin(ha) + y*np.cos(ha)) * freq_scale
	v= (-1.*t*np.sin(dec)+ z*np.cos(dec)) * freq_scale
	w= (t*np.cos(dec)+ z*np.sin(dec)) * freq_scale
	ants_uvw = np.hstack([u,v,w])
		
	res=[]
	for i in range(ants_uvw.shape[0]):
		for j in range(i+1, ants_uvw.shape[0]):
			res.append(ants_uvw[j]-ants_uvw[i])
	#Now obtain the response of the interferometer.
	basel_uvw_scan = np.array(res)
	
	
	return basel_uvw_scan



def conjugateSymmetry(basel_uvw,vis,HA):
	"""
	Apply the conjugate symmetry to the visibility data
	"""
	uu = basel_uvw[:,0]
	vv = basel_uvw[:,1]
	ww = basel_uvw[:,2]
	
	uvw = np.column_stack((uu,vv,ww))#column_stack - Stack 1-D arrays as columns into a 2-D array.
	#get conjugate symmetry
	tmp_uvw = uvw*np.array([-1.,-1.,1.])
	tmp_HA = HA
	HA = np.hstack((HA,tmp_HA))
	
	#sort the UVW data 
	uvw = np.vstack((uvw,tmp_uvw))#vstack - Stack arrays in sequence vertically (row wise).
		
	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]
	HA = HA[zs]

	tmp_vis = np.conj(vis)
	vis = np.hstack((vis,tmp_vis))
	vis = vis[zs]
	
	return uvw, vis, HA


def add_vis_noise(vis,hanum):
	#add noise
	mu = 0.
	points = len(vis)
	sigma = 1.e-3*np.sqrt(points)*np.sqrt(hanum) # 1 muJy
	#print 'Injected noise = ', sigma
	#sigma = 0
	noise = np.zeros((points), dtype=np.complex64)
	noise.real = normal(mu, sigma, points)
	noise.imag = normal(mu, sigma, points)
	
	vis_re = vis.real + noise.real
	vis_im = vis.imag + noise.imag
	vis = vis_re + 1j*vis_im

	return vis

def make_vis(basel_uvw_scan):
	
	"""
	Simulate the visibilities for a single scan (defined by the HA)
	"""
    
    	src= {'1':{'l':-0.0015,'m':-0.0058,'src_amp':200}}
	        #'2':{'l':-0.0044,'m':0.003,'src_amp':220},
	        #'3':{'l':-0.001,'m':0.003,'src_amp':230},
	        #'4':{'l':0.005,'m':-0.0065,'src_amp':190}}
	        #'5':{'l':0.003,'m':0.0065,'src_amp':110}}   # Units are Radians	
	
	#simulate the visibilities
	vis = 0 
	for s in src:
	    l = src[s]['l']
	    m = src[s]['m']
	    src_pos=np.array([l, m , np.sqrt(1 - l**2 - m**2)])
	    src_amp = src[s]['src_amp']
	    vis += src_amp*np.exp(-2j*np.pi* np.dot(basel_uvw_scan[:,0:2], src_pos[0:2]))

	return vis
