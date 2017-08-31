import numpy as np
import matplotlib.pyplot as pl
from numpy.random import normal
from numpy.linalg import inv
from skimage import color, data, restoration


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

def makedirtyImage(vis,uvw):

	wstep=2000
	over_sampling=4
	ff_size=256
	kernel_support=15
	
	T2 = 0.015  # half-width of FOV [radians]
	L2 = 4000 # half-width of uv-plane [lambda]
	N = int(T2*L2*4) # number of pixels 
	print "Making grids of side: ",N," pixels."
	pix2rad = (0.015*2)/N
	
	#######GRIDDING
	grid_uv=np.zeros([N, N], dtype=complex)
	grid_wt=np.zeros([N, N], dtype=complex)
	
	temp=uvw
	zs=np.argsort(uvw[:,2])
	uvw = uvw[zs]
	vis = vis[zs]
	
	ii=range(0, len(vis), wstep) # Bojan, is this what you mean here? Every 2000 entries or every 2000 lambda..?
	
	ir=zip(ii[:-1], ii[1:]) + [(ii[-1], len(vis))]
	ilow,ihigh=ir[0]

	uvw_sub = uvw[ilow:ihigh,:]/L2 # extract subarray for w-slice
	(x, xf), (y, yf) = [fraccoord(grid_uv.shape[i], uvw_sub[:,i], over_sampling) for i in [0,1]]

	vis_sub = vis[ilow:ihigh]
	for i in range(len(x)):
	    gridone(grid_uv, (x[i],y[i]), vis_sub[i])
	    gridone(grid_wt, (x[i],y[i]), 1.0+0j)	
	    #convgridone(grid_uv,(x[i], y[i]), (xf[i], yf[i]), wg, vis_sub[i])
	    #convgridone(grid_wt,(x[i], y[i]), (xf[i], yf[i]), wg, 1.0+0j)
	
	
	dty_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_uv)))
	psf_image=np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(grid_wt)))
	
	return dty_image, psf_image, pix2rad



ants_xyz=np.genfromtxt("./VLA_C_hor_xyz_v2.txt")
print "Number of antennas in array:",ants_xyz.shape[0]


ha = 0.0; dec = np.pi/4.  # Units are Radians pointing centre

#UVW coordinates
x,y,z=np.hsplit(ants_xyz,3)

t=x*np.cos(ha) - y*np.sin(ha)
u=x*np.sin(ha) + y*np.cos(ha)
v=-1.*t*np.sin(dec)+ z*np.cos(dec)
w=t*np.cos(dec)+ z*np.sin(dec)
ants_uvw = np.hstack([u,v,w])
#baseline distribution

res=[]
for i in range(ants_uvw.shape[0]):
    for j in range(i+1, ants_uvw.shape[0]):
        res.append(ants_uvw[j]-ants_uvw[i])

basel_uvw = np.array(res)
print "Number of baselines in array:", basel_uvw.shape[0]



#List of source positions and amplitude

#src= {'1':{'l':-0.0015,'m':-0.0058,'src_amp':100},
#        '2':{'l':-0.0044,'m':0.003,'src_amp':100},
#        '3':{'l':-0.001,'m':0.003,'src_amp':100},
#        '4':{'l':0.005,'m':-0.0065,'src_amp':100},
#        '5':{'l':0.003,'m':0.0065,'src_amp':100}}   # Units are Radians

src= {'1':{'l':-0.0015,'m':-0.00575,'src_amp':100}}

#simulate the visibilities
vis = []  
for s in src:
    l = src[s]['l']
    m = src[s]['m']
    src_pos=np.array([l, m , np.sqrt(1 - l**2 - m**2)])
    src_amp = src[s]['src_amp']
    vis.append(src_amp*np.exp(-2j*np.pi* np.dot(basel_uvw[:,0:2], src_pos[0:2])))
    
vis=np.sum(vis,axis=0)      
u = basel_uvw[:,0]
v = basel_uvw[:,1]
w = basel_uvw[:,2]


#add noise
mu = 0.
sigma = 0.0000010*np.sqrt(len(u))*np.sqrt(1) # 1 muJy
print sigma
#sigma = 0
points = len(vis)
noise = np.zeros((points), dtype=np.complex64)
noise.real = normal(mu, sigma, points)
noise.imag = normal(mu, sigma, points)

vis_re = vis.real + noise.real
vis_im = vis.imag + noise.imag
vis = vis_re + 1j*vis_im

uvw = np.column_stack((u,v,w))
# conjugate symmetry
tmp_uvw = uvw*np.array([-1.,-1.,1.])
tmp_vis = vis_re - 1j*vis_im

vis = np.hstack((vis,tmp_vis))
uvw = np.vstack((uvw,tmp_uvw))





############################IMAGING: with sources
dty_image,psf,pix2rad = makedirtyImage(vis,uvw)


deconv,_ = restoration.unsupervised_wiener(dty_image.real,psf.real)


pl.subplot(121)
pl.imshow(dty_image.real)
pl.title("Dirty Image")
pl.subplot(122)
pl.imshow(deconv)
pl.title("PSF")
pl.show()


