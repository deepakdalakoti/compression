# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:03:01 2019
Script to read S3D savefile into Paraview
@author: Harshad Ranadive
"""

from scipy.io import FortranFile
import numpy as np

####################### User Inputs ######################
dirname='../share/coarseDNS_JICF/s-3.6914E-03'

nspec=29
ndim=3

# s3d.in
nx_g=1280
ny_g=896
nz_g=576
npx=32
npy=16
npz=12
xmin=-2e-2
xmax=12e-2
ymin=0.0
ymax=9.0e-2
zmin=-3.8e-2
zmax=3.8e-2
unif_grid_x=0
unif_grid_y=0
unif_grid_z=0

# grid.in
imap_x=-1
imap_y=0
imap_z=0
lfine_x=0.01
lfine_y=0.001
lfine_z=0.001
mingridx=125e-6
mingridy=125e-6
mingridz=125e-6

# s3d stores in non-dimensional form - so provide ref values from s3d output file here
a_ref=3.47200E+02     # m/s
t_ref=1.20000E+02     # K
p_ref=1.41837E+05     # N/m2
time_ref=2.02627E-06  # s

# species names
species_names = ['NC12H26', 'H', 'O', 'OH', 'HO2', 'H2', 'H2O', 'H2O2', 'O2',
                 'CH3', 'CH4', 'CH2O', 'CO', 'CO2', 'C2H2', 'C2H4', 'C2H6',
                 'CH2CHO', 'aC3H5', 'C3H6', 'C2H3CHO', 'C4H7', 'C4H81', 'C5H9',
                 'C5H10', 'C6H12', 'C7H14', 'C8H16', 'C9H18', 'PXC9H19',
                 'C10H20', 'C12H24', 'C12H25O2', 'OC12H23OOH', 'N2']
#########################################################
## Functions for generating stretched grid
def gridbnew(ng,pos_min,pos_max,del_min,lfine,imap):
    gmax = pos_max
    gmin = pos_min

    b_stretch = del_min/(gmax-gmin)*(ng-1)
    if (imap==0):
        smin = -1
        smax =  1
    elif (imap==-1):
        smin = 0
        smax =  1
    elif (imap==1):
        smin = -1
        smax = 0
    g_range = gmax - gmin
    s_range = smax - smin
    ds = s_range/(ng-1)
    s = smin + ds*np.arange(ng)
    trs = lfine/g_range/b_stretch
    ptmp = fmapnew (s, b_stretch,trs)
    pos = (g_range/s_range)*(ptmp - smin) +  gmin
    return pos

def fmapnew(s,beta,trs):
    q = np.arctanh(0.995)
    f = (beta*s + \
         (1-beta)*(s-(1-trs)/q*np.log(np.cosh(-q*(s+1)/(1-trs))/np.cosh(-q/(1-trs))))\
         /(1+(1-trs)/q*np.log(1/np.cosh(-q/(1-trs)))) +\
         (1-beta)*(s+(1-trs)/q*np.log(np.cosh(q*(s-1)/(1-trs))/np.cosh(-q/(1-trs))))\
         /(1+(1-trs)/q*np.log(1/np.cosh(-q/(1-trs)))) )
    return f

##########################################################
# main script begins here
nx=nx_g//npx
ny=ny_g//npy
nz=nz_g//npz

# generate grid
x=gridbnew(nx_g,xmin,xmax,mingridx,lfine_x,imap_x)
y=gridbnew(ny_g,ymin,ymax,mingridy,lfine_y,imap_y)
z=gridbnew(nz_g,zmin,zmax,mingridz,lfine_z,imap_z)

yspecies_read    = np.empty([nx,ny,nz,nspec])
temperature_read = np.empty([nx,ny,nz])
pressure_read    = np.empty([nx,ny,nz])
velocity_read    = np.empty([nx,ny,nz,ndim])

yspecies         = np.zeros([nx_g,ny_g,nz_g,nspec], dtype='single')
#temperature      = np.zeros([nx_g,ny_g,nz_g]).astype('single')
#pressure         = np.zeros([nx_g,ny_g,nz_g], dtype='single')
#velocity         = np.zeros([nx_g,ny_g,nz_g,ndim], dtype='single')

# loop over savefiles
for zid in range(npz):
    for yid in range(npy):
        for xid in range(npx):

            myid = zid*npx*npy + yid*npx + xid
            filename = dirname + '/field.' + "{0:0=6d}".format(myid)
            f = FortranFile(filename, 'r')

            time=f.read_reals(np.double)*time_ref
            time_step=f.read_reals(np.double)*time_ref
            time_save=f.read_reals(np.double)*time_ref

            for L in range(nspec):
                yspecies_read[:,:,:,L]=f.read_reals(np.single).reshape((nx,ny,nz),order='F')

            temperature_read[:,:,:]=f.read_reals(np.single).reshape((nx,ny,nz),order='F')*t_ref
            pressure_read[:,:,:]=f.read_reals(np.single).reshape((nx,ny,nz),order='F')*p_ref

            for i in range(ndim):
                velocity_read[:,:,:,i]=f.read_reals(np.single).reshape((nx,ny,nz),order='F')*a_ref

            pout=f.read_reals(np.double)*p_ref
            f.close()

            xsrt=xid*nx
            xend=(xid+1)*nx
            ysrt=yid*ny
            yend=(yid+1)*ny
            zsrt=zid*nz
            zend=(zid+1)*nz

            yspecies    [xsrt:xend,ysrt:yend,zsrt:zend,:] = yspecies_read    [:,:,:,:]
            #temperature [xsrt:xend,ysrt:yend,zsrt:zend]   = temperature_read [:,:,:]
            #pressure    [xsrt:xend,ysrt:yend,zsrt:zend]   = pressure_read    [:,:,:]
            #velocity    [xsrt:xend,ysrt:yend,zsrt:zend,:] = velocity_read    [:,:,:,:]
print(yspecies.dtype)
fid = open('yspecies_JICF.dat','w')
yspecies.tofile(fid)
fid.close()            
#fid = open('temp_slice.dat','w')
#temperature[:,:,288].tofile(fid)
#fid.close()
'''

# Put data into a "ParaView Programmable Source".
# The output dataset type needs to be set to "vtkRectilinearGrid".
from vtk.numpy_interface import dataset_adapter as dsa
out = self.GetRectilinearGridOutput()
out.SetDimensions(nx_g, ny_g, nz_g)
out.SetXCoordinates(dsa.numpyTovtkDataArray(x, 'X'))
out.SetYCoordinates(dsa.numpyTovtkDataArray(y, 'Y'))
out.SetZCoordinates(dsa.numpyTovtkDataArray(z, 'Z'))
pd = out.GetPointData()
for L, name in enumerate(species_names):
    pd.AddArray(dsa.numpyTovtkDataArray(yspecies[:,:,:,L].reshape(-1, order='F'), 'Y_' + name))
pd.AddArray(dsa.numpyTovtkDataArray(temperature.reshape(-1, order='F'), 'T'))
pd.AddArray(dsa.numpyTovtkDataArray(pressure.reshape(-1, order='F'), 'p'))
pd.AddArray(dsa.numpyTovtkDataArray(velocity.reshape(-1, 3, order='F'), 'u'))
'''
