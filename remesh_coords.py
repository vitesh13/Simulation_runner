#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
import argparse
import os
from scipy.interpolate import griddata
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as PyPlot
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties

parser = argparse.ArgumentParser('Create equispaced coordinates for the CA code')

parser.add_argument('filenames', nargs='+',
                    help='DADF5 files')
parser.add_argument('-s','--spacing',dest = 'spacing', metavar = 'float',type = float)

parser.add_argument('-u','--unitscale',dest = 'unit', metavar = 'float', type = float)


options = parser.parse_args()

dx = options.spacing
unit_scale = options.unit

is2d = 0 # 1 for 2D data

data = np.loadtxt(options.filenames[0],skiprows=1)
min_x = unit_scale*np.min(data[:,0])
min_y = unit_scale*np.min(data[:,1])
min_z = unit_scale*np.min(data[:,2])


#shift coords to start from 0
data[:,0] = unit_scale*data[:,0] - min_x
data[:,1] = unit_scale*data[:,1] - min_y
data[:,2] = unit_scale*data[:,2] - min_z

nx = int(np.max(data[:,0])/dx)
ny = int(np.max(data[:,1])/dx)
nz = int(np.max(data[:,2])/dx)
print('Nx:',nx)
print('Ny:',ny)
print('Nz:',nz)

x_new = np.mgrid[0:nx+1]*dx 
y_new = np.mgrid[0:ny+1]*dx 
z_new = np.mgrid[0:nz+1]*dx 

#new_coords = np.vstack(np.meshgrid(x_new,y_new,z_new)).reshape(3,-1).T
#new_coords = np.vstack(np.meshgrid(y_new,z_new,x_new)).reshape(3,-1).T
#new_coords = np.vstack(np.meshgrid(z_new,y_new,x_new)).reshape(3,-1).T
new_coords = np.stack(np.meshgrid(x_new,y_new,z_new,indexing='ij'),axis=-1).reshape(((nx+1)*(ny+1)*(nz+1),3),order='F')

new_data = np.zeros((len(new_coords),10))
new_data[:,0:3] = new_coords

new_data[:,3:10] = griddata(data[:,0:3],data[:,3:10],new_data[:,0:3],method='nearest')

new_filename = 'remesh_' + os.path.basename(options.filenames[0])
dir_file     = os.path.dirname(os.path.abspath(options.filenames[0]))
new_path     = os.path.join(dir_file,new_filename)
np.savetxt(new_path,new_data,fmt = ' '.join(['%.10e']*3 + ['%i'] + ['%.10e']*6), \
           header='{} {}'.format(str(len(new_data)),str(int(np.max(data[:,3])))),comments='')
#plotting to check
#fig = PyPlot.figure()

# write out output file


