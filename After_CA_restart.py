#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
import os
from scipy import spatial
import h5py


class CASIPT_postprocessing():
  """
  Class combining different scripts for CASIPT post-processing.
  
  """
  def __init__(self,CA_folder):
    """
    Sets the files names and folder names.

    Parameters:
    -----------
    CA_folder : str
      Path to the folder where CA results are stored.
   
    """
    
    #self.simulation_folder = 'DRX_sample_simulations'
    self.CA_folder         = CA_folder 
    self.CA_geom           = 'resMDRX.3D.geom'

  
  def geomFromCA(self,dx):
    """
    Creates geometry out of CA output.

    Parameters:
    -----------
    dx : float
      The grid spacing.
   
    """
    os.chdir('/nethome/v.shah/{}'.format(self.CA_folder))
    with open(self.CA_geom,'r') as f:
      grid_size = f.readline().split()[2:7:2] #read only first line

    grid_size = np.array([int(i) for i in grid_size])
 
    n_elem = np.prod(grid_size)   #need to check if it is correct for 2D geometry

    data_to_geom = []
    data_to_geom.append('{} header'.format(1))  #so that we can modify this later
    data_to_geom.append('grid a {} b {} c {}'.format(*grid_size))
    data_to_geom.append('size x {} y {} z {}'.format(*(grid_size*dx))) #give input that will kind of decide the spacing (like in DREAM3D)
    data_to_geom.append('origin x 0.0 y 0.0 z 0.0')
    data_to_geom.append('homogenization 1')
    
    geom_data_1 = np.loadtxt('resMDRX.texture_MDRX.txt',skiprows=0,usecols=(2))
    print(np.shape(geom_data_1),geom_data_1)
    geom_data_1 = geom_data_1.tolist()
    data_to_geom.append('microstructures {}'.format(int(np.amax(geom_data_1))+1))

    #write microstructure part
    data_to_geom.append('<microstructure>')
    
    for count,grain in enumerate(geom_data_1):
      data_to_geom.append('[Grain{:02d}]'.format(int(grain)+1))
      data_to_geom.append('crystallite 1')
      data_to_geom.append('(constituent) phase 1 texture {} fraction 1.0'.format(int(grain)+1))
    
    #write texture part
    data_to_geom.append('<texture>')
    texture_data = np.loadtxt('resMDRX.texture_MDRX.txt',skiprows=0,usecols=((4,6,8)))
    
    for i,texture in enumerate(texture_data):
      data_to_geom.append('[Grain{:02d}]'.format(i+1))
      data_to_geom.append('(gauss) phi1 {} Phi {} phi2 {} scatter 0.0 fraction 1.0'.format(*texture))

    # calculating header length
    header_value = len(data_to_geom) - 1
    data_to_geom[0] = '{} header'.format(header_value)
    geom_data = np.loadtxt('resMDRX.3D.geom',skiprows=1,usecols=(1))
    numbers = geom_data + 1
    #write numbers in geom file 
    for i in numbers:
      data_to_geom.append(int(i))
    
    for line in data_to_geom:
      print(line)
    
    array = np.array(data_to_geom)
    np.savetxt('test.geom',array,fmt='%s',newline='\n') 

  def findNeighbours(self,regrid,remesh,hdf):
    """
    Finds neighbours by comparing the original damask datapoint and remeshed datapoints.
    Based on the neighbourhood search, creates a new restart hdf5 file.

    Parameters:
    -----------
    regrid : str
      Path of the file having regridded coords from Karo.
    remesh : str
      Path of the file having remeshed coords.
    hdf    : str
      Path of the regridded hdf5 from Karos code. 

    """
    regridding_coords = np.loadtxt(regrid,skiprows=1,usecols=(0,1,2))
    remeshed_coords   = np.loadtxt(remesh,skiprows=1,usecols=(0,1,2))
    print('length',len(remeshed_coords))
    remeshed_coords_1      = remeshed_coords 
    remeshed_coords_1[:,0] = remeshed_coords[:,0] + remeshed_coords[1,0] - remeshed_coords[0,0]
    remeshed_coords_1[:,1] = remeshed_coords[:,1] + remeshed_coords[1,0] - remeshed_coords[0,0]
    remeshed_coords_1[:,2] = remeshed_coords[:,2] + remeshed_coords[1,0] - remeshed_coords[0,0] #this works only if it is equidistant in all directions
    tree = spatial.cKDTree(regridding_coords)
    
    nbr_array = tree.query(remeshed_coords_1,1)[1] #finding the indices of the nearest neighbour
    file_name = os.path.splitext(hdf)[0] + '_CA.hdf5'

    f = h5py.File(file_name)
    
    hdf = h5py.File(options.hdf[0])

    const_values = ['C_minMaxAvg','C_volAvg','C_volAvgLastInc','F_aim','F_aimDot','F_aim_lastInc']
    
    const_group = ['HomogState']
    
    diff_values = ['F','Fi','Fp','Li','Lp','S','1_omega_plastic']
    
    diff_values_1 = ['F_lastInc','F']
    
    diff_groups = ['constituent']
    
    for i in const_values:
      f.create_dataset('/solver/' + i,data=np.array(hdf['/solver/' + i]))
    
    f.create_group('constituent')
    f.create_group('materialpoint')

    for i in diff_values:
      if i != '1_omega_plastic':
        data_array = np.zeros((len(remeshed_coords),) + np.shape(hdf[i])[1:])
        for count,point in enumerate(nbr_array):
          data_array[count] = np.array(hdf[i][point])
        f[i] = data_array  
      else:
        data_array = np.zeros((len(remeshed_coords),) + np.shape(hdf['/constituent/' + i])[1:])
        for count,point in enumerate(nbr_array):
          data_array[count] = np.array(hdf['/constituent/' + i][point])
        f['/constituent/' + i] = data_array  

    for i in diff_values_1:
        xsize      = int(round(np.max(remeshed_coords[:,0])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
        ysize      = int(round(np.max(remeshed_coords[:,1])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
        zsize      = int(round(np.max(remeshed_coords[:,2])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
        totalsize  = int(xsize*ysize*zsize)
        print(totalsize)
        data_array = np.zeros((totalsize,) + np.shape(hdf['/solver/' + i])[3:])
        input_data = np.array(hdf['/solver/' + i]).reshape(((np.prod(np.shape(np.array(hdf['/solver/' + i]))[0:3]),)+np.shape(np.array(hdf['/solver/' + i]))[3:]))
        print(np.shape(data_array),np.shape(input_data))
        for count,point in enumerate(nbr_array):
          data_array[count] = input_data[point]
        data_array = data_array.reshape((zsize,ysize,xsize,) + np.shape(hdf['/solver/' + i])[3:])
        f['/solver/' + i] = data_array

