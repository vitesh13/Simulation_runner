#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import os,sys,math,re,time,struct
import h5py
import re
import numpy as np
import os, sys, shutil
import scipy
from scipy import spatial
from scipy.interpolate import griddata
import argparse
import subprocess
import damask
import pandas as pd
import itertools as it

from scipy.linalg import polar
from numpy.linalg import inv
import shutil
from imp import reload

from damask import regriding as rgg
from damask import Rotation
from output_reader import output_reader

# combining the codes from 0_main_all and remesh_coords

class Remesh_for_CA():
  """
  This class regrids and remeshes the DAMASK output to be used in CASIPT.
  """
  
  def __init__(self,geom,load,inc,folder):
    """
    Initializes the arguments.

    Parameter
    ---------
    geom : str
      Name of the geom file
    load : str
      Name of the load file
    inc : int
      Increment for which regridding is being done
    folder : str
      Path to the folder
    """
    self.geom = geom
    self.load = load
    self.inc  = inc
    self.folder = folder

  def main_all(self,geom,load,inc,folder):
    """
    Regrids the data and generates a HDF5 and text file for CA.
    
    Parameters
    ----------
    geom : str
      Name of the geom file
    load : str
      Name of the load file
    inc : int
      Increment for which regridding is being done
    folder : str
      Path to the folder
    """ 

    isElastic = False
    scale = 1.0
    grid = False 
    os.chdir('/nethome/v.shah/{}/'.format(folder))
    geom_name = rgg.setting_util.remove_fileFormat(geom)
    load_name = rgg.setting_util.remove_fileFormat(load)
    isElastic = rgg.setting_util.set_elasticDeformation(isElastic)
    seedScale = rgg.setting_util.set_scale4gridSeedsRegridding(scale, grid)

    hdf5_name = '%s_%s.hdf5'%(geom_name,load_name)
    history_name = '%s_%s_regriddingHistory.hdf5'%(geom_name,load_name)
    
    rg = rgg.geom_regridder(geom_name,load_name,increment=inc)
    rg.read_geom()
    print(rg.microstructureFlatten_0.shape)
    rg.read_h5Output(inc)   #this function also gets the displacements of nodes and cells apparently
    rg.regrid_geom()
    self.new_size = rg.sizeRVE_regrid
    self.new_grid = rg.gridSeeds_regrid
    print('new size type',type(self.new_size))
   
    # Building the new coordinates
    elem0 = int(rg.gridSeeds_0.prod())
    elem_rg = int(rg.gridSeeds_regrid.prod())
    
    New_RVE_size = rg.sizeRVE_regrid
    new_grid_cell = rg.gridSeeds_regrid
    origin0 = rg.originRVE_0
    Cell_coords = rg.gridCoords_cell_regrid   #or it should be after the periodic shift??
    print(Cell_coords.shape)

    #------------------------------------------
    # reading main inputs for processing
    out5 = output_reader(hdf5_name)
    inc, inc_key = out5.make_incerement(inc)
    phase_name = out5.constituents[0]
    # reading main inputs
    orientations = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['orientation'])
    orientations = np.array(orientations.tolist())
    ## grain rotation
    grain_rotation = np.array(out5.data[inc_key]['constituent'][phase_name]['generic']['grain_rotation'])  #need to add [:,3] is using damask generated output
    grain_rotation_rg = grain_rotation[rg.get_nearestNeighbors()]
    grain_rotation_rg_scalar = grain_rotation_rg
    mylist = []
    for count,ori in enumerate(orientations):
      o = Rotation.fromQuaternion(list(ori))
      mylist.append(o.asEulers(degrees=None).tolist())
    
    eulers = np.array(mylist)
    eulers_rg = eulers[rg.get_nearestNeighbors()]
    ## dislocation density
    rho_m = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['rho_mob'])
    rho_d = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['rho_dip'])
    rho = rho_m + rho_d
    ## reshaped
    rho_m_rg = rho_m[rg.get_nearestNeighbors()]
    rho_d_rg = rho_d[rg.get_nearestNeighbors()]
    rho_rg = rho[rg.get_nearestNeighbors()]
    rho= np.sum(rho_rg,axis=1)
    ## subgrain sizes
    r_s   = np.array(out5.data[inc_key]['constituent'][phase_name]['plastic']['r_s']) 
    r_s_rg   = r_s[rg.get_nearestNeighbors()]
    #
    print('orientation shape before: ',np.shape(orientations))
    orientations_rg = orientations[rg.get_nearestNeighbors()] #hopefully it works
    print('orientation shape after: ',np.shape(orientations_rg))


    #--------------
    # make df
    #---------------
    
    df = pd.DataFrame() 
    # coords for new grids
    Cell_coords = Cell_coords*1E-06
    df['x'] = Cell_coords[:,0]
    df['y'] = Cell_coords[:,1]
    df['z'] = Cell_coords[:,2]
    # initial grain
    #df['grain'] = np.array(df_cell['grain'])[rebacked_id]
    df['grain'] = rg.microstructureFlatten_0[rg.nearestNeighbors_locs] 
    ## Rotation
    df['Rotation'] = grain_rotation_rg_scalar
    # # total dislo
    df['rho'] = rho
    # # subgrain sizes 
    df['r_s'] = r_s_rg
    # euler angles
    df['phi1'] = eulers_rg[:,0]
    df['PHI'] = eulers_rg[:,1]
    df['phi2'] = eulers_rg[:,2]

    header_f = '%s %s\n'%(str(int(np.prod(rg.gridSeeds_regrid))),str(max(df['grain']))) #,str(max(df['grain']))
    
    output = '%s_%s'%(geom_name,load_name)
    file_rg = '%s_%s.txt'%(output,inc)
    if not os.path.exists(os.path.join('/nethome/v.shah',folder,'postProc')):
        os.makedirs(os.path.join('/nethome/v.shah',folder,'postProc'))
    
    with open('postProc/'+file_rg,'w') as f:
        f.write(header_f)
        df.to_string(f,header=False,formatters=["{:.8f}".format,"{:.8f}".format,"{:.8f}".format, \
                                                "{:.8f}".format,"{:.8f}".format,"{:.6E}".format, \
                                                "{:.12f}".format,"{:.8f}".format,"{:.8f}".format, \
                                                "{:.8f}".format],index=False)
    
    #create a hdf5 file
    new_hdf_name = 'new_' + geom_name + '_' + load_name + '_' + str(inc) + '.hdf5'
    hdf = h5py.File(new_hdf_name,'w')
    hdf.attrs['DADF5_version_major'] = 0
    hdf.attrs['DADF5_version_minor'] = 6
    hdf.attrs['DADF5-version'] = 0.2
    hdf.create_group('geometry')
    hdf['geometry'].attrs['grid'] = np.array(rg.gridSeeds_regrid, np.int32)
    hdf['geometry'].attrs['size'] = np.array(rg.sizeRVE_regrid, np.float64)
    hdf['geometry'].attrs['origin'] = np.array(rg.originRVE_0,np.float64)
    #
    ##mapping data
    comp_dtype = np.dtype([('Name',np.string_,64),('Position',np.int32)])
    #new_len    = np.prod(np.int32(rg.new_grid))
    new_len    = np.prod(np.int32(rg.gridSeeds_regrid))
    data_name  = [phase_name]*int(new_len)
    data_value = [i for i in range(new_len)]
    new_data   = list(zip(data_name,data_value))
    new_data   = np.array(new_data,dtype=comp_dtype)
    new_data   = new_data.reshape(new_len,1)
    dataset    = hdf.create_dataset("/mapping/cellResults/constituent",(new_len,1),comp_dtype)
    dataset[...] = new_data
    
    data_name  = ['1_SX']*int(new_len)
    new_data   = list(zip(data_name,data_value))
    new_data   = np.array(new_data,dtype=comp_dtype)
    new_data   = new_data.reshape(new_len,1)
    dataset    = hdf.create_dataset("/mapping/cellResults/materialpoint",(new_len,1),comp_dtype)
    dataset[...] = new_data
    
    #orientation_rg
    comp_dtype  = np.dtype([('w',np.float64),('x',np.float64),('y',np.float64),('z',np.float64)])
    #new_len     = np.prod(np.int32(rg.new_grid))
    new_len     = np.prod(np.int32(rg.gridSeeds_regrid))
    dataset_ori = hdf.create_dataset("/{}/constituent/{}/generic/orientation".format(inc_key,phase_name),(new_len,),comp_dtype)
    orientations_rg = np.array([tuple(i) for i in orientations_rg[:]],dtype=comp_dtype)
    dataset_ori[...] = orientations_rg
    hdf["/{}/constituent/{}/generic/orientation".format(inc_key,phase_name)].attrs['Lattice'] = 'bcc'
    
    #rho_rg and grain_rotation_rg_scalar
    dataset_rho = hdf.create_dataset("/{}/constituent/{}/plastic/tot_density".format(inc_key,phase_name),(new_len,))
    dataset_rho[...] = rho
    
    dataset_rot = hdf.create_dataset("/{}/constituent/{}/generic/grain_rotation".format(inc_key,phase_name),(new_len,))
    dataset_rot[...] = grain_rotation_rg_scalar
    
    hdf.create_group('/{}/materialpoint/1_SX/generic'.format(inc_key))
    hdf.create_group('/{}/materialpoint/1_SX/plastic'.format(inc_key))
    
    
    #hdf.create_group('inc{}'.format(inc))
    print(inc_key)
    return self.new_size,self.new_grid

  def remesh_coords(self,filename,unit,folder):
    """
    Remeshes the data to equidistant grid.
    
    Parameters
    ----------
    filename : str 
      file path
    unit : float
      Our units in comparison to DAMASK
    folder : str
      simulation folder
    """ 
    os.chdir('/nethome/v.shah/{}/postProc'.format(folder))
    dx = np.min(self.new_size/self.new_grid)*1E-06
    print(dx)
    unit_scale = unit
    
    is2d = 0 # 1 for 2D data
    
    data = np.loadtxt(filename,skiprows=1)
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
    
    new_filename = 'remesh_' + os.path.basename(filename)
    dir_file     = os.path.dirname(os.path.abspath(filename))
    new_path     = os.path.join(dir_file,new_filename)
    np.savetxt(new_path,new_data,fmt = ' '.join(['%.10e']*3 + ['%i'] + ['%.10e']*6), \
               header='{} {}'.format(str(len(new_data)),str(int(np.max(data[:,3])))),comments='')
    return nx,ny,nz
      
