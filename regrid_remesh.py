#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import os,math,re,struct
import h5py
import re
import numpy as np
import os, shutil
import scipy
from scipy import spatial
from scipy.interpolate import griddata
import damask
import pandas as pd
import itertools as it

from scipy.linalg import polar
from numpy.linalg import inv
import shutil
from imp import reload

#from damask import regriding as rgg
from damask import geom_regridder
from damask import restart_regridder
from damask import Rotation
from damask import Orientation
#from output_reader import output_reader

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
    Generates a new restart file using regridding framework.
    Also create a new hdf5 results file for deformed microstructure. 
    
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
    os.chdir(folder)

    rg = geom_regridder(geom,load,increment=inc)
    rg.regrid_geom()
    rg.write_geomRegridded()
  
    rgr = restart_regridder(geom,load,increment_title='inc{}'.format(inc))
    rgr.write_h5OutRestart(rg.nearestNeighbors_locs,rg.gridSeeds_regrid,isElastic)
    
    self.new_size = rg.sizeRVE_regrid
    self.new_grid = rg.gridSeeds_regrid
   
    # Building the new coordinates
    elem0 = int(rg.gridSeeds_0.prod())
    elem_rg = int(rg.gridSeeds_regrid.prod())
    
    New_RVE_size = rg.sizeRVE_regrid
    new_grid_cell = rg.gridSeeds_regrid
    origin0 = rg.originRVE_0
    Cell_coords = rg.gridCoords_cell_regrid   #or it should be after the periodic shift??
    print(Cell_coords.shape)

    ##------------------------------------------
    ## reading main inputs for processing
    #out5 = output_reader(hdf5_name)
    #inc, inc_key = out5.make_incerement(inc)
    ## reading main inputs
    d = damask.Result(rg.h5OutputName_0 + '.hdf5')
    phase_name = d.phases 
    d.view('increments',f'inc{inc}')
    orientations = d.read_dataset(d.get_dataset_location('O'))
    ### grain rotation
    grain_rotation = d.read_dataset(d.get_dataset_location('reorientation'))
    print('location',d.get_dataset_location('reorientation'))
    print('grain rot value',grain_rotation)
    print('grain rotation',grain_rotation.shape)
    print('nearest NN array',rg.get_nearestNeighbors())
    print('NN shape',rg.get_nearestNeighbors().shape)
    grain_rotation_rg = grain_rotation[rg.get_nearestNeighbors()]
    grain_rotation_rg_scalar = grain_rotation_rg
    mylist = []
    eulers = Orientation(orientations).as_Euler_angles(degrees=True)
    eulers_rg = eulers[rg.get_nearestNeighbors()]
    ### dislocation density
    rho =  d.read_dataset(d.get_dataset_location('tot_density'))
    ### reshaped
    rho_rg = rho[rg.get_nearestNeighbors()]
    ### subgrain sizes
    r_s    =  d.read_dataset(d.get_dataset_location('r_s'))
    r_s_rg   = r_s[rg.get_nearestNeighbors()]
    #print('orientation shape before: ',np.shape(orientations))
    # orientations regridded
    orientations_rg = orientations[rg.get_nearestNeighbors()] #hopefully it works
    print('orientation shape after: ',np.shape(orientations_rg))


    #--------------
    # make df
    #---------------
    
    df = pd.DataFrame() 
    # coords for new grids
    Cell_coords = Cell_coords#*1E-06
    print(Cell_coords)
    df['x'] = Cell_coords[:,0]
    df['y'] = Cell_coords[:,1]
    df['z'] = Cell_coords[:,2]
    ## initial grain
    df['grain'] = rg.materialFlatten_0[rg.nearestNeighbors_locs] #casipt starts counting from 1 I guess 
    ## Rotation
    df['Rotation'] = grain_rotation_rg_scalar
    # # total dislo
    df['rho'] = rho_rg
    # # subgrain sizes 
    df['r_s'] = r_s_rg
    # euler angles
    df['phi1'] = eulers_rg[:,0]
    df['PHI'] = eulers_rg[:,1]
    df['phi2'] = eulers_rg[:,2]

    header_f = '%s %s\n'%(str(int(np.prod(rg.gridSeeds_regrid))),str(max(df['grain']))) #,str(max(df['grain']))
    
    output = rg.h5OutputName_0 
    file_rg = '%s_inc%s.txt'%(output,inc)
    if not os.path.exists(os.path.join('/nethome/v.shah',folder,'postProc')):
        os.makedirs(os.path.join('/nethome/v.shah',folder,'postProc'))
    
    with open('postProc/'+file_rg,'w') as f:
        f.write(header_f)
        df.to_string(f,header=False,formatters=["{:.8f}".format,"{:.8f}".format,"{:.8f}".format, \
                                                "{:.8f}".format,"{:.8f}".format,"{:.6E}".format, \
                                                "{:.12f}".format,"{:.8f}".format,"{:.8f}".format, \
                                                "{:.8f}".format],index=False)
    
    ##--------------------------------
    ## make df for initial orientation
    ##--------------------------------
    #df_init = pd.DataFrame()
    #df_init['x'] = Cell_coords[:,0]
    #df_init['y'] = Cell_coords[:,1]
    #df_init['z'] = Cell_coords[:,2]

    ## open the dataset from initial increment
    #with h5py.File(rg.h5OutputName_0 + '.hdf5') as f:
    #  orientation_0 = np.array(f['/inc0/phase/{}/mechanics/O'.format(phase_name[0])]) 
    #orientation_0_rg = orientation_0[rg.get_nearestNeighbors()]
    #df_init['q0'] = orientation_0_rg[:,0] 
    #df_init['q1'] = orientation_0_rg[:,1] 
    #df_init['q2'] = orientation_0_rg[:,2] 
    #df_init['q3'] = orientation_0_rg[:,3] 
    #
    ##df_init.to_csv('postProc/Initial_orientation_regridded_inc{}'.format(inc),index=False,header=False)
    #np.savetxt('postProc/Initial_orientation_regridded_inc{}.txt'.format(inc),df_init.values)
   
    #create a hdf5 file
    #new_hdf_name = 'new_' + rg.h5OutputName_0 + 'inc' + str(inc) + '.hdf5'
    #hdf = h5py.File(new_hdf_name,'w')
    #hdf.attrs['DADF5_version_major'] = 0
    #hdf.attrs['DADF5_version_minor'] = 11
    #hdf.attrs['DAMASK_version'] = 'v3.0.0-alpha2-619-ga99983145'
    #hdf.create_group('geometry')
    #hdf['geometry'].attrs['cells'] = np.array(rg.gridSeeds_regrid, np.int32)
    #hdf['geometry'].attrs['size'] = np.array(rg.sizeRVE_regrid, np.float64)
    #hdf['geometry'].attrs['origin'] = np.array(rg.originRVE_0,np.float64)
    ##
    ###mapping data
    #comp_dtype = np.dtype([('Name',np.string_,64),('Position',np.int32)])
    #new_len    = np.prod(np.int32(rg.gridSeeds_regrid))
    #data_name  = phase_name*int(new_len)
    #data_value = [i for i in range(new_len)]
    #new_data   = list(zip(data_name,data_value))
    #print(new_data)
    #new_data   = np.array(new_data,dtype=comp_dtype)
    #new_data   = new_data.reshape(new_len,1)
    #dataset    = hdf.create_dataset("/mapping/phase",(new_len,1),comp_dtype)
    #dataset[...] = new_data

    ###orientation_rg
    ##comp_dtype  = np.dtype([('w',np.float64),('x',np.float64),('y',np.float64),('z',np.float64)])
    ###new_len     = np.prod(np.int32(rg.new_grid))
    ##new_len     = np.prod(np.int32(rg.gridSeeds_regrid))
    #dataset_ori = hdf.create_dataset("/inc{}/phase/{}/mechanics/O".format(inc,phase_name[0]),(new_len,4)) #will work for single phase only
    ##orientations_rg = np.array([tuple(i) for i in orientations_rg[:]],dtype=comp_dtype)
    #dataset_ori[...] = orientations_rg
    #hdf["/inc{}/phase/{}/mechanics/O".format(inc,phase_name[0])].attrs['Lattice'] = 'cF'
    #hdf["/inc{}/phase/{}/mechanics/O".format(inc,phase_name[0])].attrs['Unit'] = 'q_0 (q_1 q_2 q_3)'
    ##
    ###rho_rg and grain_rotation_rg_scalar
    #dataset_rho = hdf.create_dataset("/inc{}/phase/{}/plastic/tot_density".format(inc,phase_name[0]),(new_len,))
    #dataset_rho[...] = rho
    #
    #dataset_rot = hdf.create_dataset("/inc{}/phase/{}/plastic/reorientation".format(inc,phase_name[0]),(new_len,))
    #dataset_rot[...] = grain_rotation_rg_scalar
    ##
    #hdf.create_group('/inc{}/homogenization/{}/damage'.format(inc,d.homogenizations[0]))  #works for single homogenization currently
    #hdf.create_group('/inc{}/homogenization/{}/mech'.format(inc,d.homogenizations[0]))  #works for single homogenization currently
    #hdf.create_group('/inc{}/homogenization/{}/thermal'.format(inc,d.homogenizations[0]))  #works for single homogenization currently
    ##
    #hdf.create_dataset('/inc{}/geometry/u_n'.format(inc),data=np.zeros(new_len,3))
    #hdf.create_dataset('/inc{}/geometry/u_p'.format(inc),data=np.zeros(new_len,3))
    
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
    os.chdir('{}'.format(folder))
    dx = np.min(self.new_size/self.new_grid)#*1E-06
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
    
    nx = int(round(np.max(data[:,0])/dx))
    ny = int(round(np.max(data[:,1])/dx))
    nz = int(round(np.max(data[:,2])/dx))
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

    # extra for remeshing original orientation
    #data_for_ori = np.loadtxt('postProc/Initial_orientation_regridded_inc{}.txt'.format(\
    #                          os.path.basename(os.path.splitext(filename)[0]).split('inc')[1]),usecols=(3,4,5,6))
    #print(data_for_ori.shape)
    #new_data_for_ori = np.zeros((len(new_coords),7))
    #new_data_for_ori[:,0:3] = new_coords
    #new_data_for_ori[:,3:7] = griddata(data[:,0:3],data_for_ori[:,0:4],new_data_for_ori[:,0:3],method='nearest')

    #np.savetxt('postProc/remesh_Initial_orientation_inc{}.txt'.format(\
    #           os.path.basename(os.path.splitext(filename)[0]).split('inc')[1]),\
    #           new_data_for_ori)#,fmt = ' '.join(['%.10e']*3  + ['%.10f']*4))

    return nx,ny,nz
      
  def regrid_Initial_ori0(self,geom,load,inc,folder):
    """
    regrid the initial orientation for restart after the first trigger.

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
    os.chdir(folder)

    rg = geom_regridder(geom,load,increment=inc)
    rg.regrid_geom()
  
    self.new_size = rg.sizeRVE_regrid
    self.new_grid = rg.gridSeeds_regrid
   
    # Building the new coordinates
    elem0 = int(rg.gridSeeds_0.prod())
    elem_rg = int(rg.gridSeeds_regrid.prod())
    
    New_RVE_size = rg.sizeRVE_regrid
    new_grid_cell = rg.gridSeeds_regrid
    origin0 = rg.originRVE_0
    Cell_coords = rg.gridCoords_cell_regrid   #or it should be after the periodic shift??
    print(Cell_coords.shape)

    ##------------------------------------------
    ## reading main inputs for processing
    #out5 = output_reader(hdf5_name)
    #inc, inc_key = out5.make_incerement(inc)
    ## reading main inputs
    d = damask.Result(rg.h5OutputName_0 + '.hdf5')
    phase_name = d.phases 

    #--------------------------------
    # make df for initial orientation
    #--------------------------------
    df_init = pd.DataFrame()
    df_init['x'] = Cell_coords[:,0]
    df_init['y'] = Cell_coords[:,1]
    df_init['z'] = Cell_coords[:,2]

    # open the dataset from initial increment
    with h5py.File(rg.h5OutputName_0 + '.hdf5') as f:
      orientation_0 = np.array(f['/inc0/phase/{}/mechanics/O'.format(phase_name[0])]) 
    orientation_0_rg = orientation_0[rg.get_nearestNeighbors()]
    df_init['q0'] = orientation_0_rg[:,0] 
    df_init['q1'] = orientation_0_rg[:,1] 
    df_init['q2'] = orientation_0_rg[:,2] 
    df_init['q3'] = orientation_0_rg[:,3] 
    
    #df_init.to_csv('postProc/Initial_orientation_regridded_inc{}'.format(inc),index=False,header=False)
    np.savetxt('postProc/Initial_orientation_regridded_inc{}.txt'.format(inc),df_init.values)
  
  def regrid_Initial_ori_DRX(self,geom,load,inc,folder):
    """
    regrid the initial orientation for restart during DRX run.

    Parameters
    ----------
    geom : str
      Name of the geom file
    load : str
      Name of the load file
    inc : list
      Increment for which regridding is being done
    folder : str
      Path to the folder
    """ 

    isElastic = False
    scale = 1.0
    grid = False 
    os.chdir(folder)

    print(inc[-1])
    rg = geom_regridder(geom,load,increment=inc[-1].split('inc')[1])
    rg.regrid_geom()
  
    self.new_size = rg.sizeRVE_regrid
    self.new_grid = rg.gridSeeds_regrid
   
    # Building the new coordinates
    elem0 = int(rg.gridSeeds_0.prod())
    elem_rg = int(rg.gridSeeds_regrid.prod())
    
    New_RVE_size = rg.sizeRVE_regrid
    new_grid_cell = rg.gridSeeds_regrid
    origin0 = rg.originRVE_0
    Cell_coords = rg.gridCoords_cell_regrid   #or it should be after the periodic shift??
    print(Cell_coords.shape)

    ##------------------------------------------
    ## reading main inputs for processing
    #out5 = output_reader(hdf5_name)
    #inc, inc_key = out5.make_incerement(inc)
    ## reading main inputs
    #d = damask.Result(rg.h5OutputName_0 + '.hdf5')
    #phase_name = d.phases 

    #--------------------------------
    # make df for initial orientation
    #--------------------------------
    df_init = pd.DataFrame()
    df_init['x'] = Cell_coords[:,0]
    df_init['y'] = Cell_coords[:,1]
    df_init['z'] = Cell_coords[:,2]

    # open the dataset from initial increment
    #with h5py.File(rg.h5OutputName_0 + '.hdf5') as f:
    orientation_0 = np.loadtxt('postProc/remesh_Initial_orientation_{}.txt'.format(inc[-2]),usecols=(3,4,5,6)) 
    orientation_0_rg = orientation_0[rg.get_nearestNeighbors()]
    df_init['q0'] = orientation_0_rg[:,0] 
    df_init['q1'] = orientation_0_rg[:,1] 
    df_init['q2'] = orientation_0_rg[:,2] 
    df_init['q3'] = orientation_0_rg[:,3] 
    
    #df_init.to_csv('postProc/Initial_orientation_regridded_inc{}'.format(inc),index=False,header=False)
    np.savetxt('postProc/Initial_orientation_regridded_{}.txt'.format(inc[-1]),df_init.values)

  def remesh_Initial_ori0(self,filename,unit,folder): 
    """
    Remeshes the initial orientation to equidistant grid.
    
    Parameters
    ----------
    filename : str 
      file path
    unit : float
      Our units in comparison to DAMASK
    folder : str
      simulation folder
    """ 
    os.chdir('{}'.format(folder))
    dx = np.min(self.new_size/self.new_grid)#*1E-06
    print(dx)
    unit_scale = unit
    
    is2d = 0 # 1 for 2D data
    
    data = np.loadtxt(filename)
    min_x = unit_scale*np.min(data[:,0])
    min_y = unit_scale*np.min(data[:,1])
    min_z = unit_scale*np.min(data[:,2])

    #shift coords to start from 0
    data[:,0] = unit_scale*data[:,0] - min_x
    data[:,1] = unit_scale*data[:,1] - min_y
    data[:,2] = unit_scale*data[:,2] - min_z
    
    nx = int(round(np.max(data[:,0])/dx))
    ny = int(round(np.max(data[:,1])/dx))
    nz = int(round(np.max(data[:,2])/dx))
    
    x_new = np.mgrid[0:nx+1]*dx 
    y_new = np.mgrid[0:ny+1]*dx 
    z_new = np.mgrid[0:nz+1]*dx 
    
    new_coords = np.stack(np.meshgrid(x_new,y_new,z_new,indexing='ij'),axis=-1).reshape(((nx+1)*(ny+1)*(nz+1),3),order='F')
    
    # extra for remeshing original orientation
    data_for_ori = np.loadtxt('postProc/Initial_orientation_regridded_inc{}.txt'.format(\
                              os.path.basename(os.path.splitext(filename)[0]).split('inc')[1]),usecols=(3,4,5,6))
    print(data_for_ori.shape)
    new_data_for_ori = np.zeros((len(new_coords),7))
    new_data_for_ori[:,0:3] = new_coords
    new_data_for_ori[:,3:7] = griddata(data[:,0:3],data_for_ori[:,0:4],new_data_for_ori[:,0:3],method='nearest')

    np.savetxt('postProc/remesh_Initial_orientation_inc{}.txt'.format(\
               os.path.basename(os.path.splitext(filename)[0]).split('inc')[1]),\
               new_data_for_ori)#,fmt = ' '.join(['%.10e']*3  + ['%.10f']*4))

