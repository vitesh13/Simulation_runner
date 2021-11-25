#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
import os
from scipy import spatial
import h5py
from shutil import copy,move
from damask import Grid
from damask import ConfigMaterial as cm
from damask import Rotation
from damask import Orientation
from damask import tensor
from damask import mechanics

#--------------------------------------------------------------------------
Crystal_structures = {'fcc': 1,
                      'bcc': 1,
                      'hcp': 0,
                      'bct': 7,
                      'ort': 6} #TODO: is bct Tetragonal low/Tetragonal high?
Phase_types = {'Primary': 0} #further additions to these can be done by looking at 'Create Ensemble Info' filter
#--------------------------------------------------------------------------

def eulers_toR(eulers):
  """
  This function returns a rotation matrix from given euler angle
  The rotation matrix would be the rotational part of the elastic deformation gradient
  
  """
  r_matrix = np.zeros((3,3))
  c1 = math.cos(eulers[0])
  c2 = math.cos(eulers[1])
  c3 = math.cos(eulers[2])
  s1 = math.sin(eulers[0])
  s2 = math.sin(eulers[1])
  s3 = math.sin(eulers[2])

  # copying from function eu2om in rotations.f90
  r_matrix[0][0] =  c1*c3 - s1*c2*s3
  r_matrix[0][1] =  s1*c3 + c1*s3*c2 
  r_matrix[0][2] =  s3*s2

  r_matrix[1][0] = -c1*s3 - s1*c3*c2 
  r_matrix[1][1] = -s1*s3 + c1*c3*c2 
  r_matrix[1][2] =  c3*s2

  r_matrix[2][0] =  s1*s2
  r_matrix[2][1] = -c1*s2
  r_matrix[2][2] =  c2

  r_matrix = r_matrix.transpose()

  return r_matrix

def findFe_initial(F,Fp):
  """
  This function returns elastic deformation gradient from multiplicative decomposition. 
  Assumes, F = F_e F_p.
  
  """
  
  Fe = np.matmul(F,np.linalg.inv(Fp))
  return Fe

def om2eu(om):
  if abs(om[2][2]) < 1.0:
    zeta = 1.0/math.sqrt(1.0-om[2][2]**2.0)
    eu = np.array([math.atan2(om[2][0]*zeta,-om[2][1]*zeta), \
          math.acos(om[2][2]), \
          math.atan2(om[0][2]*zeta, om[1][2]*zeta)])
  else:
    eu = np.array([math.atan2(om[0][1],om[0][0]),0.5*math.pi*(1-om[2][2]),0.0])
  
  eu = np.where(eu<0.0,(eu+2.0*math.pi)%np.array([2.0*math.pi,math.pi,2.0*math.pi]),eu)
  
  return eu

def eu2qu(eu):
  ee = 0.5*eu

  cPhi = math.cos(ee[1])
  sPhi = math.sin(ee[1])
  P = -1.0
  qu =   np.array([   cPhi*math.cos(ee[0]+ee[2]), \
                   -P*sPhi*math.cos(ee[0]-ee[2]), \
                   -P*sPhi*math.sin(ee[0]-ee[2]), \
                   -P*cPhi*math.sin(ee[0]+ee[2])])

  if qu[0] < 0.0:
    qu = qu*(-1.0)

  return qu
 

def qu2om(qu):
  qq = qu[0]**2 - (qu[1]**2 + qu[2]**2 + qu[3]**2)
  om = np.zeros((3,3))
  om[0][0] = qq + 2.0*qu[1]*qu[1]
  om[1][1] = qq + 2.0*qu[2]*qu[2]
  om[2][2] = qq + 2.0*qu[3]*qu[3]

  om[0][1] = 2.0*(qu[1]*qu[2] - qu[0]*qu[3])
  om[1][2] = 2.0*(qu[2]*qu[3] - qu[0]*qu[1])
  om[2][0] = 2.0*(qu[3]*qu[1] - qu[0]*qu[2])
  om[1][0] = 2.0*(qu[2]*qu[1] + qu[0]*qu[3])
  om[2][1] = 2.0*(qu[3]*qu[2] + qu[0]*qu[1])
  om[0][2] = 2.0*(qu[1]*qu[3] + qu[0]*qu[2])

  return om 

# --------------------------------------------------------------------

class AttributeManagerNullterm(h5py.AttributeManager): 
  """
  Attribute management for DREAM.3D hdf5 files.
  
  String attribute values are stored as fixed-length string with NULLTERM
  
  References
  ----------
    https://stackoverflow.com/questions/38267076
    https://stackoverflow.com/questions/52750232

  """ 

  def create(self, name, data, shape=None, dtype=None):
    if isinstance(data,str):
      tid = h5py.h5t.C_S1.copy()
      tid.set_size(len(data + ' '))
      super().create(name=name,data=data+' ',dtype = h5py.Datatype(tid))
    else:
      super().create(name=name,data=data,shape=shape,dtype=dtype)
     

h5py._hl.attrs.AttributeManager = AttributeManagerNullterm # 'Monkey patch'


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
    self.CA_geom           = '.3D.geom'

  
  def geomFromCA(self,dx):
    """
    Creates geometry out of CA output.

    Parameters:
    -----------
    dx : float
      The grid spacing.
   
    """
    os.chdir(self.CA_folder)
    with open(self.CA_geom,'r') as f:
      grid_size = f.readline().split()[2:7:2] #read only first line

    grain_id = np.loadtxt(self.CA_geom,skiprows=1,usecols=(1)) + 1
    grid_size = np.array([int(i) for i in grid_size])
 
    n_elem = np.prod(grid_size)   #need to check if it is correct for 2D geometry

    size = grid_size*dx
    return Grid(grain_id.reshape(grid_size,order='F'),size)

    #data_to_geom = []
    #data_to_geom.append('{} header'.format(1))  #so that we can modify this later
    #data_to_geom.append('grid a {} b {} c {}'.format(*grid_size))
    #data_to_geom.append('size x {} y {} z {}'.format(*(grid_size*dx))) #give input that will kind of decide the spacing (like in DREAM3D)
    #data_to_geom.append('origin x 0.0 y 0.0 z 0.0')
    #data_to_geom.append('homogenization 1')
    
    #geom_data_1 = np.loadtxt('resMDRX.texture_MDRX.txt',skiprows=0,usecols=(2))
    #print(np.shape(geom_data_1),geom_data_1)
    #geom_data_1 = geom_data_1.tolist()
    #data_to_geom.append('microstructures {}'.format(int(np.amax(geom_data_1))+1))

    ##write microstructure part
    #data_to_geom.append('<microstructure>')
    #
    #for count,grain in enumerate(geom_data_1):
    #  data_to_geom.append('[Grain{:02d}]'.format(int(grain)+1))
    #  data_to_geom.append('crystallite 1')
    #  data_to_geom.append('(constituent) phase 1 texture {} fraction 1.0'.format(int(grain)+1))
    #
    ##write texture part
    #data_to_geom.append('<texture>')
    #texture_data = np.loadtxt('resMDRX.texture_MDRX.txt',skiprows=0,usecols=((4,6,8)))
    #
    #for i,texture in enumerate(texture_data):
    #  data_to_geom.append('[Grain{:02d}]'.format(i+1))
    #  data_to_geom.append('(gauss) phi1 {} Phi {} phi2 {} scatter 0.0 fraction 1.0'.format(*texture))

    ## calculating header length
    #header_value = len(data_to_geom) - 1
    #data_to_geom[0] = '{} header'.format(header_value)
    #geom_data = np.loadtxt('resMDRX.3D.geom',skiprows=1,usecols=(1))
    #numbers = geom_data.astype(int) + 1
    #numbers = numbers.reshape([grid_size[0],np.prod(grid_size[1:])],order='F').T
    #
    #for line in data_to_geom:
    #  print(line)
    

  def CAtoDREAM3D(self,dx):
    """
    Creates a dream3D file from CA output.
    This Dream3D file can then be used by DAMASK library to create the new geom and material.yaml.

    Parameters:
    -----------
    dx : float
      The grid spacing.
    """
    os.chdir(self.CA_folder)
    #--------------------------------------------------------------------------
    #Build array of euler angles for each cell
    #--------------------------------------------------------------------------
    cell_orientation_array  = np.loadtxt('..ang',skiprows=0,usecols=(0,1,2)) # ID phi1 phi phi2
    grain_orientation_array = np.loadtxt('.texture_MDRX.txt',skiprows=0,usecols=(4,6,8)) # ID phi1 phi phi2 
    
    with open(self.CA_geom,'r') as f:
      grid_size = f.readline().split()[2:7:2] #read only first line

    grid_size = np.array(grid_size,dtype=np.int32)
    print(type(grid_size))

    #--------------------------------------------------------------------------
    o = h5py.File('CA_output.dream3D','w')
    o.attrs['DADF5toDREAM3D'] = '1.0'
    o.attrs['FileVersion']    = '7.0'

    for g in ['DataContainerBundles','Pipeline']: # empty groups (needed)
      o.create_group(g)

    data_container_label = 'DataContainers/SyntheticVolumeDataContainer'        
    cell_data_label      = data_container_label + '/CellData'

    # Data phases
    o[cell_data_label + '/Phases'] = np.ones(tuple(np.flip(grid_size))+(1,),dtype=np.int32)

    # Data eulers
    orientation_data = cell_orientation_array.astype(np.float32)
    o[cell_data_label + '/Eulers'] = orientation_data.reshape(tuple(np.flip(grid_size))+(3,))

    # Attributes to CellData group
    o[cell_data_label].attrs['AttributeMatrixType'] = np.array([3],np.uint32)
    o[cell_data_label].attrs['TupleDimensions']     = np.array(grid_size,np.uint64)

    # Common Attributes for groups in CellData
    for group in ['/Phases','/Eulers']:
      o[cell_data_label + group].attrs['DataArrayVersion']      = np.array([2],np.int32)
      o[cell_data_label + group].attrs['Tuple Axis Dimensions'] = 'x={},y={},z={}'.format(*np.array(grid_size))
    
    # phase attributes
    o[cell_data_label + '/Phases'].attrs['ComponentDimensions'] = np.array([1],np.uint64)
    o[cell_data_label + '/Phases'].attrs['ObjectType']          = 'DataArray<int32_t>'
    o[cell_data_label + '/Phases'].attrs['TupleDimensions']     = np.array(grid_size,np.uint64)
    
    # Eulers attributes
    o[cell_data_label + '/Eulers'].attrs['ComponentDimensions'] = np.array([3],np.uint64)
    o[cell_data_label + '/Eulers'].attrs['ObjectType']          = 'DataArray<float>'        
    o[cell_data_label + '/Eulers'].attrs['TupleDimensions']     = np.array(grid_size,np.uint64)

    # Create EnsembleAttributeMatrix
    ensemble_label = data_container_label + '/CellEnsembleData'

    # Data CrystalStructures
    o[ensemble_label + '/CrystalStructures'] = np.uint32(np.array([999,1]))
    #                                                Crystal_structures[f.get_crystal_structure()]])).reshape((2,1))
    o[ensemble_label + '/PhaseTypes']        = np.uint32(np.array([999,Phase_types['Primary']])).reshape((2,1))

    # Attributes Ensemble Matrix
    o[ensemble_label].attrs['AttributeMatrixType'] = np.array([11],np.uint32)
    o[ensemble_label].attrs['TupleDimensions']     = np.array([2], np.uint64)

    # Attributes for data in Ensemble matrix
    for group in ['CrystalStructures','PhaseTypes']: # 'PhaseName' not required MD: But would be nice to take the phase name mapping
      o[ensemble_label+'/'+group].attrs['ComponentDimensions']   = np.array([1],np.uint64)
      o[ensemble_label+'/'+group].attrs['Tuple Axis Dimensions'] = 'x=2'
      o[ensemble_label+'/'+group].attrs['DataArrayVersion']      = np.array([2],np.int32)
      o[ensemble_label+'/'+group].attrs['ObjectType']            = 'DataArray<uint32_t>'
      o[ensemble_label+'/'+group].attrs['TupleDimensions']       = np.array([2],np.uint64)

    # Create geometry info
    geom_label = data_container_label + '/_SIMPL_GEOMETRY'
    
    o[geom_label + '/DIMENSIONS'] = np.int64(np.array(grid_size))
    o[geom_label + '/ORIGIN']     = np.float32(np.zeros(3))
    #o[geom_label + '/SPACING']    = np.float32(np.array(dummy)*4)
    o[geom_label + '/SPACING']    = np.float32(np.ones(3)*dx)
        
    o[geom_label].attrs['GeometryName']     = 'ImageGeometry'
    o[geom_label].attrs['GeometryTypeName'] = 'ImageGeometry'
    o[geom_label].attrs['GeometryType']          = np.array([0],np.uint32) 
    o[geom_label].attrs['SpatialDimensionality'] = np.array([3],np.uint32) 
    o[geom_label].attrs['UnitDimensionality']    = np.array([3],np.uint32) 

  def config_from_CA(self,simulation_folder):
    """
    Creates a material.yaml file

    Parameters
    ----------
    simulation_folder : str
      Path of the simulation_folder
    """
    ori = np.loadtxt('.texture_MDRX.txt',usecols=(4,6,8))
    ori = Rotation.from_Euler_angles(ori).as_quaternion().reshape(-1,4)
    base_config = cm.load('{}/material.yaml'.format(simulation_folder))
    phase = np.array([base_config['material'][1]['constituents'][0]['phase']]*len(ori))
    idx = np.arange(len(ori))
    constituent = {k:np.atleast_1d(v[idx].squeeze()) for k,v in zip(['O','phase'],[ori,phase])}
    new_config = base_config.material_add(**constituent,homogenization='direct')
    new_config.save()
 
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
    os.chdir(os.path.dirname(hdf))
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

    hdf = h5py.File(hdf)

    names_in_big_groups_dict = {}
    big_groups = ['homogenization','phase']
    for i in big_groups:
      names_in_big_groups_dict[i] = hdf[i].keys()

    phase_name = [i for i in names_in_big_groups_dict['phase']]
    homog_name = [i for i in names_in_big_groups_dict['homogenization']]

    values_in_phase = ['F','F_i','F_p','L_i','L_p','S','omega']

    const_values_in_solver = ['C_minMaxAvg','C_volAvg','C_volAvgLastInc','F_aim','F_aimDot','F_aim_lastInc','P_aim']
    changed_values_in_solver = ['F','F_lastInc']
    
    # creating solver group
    for i in const_values_in_solver:
      f.create_dataset('/solver/' + i,data=np.array(hdf['/solver/' + i]))

    xsize      = int(round(np.max(remeshed_coords[:,0])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
    ysize      = int(round(np.max(remeshed_coords[:,1])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
    zsize      = int(round(np.max(remeshed_coords[:,2])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
    totalsize  = int(xsize*ysize*zsize)

    for i in changed_values_in_solver:
      if i == 'F':
        data2write = np.broadcast_to(np.eye(3).flatten(),tuple((zsize,ysize,xsize,9)))
        f.create_dataset(f'solver/{i}',data = data2write)
      if i == 'F_lastInc':
        data2write = np.broadcast_to(np.eye(3), tuple((zsize,ysize,xsize,3,3)))
        f.create_dataset(f'solver/{i}',data = data2write)
    
    # creating homogenization group
    for h in homog_name: 
      f.create_group('homogenization/{}'.format(h))

    # creating phase group
    for p in phase_name: 
      f.create_group('phase/{}'.format(p))

# -------------------------------------------------------------------------------------------------------------------------------
      for i in values_in_phase:
        data_array = np.zeros((totalsize,) + np.shape(hdf[f'/phase/{p}/' + i])[1:])
        data_array[:] = np.array(hdf[f'/phase/{p}/' + i])[nbr_array]
        f[f'/phase/{p}/' + i] = data_array


  def Initialize_Fp(self,restart_file_CA,casipt_output,remesh_file,ang_file,rho_file):
      """
      Modifies the orientation and deformation gradients in the transformed parts.

      Re_0 = O^T
      F_p  = Re_0 = O^T
      originally F_p should be O, however, as hdf5 restart stores the arrays in transposed form, 
      we can F_p = O^T

      Parameters
      ----------
      restart_file_CA : str
        Path of the restart hdf5 file formed after neighbour search
      casipt_output : str
        Path of the casipt file containing info about transformed points (resMDRX.MDRX.txt)
      remesh_file : str
        Path of the remesh file.
      ang_file : str
        Path of the ang file from CASIPT.
      rho_file : str
        Path of the rho file from CASIPT.
      """
      ori_after_CA = np.loadtxt(ang_file)
      rho_CA       = np.loadtxt(rho_file)
      #data = np.loadtxt(casipt_output,usecols=((1,3,5,7,9)))

      hdf_file = h5py.File(restart_file_CA,'a')

      orig_rho = np.loadtxt(remesh_file,skiprows=1,usecols=((5))) 
      ratio = rho_CA/orig_rho
      phase_name = [i for i in hdf_file['phase'].keys()][0]

      for i in range(24):
        hdf_file['/phase/{}/omega'.format(phase_name)][:,i] = hdf_file['/phase/{}/omega'.format(phase_name)][:,i]*ratio # for BCC till 48, but for fcc till 24 only  

      Re_0 = tensor.transpose(Rotation.from_Euler_angles(ori_after_CA).as_matrix())  #convert euler angles to rotation matrix

       
      data = hdf_file['/phase/{}/F_p'.format(phase_name)]
      data[...] = Re_0
      hdf_file.close()

        
  def Initialize_Fp_no_regridding(self,restart_file,inc,remesh_file,ang_file,rho_file):
      """
      Modifies the orientation and deformation gradients in the transformed parts.
      This function is for the case when there is no regridding done before CASIPT.

      Parameters
      ----------
      restart_file : str
        Path of the restart hdf5 file
      inc : str
        Increment at which restart is done.
      remesh_file : str
        Path of the remesh file.
      ang_file : str
        Path of the ang file from CASIPT.
      rho_file : str
        Path of the rho file from CASIPT.
      """
      ori_after_CA = np.loadtxt(ang_file)
      rho_CA       = np.loadtxt(rho_file)

      print(os.path.splitext(os.path.basename(restart_file))[0] + '_regridded_{}_CA.hdf5'.format(inc))
      copy(restart_file,os.path.dirname(restart_file) + '/' + \
           os.path.splitext(os.path.basename(restart_file))[0] + '_regridded_{}_CA.hdf5'.format(inc)) 
      hdf_file = h5py.File(os.path.dirname(restart_file) + '/' + \
                           os.path.splitext(os.path.basename(restart_file))[0] + '_regridded_{}_CA.hdf5'.format(inc),'a')

      orig_rho = np.loadtxt(remesh_file,skiprows=1,usecols=((5))) 
      ratio = rho_CA/orig_rho
      phase_name = [i for i in hdf_file['phase'].keys()][0]

      for i in range(24):
        hdf_file['/phase/{}/omega'.format(phase_name)][:,i] = hdf_file['/phase/{}/omega'.format(phase_name)][:,i]*ratio # for BCC till 48, but for fcc till 24 only  

      Re_0 = tensor.transpose(Rotation.from_Euler_angles(ori_after_CA).as_matrix())  #convert euler angles to rotation matrix, O^T
       
      F_stored = tensor.transpose(np.array(hdf_file['/phase/{}/F'.format(phase_name)]))  #taking transpose as hdf5 restart transposes while storing
      Fp_stored = tensor.transpose(np.array(hdf_file['/phase/{}/F_p'.format(phase_name)])) #taking transpose as hdf5 restart transposes while storing 
      Fp_inv = np.linalg.inv(Fp_stored) 
      Fe_stored = np.matmul(F_stored,Fp_inv)

      (Re,Ue) = mechanics._polar_decomposition(Fe_stored,['R','U'])   #Fe = Re,Ue          
      Fe_modified = np.matmul(Re_0,Ue)

      Fp_stored_modified_T = np.matmul(np.linalg.inv(Fe_modified),F_stored)

      # normalize Fp
      for i in range(len(Fp_stored_modified_T)):
        Fp_stored_modified_T[i] = Fp_stored_modified_T[i]/(np.linalg.det(Fp_stored_modified_T)**(1.0/3.0))      

      Fp_stored_modified = tensor.transpose(Fp_stored_modified_T)

      data = hdf_file['/phase/{}/F_p'.format(phase_name)]
      data[...] = Fp_stored_modified
      hdf_file.close()

