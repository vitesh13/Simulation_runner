#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
import os
from scipy import spatial
import h5py
from Fe_decomposition import Decompose

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
    os.chdir(self.CA_folder)
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
    numbers = geom_data.astype(int) + 1
    numbers = numbers.reshape([grid_size[0],np.prod(grid_size[1:])],order='F').T
    
    for line in data_to_geom:
      print(line)
    
    np.savetxt('test.geom',numbers,fmt='%s',newline='\n',header='\n'.join(data_to_geom),comments='') 

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
    
    hdf = h5py.File(hdf)

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
        data_array[:] = np.array(hdf[i])[nbr_array]
        f[i] = data_array  
      else:
        data_array = np.zeros((len(remeshed_coords),) + np.shape(hdf['/constituent/' + i])[1:])
        data_array[:] = np.array(hdf['/constituent/' + i])[nbr_array]
        f['/constituent/' + i] = data_array  

    for i in diff_values_1:
        xsize      = int(round(np.max(remeshed_coords[:,0])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
        ysize      = int(round(np.max(remeshed_coords[:,1])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
        zsize      = int(round(np.max(remeshed_coords[:,2])/(remeshed_coords[1,0] - remeshed_coords[0,0]))) #+ 1
        totalsize  = int(xsize*ysize*zsize)
        data_array = np.zeros((totalsize,) + np.shape(hdf['/solver/' + i])[3:])
        input_data = np.array(hdf['/solver/' + i]).reshape(((np.prod(np.shape(np.array(hdf['/solver/' + i]))[0:3]),)+np.shape(np.array(hdf['/solver/' + i]))[3:]))
        data_array[:] = input_data[nbr_array]
        data_array = data_array.reshape((zsize,ysize,xsize,) + np.shape(hdf['/solver/' + i])[3:])
        f['/solver/' + i] = data_array

  def Initialize_Fp(self,restart_file_CA,casipt_output,remesh_file):
      """
      Modifies the orientation and deformation gradients in the transformed parts.

      Parameters
      ----------
      restart_file_CA : str
        Path of the restart hdf5 file formed after neighbour search
      casipt_output : str
        Path of the casipt file containing info about transformed points (resMDRX.MDRX.txt)
      remesh_file : str
        Path of the remesh file.
      """
      data = np.loadtxt(casipt_output,usecols=((1,3,5,7,9)))
      hdf_file = h5py.File(restart_file_CA,'a')

      orig_rho = np.loadtxt(remesh_file,skiprows=1,usecols=((5))) 
      orig_rho = np.loadtxt('resMDRX._rho.txt')
      ratio = rho_CA/orig_rho

      for i in range(24):
        hdf_file['/constituent/1_omega_plastic'][:,i] = hdf_file['/constituent/1_omega_plastic'][:,i]*ratio # for BCC till 48, but for fcc till 24 only  

      for i in data:
        hdf_file['/constituent/1_omega_plastic'][i[0],0:24] = 5E11 # for BCC till 48, but for fcc till 24 only  
        Fp = np.array(hdf_file['Fp'][i[0]]).reshape((3,3))
        F  = np.array(hdf_file['F'][i[0]]).reshape((3,3))
        Fe = findFe_initial(F.T,Fp.T) # because restart file stores deformation gradients as transposed form 
        d = Decompose(Fe)
        R = d.math_rotationalPart33(Fe)  #rotational part of Fe = RU
        orig_eulers = om2eu(R.transpose())   #in radians O_m = R.transpose()
        stretch = np.matmul(np.linalg.inv(R),Fe)
        eulers = i[2:5]  
        eulers = eulers*math.pi/180.0 #degrees to radians
        rotation_new = eulers_toR(eulers) #you get rotation matrix R from this function 
        Fe_new       = np.matmul(rotation_new,stretch)
        Fp_new       = np.matmul(F,np.linalg.inv(Fe_new))
        Fp_new       = Fp_new.T           # because restart file stores deformation gradients as transposed form
        hdf_file['Fp'][i[0]] = Fp_new.reshape((1,1,3,3))

        

