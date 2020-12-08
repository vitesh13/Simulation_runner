#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import numpy as np
import os

class CASIPT_postprocessing():
  """
  Class combining different scripts for CASIPT post-processing.
  
  """
  def __init__(self):
    """ Sets the files names and folder names."""
    
    self.simulation_folder = 'DRX_sample_simulations'
    self.CA_folder         = 'DRX_sample_simulations/Industrial_finishing_mill/1_stand/CA_files/1.0'
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
    np.savetxt('test1.geom',array,fmt='%s',newline='\n') 


