#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import os
import subprocess
import shlex
from shutil import copyfile
from shutil import copy
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as PyPlot
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import pandas as pd
import numpy as np
import h5py

class Multi_stand_runner():
  """
  Class containing functions to carry out multi-stand simulations.

  """
  #simulation file names etc
  def __init__(self):
    """ Sets the files names and folder names."""
    self.sample_folder = 'DRX_sample_simulations/example_microstructure_2/small_scale_Shen_2016' #need to change this folder when doing other stands
    self.simulation_folder = 'DRX_sample_simulations'
    self.geom_file = '3D_small_scale_95_microns.geom'
    self.load_file = 'tensionX_10.load'
    self.config_file = 'material.config'
    self.extra_config = 'ho_cr_ph.config'
    self.job_file = '{}_{}.hdf5'.format(self.geom_file.split('.')[0],self.load_file.split('.')[0])
    self.restart_file = '{}_{}_0.hdf5'.format(self.geom_file.split('.')[0],self.load_file.split('.')[0])

# run simulations to get the files
  def run_fresh_simulation(self,simulation_folder,sample_folder,geom_file,load_file,config_file,extra_config):
    os.chdir('/nethome/v.shah/{}/'.format(sample_folder))
    copy(geom_file, '/nethome/v.shah/{}/'.format(simulation_folder))
    copy(load_file,'/nethome/v.shah/{}/'.format(simulation_folder))  
    copy(config_file,'/nethome/v.shah/{}/'.format(simulation_folder))  
    copy(extra_config,'/nethome/v.shah/{}/'.format(simulation_folder))  
    os.chdir('/nethome/v.shah/{}/'.format(simulation_folder))
    simulation = subprocess.run(shlex.split('screen -dm bash -c "DAMASK_spectral -l {} -g {} > check.txt"'.format(load_file,geom_file)))

# copy output files to avoid issues
  def copy_output(self,stand,simulation_folder,sample_folder,job_file,restart_file,geom_file,load_file,config_file,extra_config):
  
    """ 
    Copies the output files to a safe folder to avoid any issues and to have a backup. 
    
    Parameters
    ----------
    stand : int
      stand at which deformation happens.
    sample_folder : string
      folder where data gets stored.
    """
    os.chdir('/nethome/v.shah/{}/'.format(simulation_folder))
    subprocess.run(shlex.split('mkdir nethome/v.shah/{}/{}_stand'.format(sample_folder,stand)))
    storage = '{}/{}_stand/'.format(sample_folder,stand)
    copy(job_file, '/nethome/v.shah/{}/'.format(storage))
    copy(restart_file, '/nethome/v.shah/{}/'.format(storage))
    copy(geom_file, '/nethome/v.shah/{}/'.format(storage))
    copy(load_file, '/nethome/v.shah/{}/'.format(storage))
    copy(config_file, '/nethome/v.shah/{}/'.format(storage))
    copy(extra_config, '/nethome/v.shah/{}/'.format(storage))

  def change_GIT_branch(self,branch_name):
    """
    Attempts to change the GIT branch.

    Parameters
    ----------
    branch_name : str
      Name of the GIT branch to which to change.

    """
    os.chdir('/nethome/v.shah/DAMASK/')
    try:
      subprocess.run(shlex.split('git checkout {}'.format(branch_name)))
    except:
      os.chdir('/nethome/v.shah/{}/'.format(simulation_folder))
      print("Could not change the git branches, do it manually")
     

# initial processing
  def Initial_processing(self,job_file,simulation_folder):
    """
    Initial post processing required for MDRX simulations.
    
    Parameters
    ----------
    job_file : str
      Name of the damask output file to be processed.
    simulation_folder : str
      Name of the simulation folder where the job file exists.

    """
    import damask 
    os.chdir('/nethome/v.shah/{}/'.format(simulation_folder))
    d = damask.Result('{}.hdf5'.format(job_file.split('.')[0]))
    orientation0 = d.get_initial_orientation()
    d.add_grainrotation(orientation0)
    d.add_Eulers('orientation')
    d.add_calculation('tot_density','np.sum((np.sum(#rho_mob#,1),np.sum(#rho_dip#,1)),0)')
    d.add_calculation('r_s',"40/np.sqrt(#tot_density#)")

# prepare for re-gridding

  def regridding_processing(self,geom_file,load_file):
    """
    Prepare data for regridding and perform regridding.
 
    Parameters
    ----------
    geom_file : str
      Name of the geom file
    load_file : str
      Name of the load file

    """
    
    subprocess.run(shlex.split('regrid -g {} -l {}'.format(geom_file,load_file)))
    

# create stress-strain curves
  def plot_stress_strain(self,stands,sample_folder,job_file):
    """
    Plot stress strain curves of multi-stand rolling.

    Parameters
    ----------
    stands : int
      Number of stands being considered in the simulation.
    sample_folder : str
      Name of the sample folder.
    job_file : str
      Name of the output file.

    """
    
    import damask
    fig = PyPlot.figure()
    max_strain = np.zeros(1) 
    colormap = PyPlot.cm.gist_ncar #nipy_spectral, Set1,Paired   
    colors = [colormap(i) for i in np.linspace(0, 1,stands)]
    for i in range(1,stands+1):

      d = damask.Result('/nethome/v.shah/{}/{}_stand/{}'.format(sample_folder,i,job_file))
      #d.add_Cauchy()
      #d.add_strain_tensor()
      #d.add_Mises('sigma')
      #d.add_Mises('epsilon_V^0.0(F)')
      #
      #d.add_calculation('avg_sigma',"np.average(#sigma_vM#)")
      #d.add_calculation('avg_epsilon',"np.average(#epsilon_V^0.0(F)_vM#)")
      print(d.fname)  
      stress_path = d.get_dataset_location('avg_sigma')
      stress = np.zeros(len(stress_path)) 
      strain = np.zeros(len(stress_path)) 
      hdf = h5py.File(d.fname)
      
      for count,path in enumerate(stress_path):
        stress[count] = np.array(hdf[path])
        strain[count] = np.array(hdf[path.split('avg_sigma')[0] + 'avg_epsilon'])
      
      stress = np.array(stress)
      strain = (1.0 + np.array(strain))*(1.0 + max_strain) - 1.0

      PyPlot.plot(strain,stress,linestyle = '-',linewidth='2.5',label='{} hit'.format(i))
      PyPlot.xlabel(r'$\varepsilon_{VM} $',fontsize=18)
      PyPlot.ylabel(r'$\sigma_{VM}$ (Pa)',fontsize=18)
      axes = PyPlot.gca()

      max_strain = np.max(strain)
    
    expt_data = pd.read_csv('/nethome/v.shah/{}/Shen_2016_fig2.csv'.format(sample_folder),names=['strain','stress'],skiprows=4,nrows=28)
    PyPlot.plot(expt_data['strain'],expt_data['stress']*1E6,'o',label='Experimental data')
    PyPlot.legend()
    fig.savefig('/nethome/v.shah/{}/Multi_stand_stress_strain.png'.format(sample_folder))

