#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import os,re
import subprocess,signal,time
import psutil
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
import math
import h5py
import xml.etree.ElementTree as ET
import damask

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
    self.sta_file = '{}_{}.sta'.format(self.geom_file.split('.')[0],self.load_file.split('.')[0])
    self.restart_file = '{}_{}_0.hdf5'.format(self.geom_file.split('.')[0],self.load_file.split('.')[0])
    self.tmp = 'tmp_storage'

# run simulations to get the files
  def run_fresh_simulation(self,simulation_folder,sample_folder,geom_file,load_file,config_file,proc):
    """
    Runs a fresh simulation.

    Parameters
    ----------
    simulation_folder: str
      Full path to the folder where the simulation is performed
    sample_folder: str
      Full path to folder where the results/config files are stored
    geom_file: str
      Name of the geom file
    load_file: str
      Name of the load file
    config_file: str
      Name of the config (yaml) file.
    proc : int
      Number of processors to be used.
    """
    os.chdir('{}'.format(sample_folder))
    copy(geom_file, '{}'.format(simulation_folder))
    copy(load_file,'{}'.format(simulation_folder))  
    copy(config_file,'{}'.format(simulation_folder))  
    os.chdir('{}'.format(simulation_folder))
    cmd = 'mpiexec -n {} DAMASK_grid -l {} -g {} > check.txt'.format(proc,load_file,geom_file)
    p = subprocess.Popen(cmd,shell=True)
    while p.poll() == None:
      p.poll()
    return p.poll()

  def run_and_monitor_simulation(self,simulation_folder,sample_folder,geom_file,load_file,config_file,proc):
    """
    Runs and monitors a fresh simulation.

    Parameters
    ----------
    simulation_folder: str
      Full path to the folder where the simulation is performed
    sample_folder: str
      Full path to folder where the results/config files are stored
    geom_file: str
      Name of the geom file
    load_file: str
      Name of the load file
    config_file: str
      Name of the config (yaml) file.
    proc : int
      Number of processors to be used.
    """
    os.chdir('{}'.format(sample_folder))
    copy(geom_file, '{}'.format(simulation_folder))
    copy(load_file,'{}'.format(simulation_folder))  
    copy(config_file,'{}'.format(simulation_folder))  
    os.chdir('{}'.format(simulation_folder))
    cmd = 'mpiexec -n {} DAMASK_grid -l {} -g {}'.format(proc,load_file,geom_file)
    with open('check.txt','w') as f:
      P = subprocess.Popen(cmd,stdout = subprocess.PIPE, stderr = subprocess.PIPE,shell=True)
      r = re.compile(' increment [0-9]+ converged') 
      record = []
      growth_length = 0.0
      while P.poll() is None:
        for count,line in enumerate(iter(P.stdout.readline, b'')):
          record.append(line.decode('utf-8'))
          if re.search(r, record[-1]):
            P.send_signal(signal.SIGSTOP)
            print(record[-1])
            time.sleep(1)       # needed for avoid clash of fortran and python accessing the same file
            try:
                velocity = self.calc_velocity(self.calc_delta_E(record[-1],32E9,2.5E-10),5E-10)  #needs G, b and mobility  
            except OSError:
                time.sleep(10)
                velocity = self.calc_velocity(self.calc_delta_E(record[-1],32E9,2.5E-10),5E-10)  #needs shear modulus, b and mobility  
            growth_length = growth_length + velocity*self.calc_timeStep(record[-1]) 
            print(growth_length)
            if growth_length*10.0 >= self.get_min_resolution():
              print(record[-1])
              P.send_signal(signal.SIGUSR2)
              P.send_signal(signal.SIGUSR1)
              print('about to continue')
              P.send_signal(signal.SIGCONT)
              for children in psutil.Process(P.pid).children(recursive=True):
                  print(children)
                  if children.name() == 'DAMASK_grid':
                     children.terminate()
              gone, alive = psutil.wait_procs(psutil.Process(P.pid).children(recursive=True),timeout=0.5)
              for living in alive:
                  living.kill()
            else:
              os.kill(P.pid+1, signal.SIGCONT)
      for line in record:
        f.write(line)
          
  def calc_delta_E(self,inc_string,G,b):
    """
    Calculates the max stored energy difference.
    Assumes that every increment is being recorded. 

    Parameters
    ----------
    inc_string: str
      String of type 'increment [0-9]+ converged.
    G : float
      Shear modulus (Pa)
    b : float
      Burgers vector
    """
    d = damask.Result(self.job_file)
    converged_inc = inc_string.split()[1]
    recorded_inc  = int(converged_inc) - 1
    d.view('increments',f'inc{recorded_inc}')
    path = d.get_dataset_location('rho_mob')[0]
    rho_mob = d.read_dataset([path])
    rho_dip = d.read_dataset([path.split('rho_mob')[0] + 'rho_dip'])
    tot_rho_array = np.sum((np.sum(rho_mob,1),np.sum(rho_dip,1)),0)
    max_rho = np.max(tot_rho_array)
    avg_rho = np.average(tot_rho_array)
    diff_rho = max_rho - avg_rho
    
    delta_E  = G*(b**2.0)*diff_rho
    return delta_E


  def calc_velocity(self,delta_E,M):
    """
    Calculates velocity of the interface.

    Parameters
    ----------
    delta_E : float
      Difference in stored energy across an interface
    M : float
      Mobility factor.
    """
    return M*delta_E

  def calc_timeStep(self,inc_string):
    """
    Calculate the current time step in damask.

    Parameters
    ----------
    inc_string: str
      String of type 'increment [0-9]+ converged.
    """
    from damask import Config
    loading = Config.load(self.load_file)
    total_steps = len(loading['loadstep'])
    steps_list = [loading['loadstep'][i]['discretization']['N'] for i in range(len(loading['loadstep']))]
    summed_incs = np.array([sum(steps_list[:i]) for i in range(len(steps_list)+1)][1:])
    converged_inc = int(inc_string.split()[1])
    relevant_step = np.where(summed_incs > converged_inc)[0][0] 
    time_step = loading['loadstep'][relevant_step]['discretization']['t']/\
                loading['loadstep'][relevant_step]['discretization']['N']
    return time_step

  def get_min_resolution(self):
    """
    Get the minimum grid resolution. 

    Parameters
    ----------
    geom_file : str
      Name of the geom file
    """
    from damask import Grid
    geom = Grid.load(self.geom_file)
    return np.min(geom.size/geom.cells)





# modify files after CA
  def copy_modified_files(self,new_geom,new_restart,old_geom,old_restart):
    """
    Copies the new geometry and the new restart file generated by CA post-processing.

    Parameters
    ----------
    new_geom : str
      Name of the new geom file
    new_restart :  str
      Name of the new restart file
    old_geom : str
      Name of the old geom file
    old_restart :  str
      Name of the old restart file
    """
    
    copyfile(new_geom,'/nethome/v.shah/{}/{}'.format(simulation_folder,old_geom))
    copyfile(new_restart,'/nethome/v.shah/{}/{}'.format(simulation_folder,old_restart))

  def copy_CA_output(self,path_to_CA,sample_folder,stand,time):
    """
    Copies the important data output from the CA code for safekeeping.

    Parameters
    ----------
    path_to_CA : str
      Path of the folder where the CA output is stored.
    sample_folder : str
      Path of the sample folder
    stand : integer
      Stand after which CA is run
    time : float
      times for which CA is run
    """
 
    os.chdir(path_to_CA)
    if not os.path.exists('/nethome/v.shah/{}/{}_stand/CA_files/'.format(sample_folder,stand)):
      os.mkdir('/nethome/v.shah/{}/{}_stand/CA_files/'.format(sample_folder,stand))
    os.mkdir('/nethome/v.shah/{}/{}_stand/CA_files/{}'.format(sample_folder,stand,time))
    storage = '{}/{}_stand/CA_files/{}'.format(sample_folder,stand,time)
    copy('resMDRX..ang','/nethome/v.shah/{}'.format(storage))
    copy('resMDRX._rho.txt','/nethome/v.shah/{}'.format(storage))
    copy('resMDRX.3D.geom','/nethome/v.shah/{}'.format(storage))
    #copy('resMDRX.final.map.xy.dat','/nethome/v.shah/{}'.format(storage))
    copy('resMDRX.fractions.txt','/nethome/v.shah/{}'.format(storage))
    copy('resMDRX.MDRX.txt','/nethome/v.shah/{}'.format(storage))
    copy('resMDRX.texture_MDRX.txt','/nethome/v.shah/{}'.format(storage))
    copy('resMDRX.final.casipt','/nethome/v.shah/{}'.format(storage))

# modify the load file after CA
  def modify_load_file(self,load_file,config_file):
    """
    Modifies the load file for the next increment. 
    Need to give all the load cases at the start.
    This part will only remove the first commented out part. 

    Parameters
    ---------
    load_file : str
      Name of the load file to be modified.

    """
    with open(load_file,'r') as f:
      load_cases = f.readlines()

    for load in range(len(load_cases)):
      if re.search(r'^#',load_cases[load]):
        load_cases[load] = load_cases[load].split('#')[1]  
    
    print(load_cases)

    with open(load_file,'w') as f:
      for line in load_cases:
        f.write(line)
  

# run restart simulation after CA
  def run_restart_simulation(self,simulation_folder,sample_folder,geom_file,load_file,config_file,extra_config,restart_inc):
    """
    Runs restart simulation after CA.

    Parameters
    ----------
    restart_inc : int
      Number at which restart will start.
    """
    #os.chdir('/nethome/v.shah/{}/'.format(sample_folder))
    #copy(geom_file, '/nethome/v.shah/{}/'.format(simulation_folder))
    #copy(load_file,'/nethome/v.shah/{}/'.format(simulation_folder))  
    #copy(config_file,'/nethome/v.shah/{}/'.format(simulation_folder))  
    #copy(extra_config,'/nethome/v.shah/{}/'.format(simulation_folder))  
    os.chdir('/nethome/v.shah/{}/'.format(simulation_folder))
    cmd = 'DAMASK_spectral -l {} -g {} -r {} > check.txt'.format(load_file,geom_file,restart_inc)
    p = subprocess.Popen(cmd,shell=True)
    while p.poll() == None:
      p.poll()
    return p.poll()

# copy output files to avoid issues
  def copy_output(self,stand,simulation_folder,sample_folder,job_file,restart_file,geom_file,load_file,config_file,extra_config,sta_file,make_dir = True):
  
    """ 
    Copies the output files to a safe folder to avoid any issues and to have a backup. 
    
    Parameters
    ----------
    stand : int
      stand at which deformation happens.
    simulation_folder : str
      Folder where the simulation was done. 
    sample_folder : string
      folder where data gets stored.
    make_dir : bool
      Default true. False if you dont to make directory.
    """
    os.chdir('/nethome/v.shah/{}/'.format(simulation_folder))
    if make_dir:
      os.mkdir('/nethome/v.shah/{}/{}_stand/'.format(sample_folder,stand))
    storage = '{}/{}_stand/'.format(sample_folder,stand)
    copy(job_file, '/nethome/v.shah/{}/'.format(storage))
    copy(sta_file, '/nethome/v.shah/{}/'.format(storage))
    copy(restart_file, '/nethome/v.shah/{}/'.format(storage))
    copy(geom_file, '/nethome/v.shah/{}/'.format(storage))
    copy(load_file, '/nethome/v.shah/{}/'.format(storage))
    copy(config_file, '/nethome/v.shah/{}/'.format(storage))
    copy(extra_config, '/nethome/v.shah/{}/'.format(storage))

  def save_rawfiles(self,stand,simulation_folder,sample_folder,job_file,restart_file,geom_file,load_file,config_file,extra_config,sta_file):
    """
    Keeps the raw unprocessed HDF5 files (in case processing goes wrong).
    Parameters
    ----------
    stand : int
      stand at which deformation happens.
    sample_folder : string
      folder where data gets stored.
    """
    os.chdir('/nethome/v.shah/{}/'.format(simulation_folder))
    os.mkdir('/nethome/v.shah/{}/{}_stand/raw_files/'.format(sample_folder,stand))
    storage = '{}/{}_stand/raw_files'.format(sample_folder,stand)
    copy(job_file, '/nethome/v.shah/{}/'.format(storage))
    copy(sta_file, '/nethome/v.shah/{}/'.format(storage))
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
    
# modify the CA xml file
  def modify_CA_setting(self,filename,T,grid,delta_t,dx,start_file,basefn):
    """
    Modifies the XML file to run the CA code.

    Parameters
    ----------
    filename : str
      Path of the XML file
    T : float
      Temperature for CA simulation
    grid : list or np.array
      Grid size of the RVE
    delta_t : float
      Run time of the simulation
    dx : float
      Spacing of the CA grid
    start_file : str
      Path of the input text file for CA
    basefn : str
      Path of the folder where the output of CA is stored

    """
    
    tree = ET.parse(filename)
    root = tree.getroot()
    
    root.find('T0').text = str(T)
    root.find('nx').text = str(grid[0])
    root.find('ny').text = str(grid[1])
    root.find('nz').text = str(grid[2])
    root.find('deltats').text = '{:.8f}'.format(delta_t)
    root.find('dx').text = '{:.8f}'.format(dx)
    root.find('startfile_fn').text = start_file
    root.find('basefn').text = basefn
    tree.write(filename)
    
  def perform_CA(self,input_settings):
    """
    Starts a CA simulation.
 
    Parameters
    ----------
    input_settings : str
      Name of the input xml file for CA
    """
   
    os.chdir('/nethome/v.shah/casipt/casipt/')
    subprocess.run(shlex.split('bin/casipt input/{}'.format(input_settings)))  



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
    print(colors)
    for i in range(1,stands+1):

      d = damask.Result('/nethome/v.shah/{}/{}_stand/{}'.format(sample_folder,i,job_file))
      d.add_Cauchy()
      d.add_strain_tensor()
      d.add_Mises('sigma')
      d.add_Mises('epsilon_V^0.0(F)')
      
      d.add_calculation('avg_sigma',"np.average(#sigma_vM#)")
      d.add_calculation('avg_epsilon',"np.average(#epsilon_V^0.0(F)_vM#)")
      print(d.fname)  
      stress_path = d.get_dataset_location('avg_sigma')
      stress = np.zeros(len(stress_path)) 
      strain = np.zeros(len(stress_path)) 
      hdf = h5py.File(d.fname)
      
      for count,path in enumerate(stress_path):
        stress[count] = np.array(hdf[path])
        strain[count] = np.array(hdf[path.split('avg_sigma')[0] + 'avg_epsilon'])
      
      stress = np.array(stress)/1E6
      strain = (1.0 + np.array(strain))*(1.0 + max_strain) - 1.0

      PyPlot.plot(strain,stress,linestyle = '-',linewidth='2.5',label='{} hit'.format(i))
      PyPlot.xlabel(r'$\varepsilon_{VM} $',fontsize=18)
      PyPlot.ylabel(r'$\sigma_{VM}$ (MPa)',fontsize=18)
      axes = PyPlot.gca()

      max_strain = np.max(strain)
    
    expt_data = pd.read_csv('/nethome/v.shah/{}/Shen_2016_fig2.csv'.format(sample_folder),names=['strain','stress'],skiprows=4,nrows=28)
    expt_data_1 = pd.read_csv('/nethome/v.shah/{}/Shen_2016_fig2_2nd_hit.csv'.format(sample_folder),names=['strain','stress'],skiprows=1)
    expt_data = expt_data.append(expt_data_1)
    PyPlot.plot(expt_data['strain'],expt_data['stress'],'o',label='Experimental data')
    PyPlot.legend()
    fig.savefig('/nethome/v.shah/{}/Multi_stand_stress_strain.png'.format(sample_folder),dpi=300)


  def data_for_stress_strain(self,stands,sample_folder,job_file):
    """
    Calculation for data of stress strain curves of multi-stand rolling.

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
    for i in range(1,stands+1):

      d = damask.Result('/nethome/v.shah/{}/{}_stand/{}'.format(sample_folder,i,job_file))
      d.add_Cauchy()
      d.add_strain_tensor()
      d.add_Mises('sigma')
      d.add_Mises('epsilon_V^0.0(F)')
      
      d.add_calculation('avg_sigma',"np.average(#sigma_vM#)")
      d.add_calculation('avg_epsilon',"np.average(#epsilon_V^0.0(F)_vM#)")

  def plot_fraction(self,sample_folder):
    """
    Plot RX fraction curves of multi-stand rolling.

    Parameters
    ----------
    sample_folder : str
      Name of the sample folder.

    """
    fig = PyPlot.figure()
    fraction_data = pd.read_csv('/nethome/v.shah/{}/resMDRX.fractions.txt'.format(sample_folder),\
                                 delimiter='\s+',names=['time','fraction'])
    PyPlot.plot(fraction_data['time'],fraction_data['fraction'],linestyle='-',linewidth='2.5',label='Simulation')
    PyPlot.xlabel('time (s)')
    PyPlot.ylabel('MDRX volume fraction (%)')
    PyPlot.legend()
    fig.savefig('/nethome/v.shah/{}/RXfractions.png'.format(sample_folder),dpi=300) 

  def add_nuclei_info(self,sample_folder,job_file,casipt,stand):
    """
    Add nucleation tag to the data for visualization of the nuclei.

    Parameters
    ----------
    sample_folder : str
      Name of the sample folder.
    job_file : str
      Name of the output file (2nd stand and so on).
    casipt : str
      Name of the casipt file with nucleation info.
    stand : int
      Number of the stand.
    
    """
    import damask
    d = damask.Result('/nethome/v.shah/{}/{}_stand/{}'.format(sample_folder,stand,job_file)) 
    rex_array = np.loadtxt(casipt,dtype=int,usecols=(1))
    total_cells = np.shape(d.read_dataset([d.get_dataset_location('grain_rotation')[-1]]))[0] 
    nuclei_array = np.zeros(total_cells)
   
    for i in rex_array:
      nuclei_array[i] = 1 

    path = d.groups_with_datasets('rho_mob')
    with h5py.File(d.fname,'a') as f:
      for i in range(len(path)):
        try:
          f[path[i] + '/Nucleation_tag'] != []
          data = f[path[i] + '/Nucleation_tag']
          data[...] = nuclei_array
        except KeyError:
          f[path[i] + '/Nucleation_tag'] = nuclei_array
          
          

class Grain_rotation_history():
  """ 
  Class containing functions to track evolution of grain rotation.

  Can trace back orientation change to very initial microstructure.
  """

  def __init__(self,current_file):
    """ Opens the HDF5 file in which rotation is to be calculated."""

    import damask
    self.current_file = damask.Result(current_file) 

  def get_regridded_coords(self,regridded_file):
    """
    Returns array of regridded coords.

    Parameters
    ----------
    regridded_file : str 
      txt file generated by 0_main code.

    """

    self.regridding_coords = np.loadtxt(regridded_file,skiprows=1,usecols=(0,1,2))

  def get_remeshed_coords(self,remesh_file):
    """
    Returns array of remeshed coords.
    Also shifts the remeshed CA coords to coincide with DAMASK cells. 

    Parameters
    ----------
    remesh_file : str 
      txt file containing results to be used for CA.
    """
   
    self.remeshed_coords = np.loadtxt(remesh_file,skiprows=1,usecols=((0,1,2)))
    remeshed_coords_1      = self.remeshed_coords
    remeshed_coords_1[:,0] = self.remeshed_coords[:,0] + self.remeshed_coords[1,0] - self.remeshed_coords[0,0]
    remeshed_coords_1[:,1] = self.remeshed_coords[:,1] + self.remeshed_coords[1,0] - self.remeshed_coords[0,0]
    remeshed_coords_1[:,2] = self.remeshed_coords[:,2] + self.remeshed_coords[1,0] - self.remeshed_coords[0,0]
    self.remeshed_coords_1 = remeshed_coords_1   

  def get_rotations(self,remesh_file):
    """
    Get the grain rotation values at different stages.

    Parameters:
    -----------
    remesh_file : str
      txt file containing results to be used for CA.
    """
   
    import damask
    self.first_grain_rotation = np.loadtxt(remesh_file,skiprows=1,usecols=(4))
    
    # get rotation at all recorded increments and store them in an array ( no of cols = no of increments)
    num_second_rotation = len(self.current_file.get_dataset_location('grain_rotation'))
    len_second_rotation = np.shape(self.current_file.read_dataset([self.current_file.get_dataset_location('grain_rotation')[-1]]))[0]
    self.second_grain_rotation = np.zeros((len_second_rotation,num_second_rotation))
    for i in range(num_second_rotation):
      self.second_grain_rotation[:,i] = self.current_file.read_dataset\
                                            ([self.current_file.get_dataset_location('grain_rotation')[i]])\
                                             .reshape(np.shape(self.second_grain_rotation[:,0]))


  def get_nucleation_info(self,casipt_file):
    """
    The function reads a casipt output file resMDRX.MDRX. 
    
    Parameters
    ----------
    casipt_file : str
      Name of output file containing nucleation info.
    """
    
    nucleation_info = np.loadtxt(casipt_file, usecols=(1,3,5,7,9))
    if nucleation_info.shape[0] != 0:
      nucleation_info[:,2:5] = nucleation_info[:,2:5]*math.pi/180.0

    return nucleation_info

  def modify_rotation(self,casipt_file):
    """  
    Uses the nucleation info and different rotations to calculate the cummulative rotations.
    
    Parameters
    ----------
    deformed_file : str
      H5py file generated from 0_main code (it has the orientations on the deformed grid).
    casipt_file : str
      Name of output file containing nucleation info.
    """
    
    import damask
    print(self.get_nucleation_info(casipt_file).shape)
    if self.get_nucleation_info(casipt_file).shape[0] != 0:
      self.first_grain_rotation[self.get_nucleation_info(casipt_file).astype(np.int32)[:,0]] = 0.0
    
    for i in range(np.shape(self.second_grain_rotation)[1]):
      self.total_grain_rotation = self.first_grain_rotation.reshape(np.shape(self.second_grain_rotation)[0]) \
                              + self.second_grain_rotation[:,i]
      with h5py.File(self.current_file.fname,'a') as f:
        data = f[self.current_file.get_dataset_location('grain_rotation')[i]]
        print(data[641],data[640])
        data[...] = self.total_grain_rotation.reshape(np.shape(data))


      

    self.second_grain_rotation = self.current_file.read_dataset([self.current_file.get_dataset_location('grain_rotation')[-1]])

