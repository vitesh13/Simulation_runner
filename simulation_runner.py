#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import os,re
import subprocess,signal,time
import psutil
import shlex
from shutil import copyfile
from shutil import copy
from shutil import move
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
from scipy import constants
from damask import ConfigMaterial as cm

class Multi_stand_runner():
  """
  Class containing functions to carry out multi-stand simulations.

  """
  #simulation file names etc
  def __init__(self):
    """ Sets the files names and folder names."""
    self.sample_folder = '/nethome/v.shah/DAMASK/examples/grid/' #need to change this folder when doing other stands
    self.simulation_folder = '/nethome/v.shah/DAMASK/examples/grid/simulation'
    self.geom_file = '20grains16x16x16.vtr'
    self.load_file = 'tensionX.yaml'
    self.config_file = 'material.yaml'
    self.extra_config = 'ho_cr_ph.config'
    self.job_file = '{}_{}.hdf5'.format(self.geom_file.split('.')[0],self.load_file.split('.')[0])
    self.sta_file = '{}_{}.sta'.format(self.geom_file.split('.')[0],self.load_file.split('.')[0])
    self.restart_file = '{}_{}_restart.hdf5'.format(self.geom_file.split('.')[0],self.load_file.split('.')[0])
    self.tmp = 'tmp_storage'
    self.casipt_input = '/nethome/v.shah/casipt/casipt/input/test_drx.xml'

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

  def run_and_monitor_simulation(self,simulation_folder,sample_folder,geom_file,load_file,config_file,proc,freq):
    """
    Runs and monitors a fresh simulation.
    Will return negative value if terminated bz signals.

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
    freq: int
      Required output frequency
    """
    self.time_for_CA = 0.0
    os.chdir('{}'.format(sample_folder))
    copy(geom_file, '{}'.format(simulation_folder))
    copy(load_file,'{}'.format(simulation_folder))  
    copy(config_file,'{}'.format(simulation_folder))  
    os.chdir('{}'.format(simulation_folder))
    cmd = 'mpiexec -n {} DAMASK_grid -l {} -g {}'.format(proc,load_file,geom_file)
    mu  = self.get_mu()
    with open('check.txt','w') as f:
      P = subprocess.Popen(shlex.split(cmd),stdout = subprocess.PIPE, stderr = subprocess.PIPE)
      r = re.compile(' Increment [0-9]+/[0-9]+-1/1 @ Iteration 1≤0',re.U) 
      r2 = re.compile(' ... wrote initial configuration',re.U)
      record = []
      growth_length = 0.0
      while P.poll() is None:
        record = P.stdout.readline().decode('utf-8') 
        if re.search(r2,record):
          for children in psutil.Process(P.pid).children(recursive=True):
            #print(children)
            if children.name() == 'DAMASK_grid':
              children.suspend()
          #P.send_signal(signal.SIGSTOP)
          copy(self.job_file,'{}'.format(self.tmp))  #copying initial file to use it as replacement
          for children in psutil.Process(P.pid).children(recursive=True):
            #print(children)
            if children.name() == 'DAMASK_grid':
              children.resume()
          #P.send_signal(signal.SIGCONT)

        if re.search(r, record):
          for children in psutil.Process(P.pid).children(recursive=True):
            #print(children)
            if children.name() == 'DAMASK_grid':
              children.suspend()
          #P.send_signal(signal.SIGSTOP)
          print(record)
          velocity = self.calc_velocity(self.calc_delta_E(record,mu,2.5E-10),self.casipt_input)  #needs G, b and mobility  
          growth_length = growth_length + velocity*self.calc_timeStep(record) 
          self.time_for_CA = self.time_for_CA + self.calc_timeStep(record)
          print(growth_length)
          self.file_transfer(record,freq)
          if growth_length >= self.get_min_resolution():
            print(record)
            self.file_transfer(record,freq,trigger=True)
            #P.send_signal(signal.SIGUSR1)  #keeping this signal off for now
            P.send_signal(signal.SIGUSR2)
            # https://www.open-mpi.org/doc/v3.0/man1/mpiexec.1.php
            for children in psutil.Process(P.pid).children(recursive=True):
              print(children)
              if children.name() == 'DAMASK_grid':
                children.terminate()
            for children in psutil.Process(P.pid).children(recursive=True):
              #print(children)
              if children.name() == 'DAMASK_grid':
                children.resume()
            #P.send_signal(signal.SIGCONT)
          else:
            print("continuing")
            for children in psutil.Process(P.pid).children(recursive=True):
              #print(children)
              if children.name() == 'DAMASK_grid':
                children.resume()
            #P.send_signal(signal.SIGCONT)
            print("continued")
      return P.poll()

  def run_and_monitor_simulation_no_MPI(self,simulation_folder,sample_folder,geom_file,load_file,config_file,proc,freq):
    """
    Runs and monitors a fresh simulation.
    Will return negative value if terminated bz signals.

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
    freq: int
      Required output frequency
    """
    self.time_for_CA = 0.0
    os.chdir('{}'.format(sample_folder))
    copy(geom_file, '{}'.format(simulation_folder))
    copy(load_file,'{}'.format(simulation_folder))  
    copy(config_file,'{}'.format(simulation_folder))  
    os.chdir('{}'.format(simulation_folder))
    cmd = 'DAMASK_grid -l {} -g {}'.format(load_file,geom_file)
    mu  = self.get_mu()
    with open('check.txt','w') as f:
      P = subprocess.Popen(shlex.split(cmd),stdout = subprocess.PIPE, stderr = subprocess.PIPE)
      r = re.compile(' Increment [0-9]+/[0-9]+-1/1 @ Iteration 1≤0',re.U) 
      r2 = re.compile(' ... wrote initial configuration',re.U)
      record = []
      growth_length = 0.0
      while P.poll() is None:
        record = P.stdout.readline().decode('utf-8') 
        if re.search(r2,record):
          P.send_signal(signal.SIGSTOP)
          copy(self.job_file,'{}'.format(self.tmp))  #copying initial file to use it as replacement
          P.send_signal(signal.SIGCONT)

        if re.search(r, record):
          P.send_signal(signal.SIGSTOP)
          print(record)
          velocity = self.calc_velocity(self.calc_delta_E(record,mu,2.5E-10),self.casipt_input)  #needs G, b and mobility  
          growth_length = growth_length + velocity*self.calc_timeStep(record) 
          self.time_for_CA = self.time_for_CA + self.calc_timeStep(record)
          print(growth_length)
          self.file_transfer(record,freq)
          if growth_length >= self.get_min_resolution():
            print(record)
            self.file_transfer(record,freq,trigger=True)
            #P.send_signal(signal.SIGUSR1)  #keeping this signal off for now
            P.send_signal(signal.SIGUSR2)
            # https://www.open-mpi.org/doc/v3.0/man1/mpiexec.1.php
            for children in psutil.Process(P.pid).children(recursive=True):
              print(children)
              if children.name() == 'DAMASK_grid':
                children.terminate()
            P.send_signal(signal.SIGCONT)
          else:
            print("continuing")
            P.send_signal(signal.SIGCONT)
            print("continued")
      return P.poll()

  def file_transfer(self,inc_string,freq,inc='inc1',trigger=False):
    """
    Moves around the result file to avoid too many increments. 

    Parameters
    ----------
    inc_string: str
      String of type 'increment [0-9]+ converged.
    freq: int
      Required output frequency
    inc : str
      increment at which DRX restart happened.
    trigger : bool
      Indication to remove extra increments before trigger. Default false.
    """
    converged_inc = int(re.search('[0-9]+',inc_string).group())
    if (converged_inc)%freq == 0:
      copy(self.tmp + '/' + self.job_file,'{}'.format(self.simulation_folder))
    if (converged_inc - 1)%freq == 0:
      copy(self.job_file,'{}'.format(self.tmp))
    if converged_inc == int(inc.split('inc')[1]) + 2:
      copy(self.job_file,'{}'.format(self.tmp))
    if trigger:
      copy(self.tmp + '/' + self.job_file,'{}'.format(self.simulation_folder))


  def get_mu(self):
    """
    Get shear modulus as a function of temperature.

    """
    from damask import Config
    loading = Config.load(self.load_file)
    T = loading['initial_conditions']['thermal']['T']  
    return 92.648E9*(1 - 7.9921E-07*(T**2.0) + 3.3171E-10*(T**3.0))

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
    converged_inc = re.search('[0-9]+',inc_string).group() 
    recorded_inc  = int(converged_inc) - 1
    print(d.increments)
    d.view('increments',f'inc{recorded_inc}')
    path = d.get_dataset_location('rho_mob')[0]
    rho_mob = d.read_dataset([path])
    rho_dip = d.read_dataset([path.split('rho_mob')[0] + 'rho_dip'])
    tot_rho_array = np.sum((np.sum(rho_mob,1),np.sum(rho_dip,1)),0)
    max_rho = np.max(tot_rho_array)
    #avg_rho = np.average(tot_rho_array)
    min_rho = np.min(tot_rho_array)
    #diff_rho = max_rho - avg_rho
    diff_rho = max_rho - min_rho
    austenite_mv = 0.0000073713716  # austenite molar volume
     
    print(diff_rho)
    delta_E  = G*(b**2.0)*diff_rho*austenite_mv
    return delta_E


  def calc_velocity(self,delta_E,casipt_input):
    """
    Calculates velocity of the interface.

    Parameters
    ----------
    delta_E : float
      Difference in stored energy across an interface
    M : float
      Mobility factor.
    """
    from damask import Config
    loading = Config.load(self.load_file)
    T = loading['initial_conditions']['thermal']['T']
    tree = ET.parse(casipt_input)
    root = tree.getroot()
    M_0 = float(root.find('v0_gg').text)
    Q   = float(root.find('Qg_gg').text)
     
    return M_0*np.exp(-Q/(constants.R*T))*delta_E

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
    converged_inc = int(re.search('[0-9]+',inc_string).group())
    try: 
      relevant_step = np.where(summed_incs > converged_inc)[0][0] 
      time_step = loading['loadstep'][relevant_step]['discretization']['t']/\
                loading['loadstep'][relevant_step]['discretization']['N']
    except IndexError:
      time_step = 0.0
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
    if not os.path.exists('{}{}_stand/CA_files/{}'.format(sample_folder,stand,time)):
      os.makedirs('{}{}_stand/CA_files/{}'.format(sample_folder,stand,time))
    #os.mkdir('{}{}_stand/CA_files/{}'.format(sample_folder,stand,time))
    storage = '{}{}_stand/CA_files/{}'.format(sample_folder,stand,time)
    move('..ang','{}'.format(storage))
    move('._rho.txt','{}'.format(storage))
    move('.3D.geom','{}'.format(storage))
    #move(X.final.map.xy.dat','/nethome/v.shah/{}'.format(storage))
    move('.fractions.txt','{}'.format(storage))
    move('.growth_lengths.txt','{}'.format(storage))
    move('.MDRX.txt','{}'.format(storage))
    move('.texture_MDRX.txt','{}'.format(storage))
    move('.final.casipt','{}'.format(storage))
    return '{}{}_stand/CA_files/{}'.format(sample_folder,stand,time)

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
  def run_restart_simulation(self,simulation_folder,geom_file,load_file,config_file,restart_inc,proc):
    """
    Runs restart simulation after CA.

    Parameters
    ----------
    restart_inc : str
      Number at which restart will start.
    proc : int
      Number of processors to be used.
    """
    os.chdir(simulation_folder)
    copy(os.path.splitext(geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc), geom_file)
    copy(os.path.splitext(self.restart_file)[0] + '_regridded_{}_CA.hdf5'.format(restart_inc),self.restart_file)
    cmd = 'mpiexec -n {} DAMASK_grid -l {} -g {} -r {} > check.txt'.format(proc,load_file,geom_file,restart_inc.split('inc')[1])
    p = subprocess.Popen(cmd,shell=True)
    while p.poll() == None:
      p.poll()
    return p.poll()

  def run_restart_regridded(self,simulation_folder,geom_file,load_file,config_file,restart_inc,proc):
    """
    Runs restart simulation after a simple regridding.

    Parameters
    ----------
    restart_inc : str
      Number at which restart will start.
    proc : int
      Number of processors
    """
    os.chdir(simulation_folder)
    copy(os.path.splitext(geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc), geom_file)
    copy(os.path.splitext(self.job_file)[0] + '_restart_regridded_{}.hdf5'.format(restart_inc),self.restart_file)
    cmd = 'mpiexec -n {} DAMASK_grid -l {} -g {} -r {} > check.txt'.format(proc,load_file,geom_file,restart_inc.split('inc')[1])
    p = subprocess.Popen(cmd,shell=True)
    while p.poll() == None:
      p.poll()
    return p.poll()

  def run_restart_DRX(self,inc,proc,freq):
    """
    Restart simulation after initial DRX trigger. 

    Parameters
    ----------
    inc : str
      Increment at which restart is done.
    proc :int
      Number of processors.
    freq: int
      Required output frequency
    """
    os.chdir(self.simulation_folder)
    restart_geom = os.path.splitext(self.geom_file)[0] + '_regridded_{}.vtr'.format(inc)
    restart_hdf  = os.path.splitext(self.restart_file)[0] + '_regridded_{}_CA.hdf5'.format(inc) 
    copy(restart_geom,self.geom_file)
    copy(restart_hdf, self.restart_file)
    copy(self.job_file,'{}'.format(self.tmp))                          # recording the increment at which the simulation triggered
    self.time_for_CA = 0.0
    cmd = 'mpiexec -n {} DAMASK_grid -l {} -g {} -r {}'.format(proc,self.load_file,self.geom_file,inc.split('inc')[1])
    with open('check.txt','w') as f:
      P = subprocess.Popen(shlex.split(cmd),stdout = subprocess.PIPE, stderr = subprocess.PIPE)
      r = re.compile(' Increment [0-9]+/[0-9]+-1/1 @ Iteration 1≤0',re.U) 
      growth_length = 0.0
      while P.poll() is None:
        record = P.stdout.readline().decode('utf-8')

        if re.search(r, record):
          for children in psutil.Process(P.pid).children(recursive=True):
            #print(children)
            if children.name() == 'DAMASK_grid':
              children.suspend()
          #P.send_signal(signal.SIGSTOP)
          print(record)
          velocity = self.calc_velocity(self.calc_delta_E(record,32E9,2.5E-10),self.casipt_input)  #needs G, b and mobility  
          print('velocity after trigger',velocity)
          growth_length = growth_length + velocity*self.calc_timeStep(record) 
          self.time_for_CA = self.time_for_CA + self.calc_timeStep(record)
          print(growth_length)
          self.file_transfer(record,freq,inc)
          if growth_length >= self.get_min_resolution():
          #  print(record[-1])
            self.file_transfer(record,freq,trigger=True)
            #P.send_signal(signal.SIGUSR1)
            P.send_signal(signal.SIGUSR2)
            # https://www.open-mpi.org/doc/v3.0/man1/mpiexec.1.php
            for children in psutil.Process(P.pid).children(recursive=True):
              print(children)
              if children.name() == 'DAMASK_grid':
                children.terminate()
            #gone, alive = psutil.wait_procs(psutil.Process(P.pid).children(recursive=True), timeout=10)
            #print('alive',alive)
            for children in psutil.Process(P.pid).children(recursive=True):
              #print(children)
              if children.name() == 'DAMASK_grid':
                children.resume()
            #P.send_signal(signal.SIGCONT)
        else:
          for children in psutil.Process(P.pid).children(recursive=True):
            #print(children)
            if children.name() == 'DAMASK_grid':
              children.resume()
          #P.send_signal(signal.SIGCONT)
      return P.poll()
    
  def run_restart_DRX_no_MPI(self,inc,proc,freq):
    """
    Restart simulation after initial DRX trigger. 

    Parameters
    ----------
    inc : str
      Increment at which restart is done.
    proc :int
      Number of processors.
    freq: int
      Required output frequency
    """
    os.chdir(self.simulation_folder)
    restart_geom = os.path.splitext(self.geom_file)[0] + '_regridded_{}.vtr'.format(inc)
    restart_hdf  = os.path.splitext(self.restart_file)[0] + '_regridded_{}_CA.hdf5'.format(inc) 
    copy(restart_geom,self.geom_file)
    copy(restart_hdf, self.restart_file)
    copy(self.job_file,'{}'.format(self.tmp))                          # recording the increment at which the simulation triggered
    self.time_for_CA = 0.0
    cmd = 'DAMASK_grid -l {} -g {} -r {}'.format(self.load_file,self.geom_file,inc.split('inc')[1])
    with open('check.txt','w') as f:
      P = subprocess.Popen(shlex.split(cmd),stdout = subprocess.PIPE, stderr = subprocess.PIPE)
      r = re.compile(' Increment [0-9]+/[0-9]+-1/1 @ Iteration 1≤0',re.U) 
      growth_length = 0.0
      while P.poll() is None:
        record = P.stdout.readline().decode('utf-8')

        if re.search(r, record):
          P.send_signal(signal.SIGSTOP)
          print(record)
          velocity = self.calc_velocity(self.calc_delta_E(record,32E9,2.5E-10),self.casipt_input)  #needs G, b and mobility  
          print('velocity after trigger',velocity)
          growth_length = growth_length + velocity*self.calc_timeStep(record) 
          self.time_for_CA = self.time_for_CA + self.calc_timeStep(record)
          print(growth_length)
          self.file_transfer(record,freq,inc)
          if growth_length >= self.get_min_resolution():
          #  print(record[-1])
            self.file_transfer(record,freq,trigger=True)
            #P.send_signal(signal.SIGUSR1)
            P.send_signal(signal.SIGUSR2)
            # https://www.open-mpi.org/doc/v3.0/man1/mpiexec.1.php
            P.send_signal(signal.SIGTERM)
            P.send_signal(signal.SIGCONT)
            break
          else:
            P.send_signal(signal.SIGCONT)
      print('returncode after signals',P.returncode)
      P.wait()
      return P.poll()
 
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

  def save_rawfiles(self,stand,simulation_folder,sample_folder,job_file,restart_file,geom_file,load_file,config_file,sta_file):
    """
    Keeps the raw unprocessed HDF5 files (in case processing goes wrong).
    Parameters
    ----------
    stand : int
      stand at which deformation happens.
    sample_folder : string
      folder where data gets stored.
    """
    if not os.path.exists('{}/{}_stand/raw_files/'.format(sample_folder,stand)):
      os.mkdir('{}{}_stand/raw_files/'.format(sample_folder,stand))
    os.chdir('{}/'.format(simulation_folder))
    storage = '{}/{}_stand/raw_files'.format(sample_folder,stand)
    copy(job_file, '{}/'.format(storage))
    copy(sta_file, '{}/'.format(storage))
    copy(restart_file, '{}/'.format(storage))
    copy(geom_file, '{}/'.format(storage))
    copy(load_file, '{}/'.format(storage))
    copy(config_file, '{}/'.format(storage))

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
    os.chdir(simulation_folder)
    d = damask.Result(job_file)
    orientation0 = d.get_initial_orientation()
    d.add_grainrotation(orientation0,degrees=True,with_axis=False,without_rigid_rotation=True)
    #d.add_Eulers('orientation')
    d.add_calculation('tot_density','np.sum((np.sum(#rho_mob#,1),np.sum(#rho_dip#,1)),0)')
    d.add_calculation('r_s',"40/np.sqrt(#tot_density#)")

# initial processing
  def Initial_processing_DRX(self,job_file,simulation_folder,casipt_folder):
    """
    Initial post processing required for DRX simulations.
    Needs the remeshed original orientation data to calculate reorientation.
    
    Parameters
    ----------
    job_file : str
      Name of the damask output file to be processed.
    simulation_folder : str
      Name of the simulation folder where the job file exists.
    casipt_folder: str
      Path of the ..ang file that contains the orientations in form of euler angles.

    """
    import damask 
    from damask import Orientation
    from damask import Rotation
    os.chdir(simulation_folder)
    d = damask.Result(job_file)
    #orientation0 = np.loadtxt(simulation_folder + '/postProc/remesh_Initial_orientation_{}.txt'.format(inc),usecols=(3,4,5,6))
    orientation0 = np.loadtxt(casipt_folder)
    orientation0 = Orientation(Rotation.from_Euler_angles(orientation0))
    d.add_grainrotation(orientation0,degrees=True,with_axis=False,without_rigid_rotation=True)
    #d.add_Eulers('orientation')
    d.add_calculation('tot_density','np.sum((np.sum(#rho_mob#,1),np.sum(#rho_dip#,1)),0)')
    d.add_calculation('r_s',"40/np.sqrt(#tot_density#)")

# modify the CA xml file
  def modify_CA_setting(self,filename,T,grid,delta_t,dx,inherit_growth,start_file,basefn):
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
    inherit_growth : int
      Tag to inherit growth or not. 0 is no, 1 is yes.
    start_file : str
      Path of the input text file for CA
    basefn : str
      Path of the folder where the output of CA is stored

    """
    
    tree = ET.parse(filename)
    root = tree.getroot()
    
    
    damask_mat = cm.load(self.simulation_folder + '/' + self.config_file) 
    phase_name  = [i for i in damask_mat['phase'].keys()][0]
    rho_mob_0   = float(damask_mat['phase'][phase_name]['mechanics']['plasticity']['rho_mob_0'][0])*12
    rho_dip_0   = float(damask_mat['phase'][phase_name]['mechanics']['plasticity']['rho_dip_0'][0])*12 #multiplying by slip systems
    root.find('T0').text = str(T)
    root.find('nx').text = str(grid[0])
    root.find('ny').text = str(grid[1])
    root.find('nz').text = str(grid[2])
    root.find('deltats').text = '{:.8f}'.format(delta_t)
    root.find('dx').text = '{:.8f}'.format(dx)
    root.find('startfile_fn').text = start_file
    root.find('basefn').text = basefn
    root.find('growthfile_fn').text = basefn + '.growth_lengths.txt'
    root.find('mvInitialDislocationDensity').text = str(rho_mob_0 + rho_dip_0)
    root.find('mvUseGrowthLengths').text = str(inherit_growth)
    tree.write(filename)
    
  def modify_geom_attributes(self):
    """
    Modifies the attributes associated with group geometry in hdf5 file. 
    This is needed when a simulation is restarted with newer geometry.

    """
    from damask import Grid
    os.chdir(self.simulation_folder)
    new_geom = Grid.load(self.geom_file)
    with h5py.File(self.job_file) as f:
      f['geometry'].attrs['cells'] = new_geom.cells.astype(np.int32)
      f['geometry'].attrs['size']  = new_geom.size
      
  def modify_mapping(self):
    """
    Modify the datasets of group mapping in hdf5 file. 
    This is needed when a simulation is restarted with newer geometry.

    """
    from damask import Grid
    os.chdir(self.simulation_folder)
    new_geom = Grid.load(self.geom_file)
    new_len = np.prod(new_geom.cells)

    d = damask.Result(self.job_file)
    phase = d.phases[0]
    homogenizations = d.homogenizations[0]

    f = h5py.File(self.job_file)
    comp_type_phase = np.dtype(f['mapping']['phase'].dtype) 
    comp_type_homog = np.dtype(f['mapping']['homogenization'].dtype) 

    del f['mapping']
    phase_array = np.array([(bytes('{}'.format(phase).encode()),1)]*new_len,dtype=comp_type_phase)
    phase_array['Position'] = np.arange(new_len)

    homog_array = np.array([(bytes('{}'.format(homogenizations).encode()),1)]*new_len,dtype=comp_type_homog)
    homog_array['Position'] = np.arange(new_len)

    f.create_dataset('mapping/phase',dtype=comp_type_phase,shape=(new_len,1),data=phase_array)
    f.create_dataset('mapping/homogenization',dtype=comp_type_homog,shape=(new_len,1),data=homog_array)
    

  def perform_CA(self,input_settings):
    """
    Starts a CA simulation.
 
    Parameters
    ----------
    input_settings : str
      Name of the input xml file for CA
    """
   
    os.chdir('/nethome/v.shah/casipt/casipt_dens/casipt')
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
    Get the indices where the MDRX nuclei have appeared and have grown in that CA ste. 
    
    Parameters
    ----------
    casipt_file : str
      Path of output file containing nucleation info.
      Generally, .MDRX.txt is the name.
    """
    
    nucleation_info = np.loadtxt(casipt_file, dtype = np.int, usecols=(1))

    return nucleation_info

  def read_CASIPT_orientation(self,ang_file):
    """
    Read the orientation from the ..ang file generated from CASIPT simulation.

    Parameters
    ----------
    ang_file : str
      Path of file that contains the orientations of each point fronm CASIPT simulation.
      Generally, ..ang.txt is the name.
    """
    orientation_casipt = np.loadtxt(ang_file)

    return orientation_casipt

  def modify_initial_orientation_file(self,initial_ori_file,casipt_file,ang_file,CA_input_file):
    """  
    Uses the nucleation info and casipt orientations to reset the orientations of the transformed points to
    the current casipt orientations. 
    The transformed points at that casipt stage should have no accumulated rotation. 
    From the transformation point on, the calculation of the rotation is started from this point of time. 
    Therefore, the initial orientation should be modified to current orientation for the transformed points.
    
    Parameters
    ----------
    initial_ori_file : str
      Path of the file which contains initial orientations.
      Generally named as remesh_geomname_loadname_inc.txt
    casipt_file : str
      Path of output file containing nucleation info.
      Generally, .MDRX.txt is the name.
    CA_input_file : str
      Path of the CA input file.
    ang_file : str
      Path of file that contains the orientations of each point fronm CASIPT simulation.
      Generally, ..ang.txt is the name.
    """
    from damask import Rotation 
    
    # changing the MDRXed points orientation
    init_ori = np.loadtxt(initial_ori_file)
    CA_input = np.loadtxt(CA_input_file,skiprows = 1)
    indices_RX = self.get_nucleation_info(casipt_file) 
    init_ori[indices_RX,3:7] = Rotation.from_Euler_angles(self.read_CASIPT_orientation(ang_file)[indices_RX,:]).as_quaternion()
    
    # changing the GGed points orientation
    indices_GG = np.unique(np.where(np.isclose(CA_input[:,7:10],self.read_CASIPT_orientation(ang_file)) == False)[0])
    for i in indices_GG:
      root_index = np.nonzero(np.isclose(CA_input[:,7:10],CA_output[indices_GG[i]])) # look for the parent of changed orientation
      init_ori[i,3:7] = init_ori[root_index,3:7]
    np.savetxt(initial_ori_file,init_ori)
    
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

