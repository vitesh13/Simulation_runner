#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

import os,sys
sys.path.append('/nethome/v.shah/DAMASK_DRX/Simulation_runner/')
import subprocess
import shlex
import numpy as np
from shutil import copyfile
from shutil import copy
from simulation_runner import Multi_stand_runner
from simulation_runner import Grain_rotation_history
import After_CA_restart
from regrid_remesh import Remesh_for_CA
import damask
from damask import ConfigMaterial as cm
from damask import Config

#simulation file names etc
a = Multi_stand_runner()
a.simulation_folder = '/nethome/v.shah/DAMASK_DRX/testing/simulation' #need to change this folder when doing other stands
a.sample_folder = '/nethome/v.shah/DAMASK_DRX/testing/'
a.sta_file = '25grains_24x24x48_4microns_tensionX.sta' 
a.job_file = '25grains_24x24x48_4microns_tensionX.hdf5'
a.restart_file = '25grains_24x24x48_4microns_tensionX_restart.hdf5' 
a.geom_file = '25grains_24x24x48_4microns.vtr'
a.load_file = 'tensionX.yaml'
a.config_file = 'material.yaml'
proc          = 4 
freq          = 1000
casipt_input  = '/nethome/v.shah/casipt/casipt_dens/casipt/input/test_drx_sr10.xml'
growth_file_path = '/nethome/v.shah/casipt/dpca_sims/mdrx/T_1273_sr10'
T             = 1273.0
K             = 5.0 #0.00005
restart_inc   = []


# -------------------------------------------------------------------------------------------------------------------------------
# start simulation run till DRX trigger
signal = a.run_and_monitor_simulation(a.simulation_folder,a.sample_folder,a.geom_file,a.load_file,a.config_file,proc,freq,K)

d = damask.Result(a.job_file)
restart_inc.append(d.increments[-1])    #increment in form of 'inc200'
print(restart_inc[-1])
print(a.time_for_CA)
if a.enough_displacement():
  print('regridding needed')

  a.Initial_processing(a.job_file,a.simulation_folder)
  r = Remesh_for_CA(a.geom_file,a.load_file,int(restart_inc[-1].split('inc')[1]),os.getcwd())
  #r.main_all(r.geom,r.load,r.inc,r.folder)
  #nx,ny,nz = r.remesh_coords('postProc/{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),1.0,os.getcwd())
  nx,ny,nz,dx = r.output_without_regridding(r.geom,r.load,r.inc,r.folder)
  
  inherit_growth = 0
  a.modify_CA_setting(casipt_input,T,np.array([nx,ny,nz]),a.time_for_CA,np.min(dx),\
                      inherit_growth,a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],\
                      restart_inc[-1]),\
                      '/nethome/v.shah/casipt/dpca_sims/mdrx/{}/'.format('T_1273_sr10'))
  print(os.path.splitext(casipt_input))
  a.perform_CA(os.path.basename(casipt_input))
  path_CA_stored = a.copy_CA_output('/nethome/v.shah/casipt/dpca_sims/mdrx/{}'.format('T_1273_sr10'),a.sample_folder,r.inc,a.time_for_CA)
  
  new_geom = After_CA_restart.CASIPT_postprocessing(path_CA_stored).geomFromCA(np.min(dx))
  new_geom.save(path_CA_stored + '/' + os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]))
  After_CA_restart.CASIPT_postprocessing(path_CA_stored).config_from_CA(a.simulation_folder)
  After_CA_restart.CASIPT_postprocessing(path_CA_stored).Initialize_Fp_no_regridding(\
                   a.simulation_folder + '/' + a.restart_file,\
                   restart_inc[-1],\
                   a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                   path_CA_stored + '/..ang',\
                   path_CA_stored + '/._rho.txt')
  os.chdir(path_CA_stored)
  copy(os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]),a.simulation_folder)
  copy(a.config_file,a.simulation_folder)
else:
  a.Initial_processing(a.job_file,a.simulation_folder)
  r = Remesh_for_CA(a.geom_file,a.load_file,int(restart_inc[-1].split('inc')[1]),os.getcwd())
  r.main_all(r.geom,r.load,r.inc,r.folder)
  nx,ny,nz = r.remesh_coords('postProc/{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),1.0,os.getcwd())
  
  inherit_growth = 0
  a.modify_CA_setting(casipt_input,T,np.array([nx+1,ny+1,nz+1]),a.time_for_CA,np.min(r.new_size/r.new_grid),\
                      inherit_growth,a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],\
                      restart_inc[-1]),\
                      '/nethome/v.shah/casipt/dpca_sims/mdrx/{}/'.format('T_1273_sr10'))
  print(os.path.splitext(casipt_input))
  a.perform_CA(os.path.basename(casipt_input))
  path_CA_stored = a.copy_CA_output('/nethome/v.shah/casipt/dpca_sims/mdrx/{}'.format('T_1273_sr10'),a.sample_folder,r.inc,a.time_for_CA)
  
  new_geom = After_CA_restart.CASIPT_postprocessing(path_CA_stored).geomFromCA(np.min(r.new_size/r.new_grid))
  new_geom.save(path_CA_stored + '/' + os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]))
  After_CA_restart.CASIPT_postprocessing(path_CA_stored).config_from_CA(a.simulation_folder)
  After_CA_restart.CASIPT_postprocessing(path_CA_stored).findNeighbours(\
                   a.simulation_folder + '/postProc/{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                   a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                   a.simulation_folder + '/' + os.path.splitext(a.restart_file)[0] + '_regridded_{}.hdf5'.format(restart_inc[-1]))
  After_CA_restart.CASIPT_postprocessing(path_CA_stored).Initialize_Fp(\
                   a.simulation_folder + '/' + os.path.splitext(a.restart_file)[0] + '_regridded_{}_CA.hdf5'.format(restart_inc[-1]),\
                   path_CA_stored + '/.MDRX.txt',\
                   a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                   path_CA_stored + '/..ang',\
                   path_CA_stored + '/._rho.txt')
  os.chdir(path_CA_stored)
  copy(os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]),a.simulation_folder)
  copy(a.config_file,a.simulation_folder)
  
# -------------------------------------------------------------------------------------------------------------------------------
# modifying total incs in the simulation (for testing)
#os.chdir(a.simulation_folder)
#loading = Config.load(a.load_file)
#loading['loadstep'][0]['boundary_conditions']['mechanical']['dot_F'][8] = 1e-06
#loading['loadstep'][0]['discretization']['t'] = 0.0005
#loading['loadstep'][0]['discretization']['N'] = 10
#loading.save(a.load_file)


# finding total incs in simulation
os.chdir(a.simulation_folder)
loading = Config.load(a.load_file)
total_steps = len(loading['loadstep'])
steps_list = [loading['loadstep'][i]['discretization']['N'] for i in range(len(loading['loadstep']))]
summed_incs = np.array([sum(steps_list[:i]) for i in range(len(steps_list)+1)][1:])
total_incs = summed_incs[-1]
### -------------------------------------------------------------------------------------------------------------------------------
##
### after the initial DRX trigger
while int(restart_inc[-1].split('inc')[1]) < total_incs: 
  #signal = a.run_restart_DRX_no_MPI(restart_inc[-1],proc,freq)
  signal = a.run_restart_DRX(restart_inc[-1],proc,freq,K)
  print('signal',signal)
  if signal == 134:
    break
  d = damask.Result(a.job_file)
  restart_inc.append(d.increments[-1])    #increment in form of 'inc200'
  print(restart_inc[-1])
  print(a.time_for_CA)
  if a.enough_displacement():
    print('regridding needed')
    r = Remesh_for_CA(a.geom_file,a.load_file,int(restart_inc[-1].split('inc')[1]),os.getcwd())
    a.Initial_processing_DRX(a.job_file,a.simulation_folder,path_CA_stored + '/..ang')
    nx,ny,nz,dx = r.output_without_regridding(r.geom,r.load,r.inc,r.folder,needs_hist=True,\
                                              casipt_input=a.simulation_folder + '/postProc/remesh_{}_{}.txt'\
                                              .format(os.path.splitext(a.job_file)[0],restart_inc[-2]),\
                                              path_CA_stored = path_CA_stored)
  
    copy(path_CA_stored + '/.growth_lengths.txt',path_CA_stored + '/regridded_growth_lengths.txt')
    copy(path_CA_stored + '/.growth_lengths.txt',path_CA_stored + '/remeshed_growth_lengths.txt')
    inherit_growth = 1
    print(casipt_input)
    a.modify_CA_setting(casipt_input,T,np.array([nx,ny,nz]),a.time_for_CA,np.min(dx),\
                        inherit_growth, a.simulation_folder + '/postProc/remesh_{}_{}.txt'\
                        .format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                        '/nethome/v.shah/casipt/dpca_sims/mdrx/{}/'.format('T_1273_sr10'))
    copy(path_CA_stored + '/regridded_growth_lengths.txt',growth_file_path + '/.growth_lengths.txt')
    a.perform_CA(os.path.basename(casipt_input))
    path_CA_stored = a.copy_CA_output('/nethome/v.shah/casipt/dpca_sims/mdrx/{}'.format('T_1273_sr10'),a.sample_folder,r.inc,a.time_for_CA)
  
    new_geom = After_CA_restart.CASIPT_postprocessing(path_CA_stored).geomFromCA(np.min(dx))
    new_geom.save(path_CA_stored + '/' + os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]))
    After_CA_restart.CASIPT_postprocessing(path_CA_stored).config_from_CA(a.simulation_folder)
    After_CA_restart.CASIPT_postprocessing(path_CA_stored).findNeighbours(\
                     a.simulation_folder + '/postProc/{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                     a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                     a.simulation_folder + '/' + os.path.splitext(a.restart_file)[0] + '_regridded_{}.hdf5'.format(restart_inc[-1]))
    After_CA_restart.CASIPT_postprocessing(path_CA_stored).Initialize_Fp_no_regridding(\
                     a.simulation_folder + '/' + a.restart_file,\
                     restart_inc[-1],\
                     a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                     path_CA_stored + '/..ang',\
                     path_CA_stored + '/._rho.txt')
    os.chdir(path_CA_stored)
    copy(os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]),a.simulation_folder)
    copy(a.config_file,a.simulation_folder)
  else:
    a.modify_geom_attributes()
    a.modify_mapping()
    r = Remesh_for_CA(a.geom_file,a.load_file,int(restart_inc[-1].split('inc')[1]),os.getcwd())
    #r.regrid_Initial_ori_DRX(r.geom,r.load,restart_inc,r.folder)
    #r.remesh_Initial_ori0('postProc/Initial_orientation_regridded_inc{}.txt'.format(r.inc),1.0,os.getcwd())
    a.Initial_processing_DRX(a.job_file,a.simulation_folder,path_CA_stored + '/..ang')
    r.main_all(r.geom,r.load,r.inc,r.folder,needs_hist=True,\
               casipt_input=a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-2]),\
               path_CA_stored=path_CA_stored)
    nx,ny,nz = r.remesh_coords('postProc/{}_inc{}.txt'.format(os.path.splitext(a.job_file)[0],r.inc),1.0,os.getcwd())
    
    r.regrid_growth_lengths(a.geom_file,a.load_file,restart_inc,r.folder,path_CA_stored) 
    r.remesh_growth_lengths('postProc/{}_inc{}.txt'.format(os.path.splitext(a.job_file)[0],r.inc),1.0, \
                            a.simulation_folder,path_CA_stored)
    inherit_growth = 1
    print(casipt_input)
    a.modify_CA_setting(casipt_input,T,np.array([nx+1,ny+1,nz+1]),a.time_for_CA,np.min(r.new_size/r.new_grid),\
                        inherit_growth, a.simulation_folder + '/postProc/remesh_{}_{}.txt'\
                        .format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                        '/nethome/v.shah/casipt/dpca_sims/mdrx/{}/'.format('T_1273_sr10'))
    copy(path_CA_stored + '/regridded_growth_lengths.txt',growth_file_path + '/.growth_lengths.txt')
    a.perform_CA(os.path.basename(casipt_input))
    path_CA_stored = a.copy_CA_output('/nethome/v.shah/casipt/dpca_sims/mdrx/{}'.format('T_1273_sr10'),a.sample_folder,r.inc,a.time_for_CA)
    
    new_geom = After_CA_restart.CASIPT_postprocessing(path_CA_stored).geomFromCA(np.min(r.new_size/r.new_grid))
    new_geom.save(path_CA_stored + '/' + os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]))
    After_CA_restart.CASIPT_postprocessing(path_CA_stored).config_from_CA(a.simulation_folder)
    After_CA_restart.CASIPT_postprocessing(path_CA_stored).findNeighbours(\
                     a.simulation_folder + '/postProc/{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                     a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                     a.simulation_folder + '/' + os.path.splitext(a.restart_file)[0] + '_regridded_{}.hdf5'.format(restart_inc[-1]))
    After_CA_restart.CASIPT_postprocessing(path_CA_stored).Initialize_Fp(\
                     a.simulation_folder + '/' + os.path.splitext(a.restart_file)[0] + '_regridded_{}_CA.hdf5'.format(restart_inc[-1]),\
                     path_CA_stored + '/.MDRX.txt',\
                     a.simulation_folder + '/postProc/remesh_{}_{}.txt'.format(os.path.splitext(a.job_file)[0],restart_inc[-1]),\
                     path_CA_stored + '/..ang',\
                     path_CA_stored + '/._rho.txt')
    os.chdir(path_CA_stored)
    copy(os.path.splitext(a.geom_file)[0] + '_regridded_{}.vtr'.format(restart_inc[-1]),a.simulation_folder)
    copy(a.config_file,a.simulation_folder)
    
