#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

# example code to allow the suspending DAMASK simulation and 
# doing manipulation to files using python

import subprocess, signal, time, os
import damask
import h5py

cmd = "DAMASK_spectral -l tensionX.load -g 20grains16x16x16.geom > check.txt"

P = subprocess.Popen(cmd,shell=True)

while P.poll() != 0:
   time.sleep(5)
   os.kill(P.pid, signal.SIGSTOP)
   print("doing something")
   d = damask.Result('20grains16x16x16_tensionX.hdf5')
   path = d.get_dataset_location('F')[-1]
   f = h5py.File('20grains16x16x16_tensionX.hdf5')
   print('values',f[path][0,:,:])
   os.kill(P.pid, signal.SIGCONT)
