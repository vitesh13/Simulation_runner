#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

# example code to allow the suspending DAMASK simulation and 
# doing manipulation to files using python

import subprocess, signal, time, os
import damask
import h5py
import re

cmd = "DAMASK_grid -l tensionX.yaml -g 20grains16x16x16.vtr"
with open('check.txt','w') as f:
    P = subprocess.Popen(cmd,stdout = subprocess.PIPE, stderr = subprocess.PIPE,shell=True)
    r = re.compile(' increment 3 converged')
    record = []
    while P.poll() is None:
      for count,line in enumerate(iter(P.stdout.readline, b'')):
             record.append(line.decode('utf-8'))
             if re.search(r, record[-1]):
               os.kill(P.pid+1, signal.SIGSTOP)
               d = damask.Result('20grains16x16x16_tensionX.hdf5')
               print(d.get_dataset_location('F'))
               os.kill(P.pid+1, signal.SIGUSR1)
               os.kill(P.pid+1, signal.SIGCONT)
    for line in record:
      f.write(line)


