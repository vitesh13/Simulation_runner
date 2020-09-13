#!/usr/bin/env python3
# -*- coding: UTF-8 no BOM -*-

# example code to allow the suspending DAMASK simulation and 
# doing manipulation to files using python

import subprocess, signal, time, os

cmd = "DAMASK_spectral -l tensionX.load -g 20grains16x16x16.geom"

P = subprocess.Popen(cmd,shell=True)

while P.poll() != 0:
   time.sleep(5)
   os.kill(P.pid, signal.SIGSTOP)
   print("doing something")
   os.kill(P.pid, signal.SIGCONT)
