# Simulation_runner
Some scripts to control the simulations using python for coupled models 

## Running regrid_remesh.py:

Login to GITLAB repository of DAMASK: https://magit1.mpie.de/
```console
$ git checkout re-griding_2
$ make processing

```
This will enable the access to the python classes for re-gridding related work. 

Then go to the folder where the code 'regrid_remesh.py' is located and in ipython terminal:
``` ipython
$ import Remesh_for_CA
$ a = Remesh_for_CA(geom,load,inc,folder)
$ new_size, new_grid = a.main_all(a.geom,a.load,a.inc,a.folder)  #this will create the regridded output and also returns new size and new grid
$ nx,ny,nz = a.remesh_coords(filename,1.0,a.folder)    #filename is the name of the file created by a.main_all function
```
