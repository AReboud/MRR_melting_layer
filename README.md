# MRR_melting_layer
Algorithm to retrieve the melting layer from MRR (Micro Rain Radar) netcdf file.

## How to use

The code is developed for python 3.8+ and should run on any recent Windows system (and most likely also Linux, but not tested).

The following python packages are required:
  * numpy
  * scipy
  * matplotlib (for plotting only)
  * netcdf4-python (for reading the MRR processed files only)

## Download Sample
The sample of the MRR netcdf file can be downloaded here (about 90 MB):
https://filesender.renater.fr/?s=download&token=685c3d8d-a518-4e41-bcec-dd1441884cc1 (June 2022)
It must be placed in the directory 'Data/MRR/' to be runned by the MRR_melting_layer.py script.

## Content
MRR_melting_layer.py used the function stored in func.py to compute and plot the melting layer elevation from the netcdf file sample.
The sample is a 1-day MRR-2 (Metek) record at Saint-Martin-d'Hères (France) on 2020-12-12. It has been processed with the IMProToo processing algorithm (https://github.com/maahn/IMProToo) at 1-min time integration.

MRR_melting_layer_debug_mode.py can be executed step-by-step. It contains several intermediate plots to better visualize and test the parameters sensitivity in the identification of the melting layer elevation.

## Questions
In case of any questions, please don't hesitate to contact Arnaud Reboud: arnaud [dot] reboud [at] univ [dash] grenoble [dash] alpes [dot] fr
