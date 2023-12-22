# MARSS
Multiband Artifact Regression in Simultaneous Slices (MARSS)

This is a MATLAB pipeline developed for use in simultaneous multi-slice (multiband; MB) fMRI data. MARSS is a regression-based method that mitigates an artifactual shared signal between simultaneously acquired slices in unprocessed MB fMRI. 
------------------------------------------------------------
Software Authors: Philip Tubiolo, John C. Williams, Mahika Gupta, and Jared X. Van Snellenberg


Software Requirements
--------------
This software uses SPM12, which is included in this distribution. For more information, visit https://www.fil.ion.ucl.ac.uk/spm/software/spm12/ . <br>
This software was developed on MATLAB R2023b and has been tested for compatibility on MATLAB R2021a. <br>
This software has been tested on the following operating systems, but should be compatible with MacOS as well: <br>
Linux: Red Hat Enterprise Linux 7.9 <br>
Windows: Windows 10 Home 64-bit <br>

Hardware Requirements
-----------------------
MARSS should only require the minimum RAM to handle a single fMRI timeseries (approximately 2GB). However, it has been tested with these minimum specifications: <br>
RAM: 16 GB <br>
Processor: Intel(R) Core(TM) i7-10750H CPU @ 2.60GHz <br>

With the above specifications, the total time taken for MARSS to complete on a single fMRI timeseries of 563 volumes is approximately 10 minutes. 

# Usage
MARSS.m
-------
MARSS.m is the main function of this pipeline, and the only function that must be directly run in order to perform MARSS on a single timeseries.

Syntax
--------
MARSS(timeseriesFile, MB, workingDir) performs MARSS artifact correction on a single unprocessed, MB fMRI timeseries. <br>

Input Arguments
--------------
timeseriesFile (string): Full path to unprocessed, MB fMRI timeseries<br>
MB (double): Multiband Acceleration Factor used during image acquisition<br>
workingDir (string): Parent directory for all MARSS outputs. MARSS will create a separate folder within this folder named after timeseriesFile.

Outputs in workingDir
--------------------
**MARSS_SliceCorrelations.mat**: this .mat file contains a structure array with slice correlation information in pre- and post-MARSS data. This includes the slice correlation matrices, average correlation between simultaneously acquired slices, and average correlation between non-simultaneously acquired slices. <br>
**za_.nii**: this NIFTI is the MARSS corrected timeseries. <br>
**_slcart.nii**: this NIFTI is the timeseries of MARSS-estimated artifact signal that was subtracted from timeseriesFile to produce za*.nii <br>
**_AVGslcart.nii**: this NIFTI is the average across timepoints of slcart.nii (shown as a single 3D volume). <br>
**MARSS_.png**: this is a summary diagnostic figure depicting pre- and post-MARSS slice correlation matrices, as well as orthogonal views of the artifact spatial distribution (from _AVGslcart.nii). <br>

Citation
---------
When using MARSS, please cite the following: Insert paper here <br>

License
----------
This software is released under the GNU General Public License Version 3.
