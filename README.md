# MARSS
Multiband Artifact Regression in Simultaneous Slices (MARSS)

This is a MATLAB pipeline developed for use in simultaneous multi-slice (multiband; MB) fMRI data. MARSS is a regression-based method that mitigates an artifactual shared signal between simultaneously acquired slices in unprocessed MB fMRI. For more information, please read (insert paper here)
------------------------------------------------------------

Note to Users
--------------
This software requires SPM12 to be on the user path. SPM12 is included as part of this package. It can be obtained externally at https://www.fil.ion.ucl.ac.uk/spm/software/spm12/ .


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


This software is released under the GNU General Public License Version 3.
