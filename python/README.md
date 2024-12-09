# Multiband Artifact Regression in Simultaneous Slices (MARSS)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)<br>
This is a Python version of a MATLAB pipeline developed for use in simultaneous multi-slice (multiband; MB) fMRI data.<br>
MARSS is a regression-based method that mitigates an artifactual shared signal between simultaneously acquired slices in unprocessed MB fMRI. <br>
Software Authors: Philip N. Tubiolo, John C. Williams, Ashley Zhao, and Jared X. Van Snellenberg<br>

Accompanies the following manuscript:<br>
Tubiolo PN, Williams JC, Van Snellenberg JX.<br> Characterization and Mitigation of a Simultaneous Multi-Slice fMRI Artifact: Multiband Artifact Regression in Simultaneous Slices.<br> Hum Brain Mapp. 2024 Nov;45(16):e70066. doi: 10.1002/hbm.70066. PMID: 39501896; PMCID: PMC11538719.<br>

Software Requirements
--------------
FSL must be installed and included in the system's PATH prior to use. For more information, visit https://fsl.fmrib.ox.ac.uk/fsl/docs/#/ <br>
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
Installation
-------
To install this [package](https://pypi.org/project/MARSS/), run the following command:

```
pip install MARSS
```

MARSS_main
-------
MARSS_main is the main function of this pipeline, and the only function that must be directly run in order to perform MARSS on a single timeseries.

Syntax
--------
MARSS_main(timeseriesFile, MB, workingDir,*args) performs MARSS artifact correction on a single unprocessed, MB fMRI timeseries. <br>

Input Arguments
--------------
timeseriesFile (string): Full path to unprocessed, MB fMRI timeseries<br>
MB (double): Multiband Acceleration Factor used during image acquisition<br>
workingDir (string): Parent directory for all MARSS outputs. MARSS will create a separate folder within this folder named after timeseriesFile.<br>
*args (string): Optional argument specifying a path to motion parameters. 

Outputs in workingDir
--------------------
**za_.nii**: this NIFTI is the MARSS corrected timeseries. <br>
**_slcart.nii**: this NIFTI is the timeseries of MARSS-estimated artifact signal that was subtracted from timeseriesFile to produce za*.nii <br>
**_AVGslcart.nii**: this NIFTI is the mean absolute value across timepoints of slcart.nii (shown as a single 3D volume). <br>
**corrMatrix\*.png**: slice correlation matrix of pre-MARSS data, along with the average difference in pearson correlation between simultaneously acquired slices and adjacent-to-simultaneous slices. <br>
**corrMatrixza\*.png:** slice correlation matrix of MARSS-corrected data. <br>

Citation
---------
When using MARSS, please cite the following:<br>
Tubiolo PN, Williams JC, Van Snellenberg JX.<br> Characterization and Mitigation of a Simultaneous Multi-Slice fMRI Artifact: Multiband Artifact Regression in Simultaneous Slices.<br> Hum Brain Mapp. 2024 Nov;45(16):e70066. doi: 10.1002/hbm.70066. PMID: 39501896; PMCID: PMC11538719.<br>

License
----------
This software is released under the GNU General Public License Version 3.
