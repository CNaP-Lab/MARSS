# Multiband Artifact Regression in Simultaneous Slices (MARSS)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)<br>
This is a Python and MATLAB pipeline developed for use in simultaneous multi-slice (multiband; MB) fMRI data.<br>
MARSS is a regression-based method that mitigates an artifactual shared signal between simultaneously acquired slices in unprocessed MB fMRI. <br>
Software Authors: Philip N. Tubiolo, John C. Williams, Ashley Zhao, Mahika Gupta, and Jared X. Van Snellenberg<br>

Accompanies the following manuscript:<br>
Tubiolo PN, Williams JC, Van Snellenberg JX.<br> Characterization and Mitigation of a Simultaneous Multi-Slice fMRI Artifact: Multiband Artifact Regression in Simultaneous Slices.<br> Hum Brain Mapp. 2024 Nov;45(16):e70066. doi: 10.1002/hbm.70066. PMID: 39501896; PMCID: PMC11538719.<br>

Software Requirements
--------------
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
MATLAB
-------
Prior to running MARSS, it must be added to the MATLAB path via the following command: 
```
addpath(genpath(('/path/to/MARSS/'))
```
For more information, see the [MATLAB README](https://github.com/CNaP-Lab/MARSS/tree/main/MATLAB). <br>

Python
-------
To install this [package](https://pypi.org/project/MARSS/), run the following command:

```
pip install MARSS
```
For more information, see the [Python README](https://github.com/CNaP-Lab/MARSS/tree/main/python). <br>

Citation
---------
When using MARSS, please cite the following:<br>
Tubiolo PN, Williams JC, Van Snellenberg JX.<br> Characterization and Mitigation of a Simultaneous Multi-Slice fMRI Artifact: Multiband Artifact Regression in Simultaneous Slices.<br> Hum Brain Mapp. 2024 Nov;45(16):e70066. doi: 10.1002/hbm.70066. PMID: 39501896; PMCID: PMC11538719.<br>

License
----------
This software is released under the GNU General Public License Version 3.
