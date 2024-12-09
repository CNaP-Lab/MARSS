function [out] = MARSS_mbCorrPlot(fname,MPs,MB)
% This function generates pairwise correlations between average signal in
% all slices
% Inputs:
% fname = string path to a volume timeseries
% MPs = time x Parameter matrix of nuisance parameters (assumed to be motion) which
% will be partialled out of the pairwise correlation in average slice
% signals
% MB = MB acceleration factor
%
% Outputs:
% out = struct containing all slice correlation outputs (see below for
% explanation of salient outputs)
%-----------------------------------------------------------------------------
% Philip Tubiolo, John C. Williams, Mahika Gupta, & Jared Van Snellenberg 2023

% When used, please CITE:  
%-----------------------------------------------------------------------------

[~,f,x] = fileparts(fname);
if strcmp(x,'.txt')
    d = load(fname);
else
    V = spm_vol(fname);
    nY = spm_read_vols(V);
    nY = double(nY); %cast to double

    d = squeeze(mean(mean(nY,1),2))';
end
[dasimulSlicesZ,daAdjacentSlicesZ] = MARSS_avgMBcorr(d,MB);

if size(d,1) ~= size(MPs,1)
    if size(MPs,1) < size(d,1)
        d(1:(size(d,1)-size(MPs,1)),:) = [];
    end
end

% nuisance regressor design matrix [intercept, linearDetrend, MPs,squared
% MPs, derivatives of MPs, squared derivatives]
X = [ones(size(d,1),1) (1:size(d,1))' MPs MPs.^2 [zeros(1,6); diff(MPs)] [zeros(1,6); diff(MPs)].^2];
rd = zeros(size(d));
for i = 1:size(d,2)
    rd(:,i) = d(:,i) - (X*(X\d(:,i)));
end
[rdasimulSlicesZ,rdaAdjacentSlicesZ] = MARSS_avgMBcorr(rd,MB);



out.name = fname; % run name
out.data = d; % average slice timeseries (slices * time)
out.regdata = rd; % average slice timeseries after nuisance regression (slices * time)
out.dasimulSlicesZ = dasimulSlicesZ;
out.daAdjacentSlicesZ = daAdjacentSlicesZ;
out.rdasimulSlicesZ = rdasimulSlicesZ; % avg z-transformed correlation in simultaneously acquired slices (after motion regression)
out.rdaAdjacentSlicesZ = rdaAdjacentSlicesZ; % avg z-transformed correlation in adjacent slices (after motion regression)
out.corrMat_motionRegressed = corr(rd); % slice correlation matrix after motion regression (this is the pre-MARSS correlation matrix displayed in the summary figure)
end
