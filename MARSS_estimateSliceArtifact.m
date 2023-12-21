function [artifactEstimate, globalSignal] = MARSS_estimateSliceArtifact(Y,MB,X)
% This function estimates slice-wise artifact contribution for subsequent
% regression-based removal
%
% Inputs:
% Y = 4d timeseries data
% MB = multiband acceleration factor
% X = design matrix of nuisance parameters (MPs, MP^2, diff(MPs),
% diff(MPs)^2)
%
% Outputs:
% artifactEstimate = a matrix of estimated artifact times x slices
% globalSignal = a global signal estimate for each slice that *excludes*
% all data from all slices acquired simultaneously with the slice in
% question
% 
% Note that global image signal is calculated and removed from artifact
% estimate automatically.
%-----------------------------------------------------------------------------
% Philip Tubiolo, John C. Williams, Mahika Gupta, & Jared Van Snellenberg 2023

% When used, please CITE:  
%-----------------------------------------------------------------------------

if mod(size(Y,3),MB)
    error('# of slices and MB factor are incompatible.');
end

nSlices = size(Y,3);
nSliceSets = nSlices/MB;
data = squeeze(mean(mean(Y,1),2))';

X = zscore(X);

[sameSetData,artifactEstimate,globalSignal] = deal(zeros(size(data)));
for j = 1:nSlices
    sliceSet = mod(j,nSliceSets);
    if ~sliceSet
        sliceSet = nSliceSets;
    end
    
    sameSliceSet = false(1,nSlices);
    lowerSlices = sliceSet:nSliceSets:j-1;
    higherSlices = j+nSliceSets:nSliceSets:nSlices;
    sameSliceSet([lowerSlices higherSlices]) = true; %this excludes current slice but includes all other slices in the simultaneous set
    
    sameSetData(:,j) = mean(data(:,sameSliceSet),2); 
    
    sameSliceSet(j) = true; %put current slice back into current slice set to exclude from global signal calc
    globalSignal(:,j) = mean(data(:,~sameSliceSet),2);
    % design matrix = [intercept globalSignal MPs]    
    Xout = [ones(size(data,1),1) zscore(globalSignal(:,j)) X]; 
    B = Xout\sameSetData(:,j);
    
    artifactEstimate(:,j) = sameSetData(:,j) - Xout * B;
    
end


end
