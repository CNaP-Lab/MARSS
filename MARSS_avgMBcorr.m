function [simulSlicesZ,allnonSimulSlicesZ] = MARSS_avgMBcorr(dat,MB)
% This function calculates the average signal correlation between
% simultaneously acquired slices and between non-simultaneously acquired slices
% [avgR,avgR_Z,sliceAvgs] = avgMBcorr(dat,MB)
%
% Inputs:
% dat = matrix of average timeseries values over slices. Expected to be
% volumes x slices, but can be slices x volumes (a warning is generated) as
% long as there are fewer slices than volumes.
% 
% MB = multiband accelleration factor
% 
%
% Outputs:
% simulSlicesZ = avg z-transformed correlation in simultaneously acquired slices
%
% allnonSimulSlicesZ = avg z-transformed correlation in ALL non-simultaneously acquired slices 
%-----------------------------------------------------------------------------
% Philip Tubiolo, John C. Williams, Mahika Gupta, & Jared Van Snellenberg 2023

% When used, please CITE:  
%-----------------------------------------------------------------------------

if size(dat,1) < size(dat,2)
    warning(['The dat matrix appears to have been input as slices x volumes. Transposing the matrix and assuming data reflects ' num2str(size(dat,1)) ' slices and ' num2str(size(dat,2)) ' volumes.']);
    dat = dat';
end

c = corr(dat);
% z-transform
cZ = .5*log( (1 + c) ./ (1 - c) );

sgap = size(c,1) ./ MB;

[avgS,avgAllNS,avgS_Z,avgAllNS_Z] = deal(zeros(size(c,1),1));
for i = 1:size(c,1)
    cntS = 0;
    cntNS = 0;
    ind = [(i-sgap):-sgap:1 (i+sgap):sgap:size(c,1)];
    simulSlices = [(i-sgap):-sgap:1 (i+sgap):sgap:size(c,1)];
    ind(ind<1) = [];
    % all simultaneously acquired slices
    avgS(i) = mean(c(i,ind));
    avgS_Z(i) = mean(cZ(i,ind));
    ind = [ind-1 ind+1];
    ind(ind<1|ind>size(c,1)) = [];
    % all non-simultaneously acquired slices
    sliceList = 1:1:size(c,1);
    sliceList([i simulSlices]) = [];
    avgAllNS(i) = mean(c(i,sliceList));
    avgAllNS_Z(i) = mean(cZ(i,sliceList));
end

simulSlicesZ = mean(avgS_Z); % avg z-transformed correlation in simultaneously acquired slices
allnonSimulSlicesZ = mean(avgAllNS_Z); % avg z-transformed correlation in ALL non-simultaneously acquired slices   
    
















end