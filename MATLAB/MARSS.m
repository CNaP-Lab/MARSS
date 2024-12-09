function MARSS(timeseriesFile, MB, workingDir)
% This is the main function for Multiband Artifact Regression in Simultaneous Slices (MARSS).

% Inputs:
% timeseriesFile = file path to unprocessed, unwarped, multiband fMRI
% timeseries
% MB = multiband acceleration factor of the EPI sequence used to
% acquire data
% workingDir = directory in which to save all intermediates and outputs for
% a subject (MARSS operates on a single run and will create a folder within workingDir for
% each run)
%
% Outputs (files saved in a run-specific subfolder of workingDir):
% (runName)_MARSS_SliceCorrelations.mat = struct that contains slice
% correlation information pre- and post-MARSS for the given run
% za(runName).nii = MARSS-corrected timeseries
% (runName)_slcart.nii = 4D-timeseries of MARSS-isolated artifact signal
% (runName)_AVGslcart.nii = 3D average spatial distribution of MARSS-isolated
% artifact signal
% MARSS_(runName).png = summary diagnostic figure that includes the pre- and
% post-MARSS slice correlation matrices, as well as orthviews of the
% average artifact distribution ((runName)_AVGslcart.nii)


% Dependencies (included with distribution):
% subaxis (Copyright (c) 2014, Aslak Grinsted, All rights reserved.)
% SPM12 (Copyright (C) 1991,1994-2019 Wellcome Trust Centre for Neuroimaging, All rights reserved.)
%-----------------------------------------------------------------------------
% Philip Tubiolo, John C. Williams, Mahika Gupta, & Jared Van Snellenberg 2023

% When used, please CITE:  
%-----------------------------------------------------------------------------

    % get name of run
    [~,runName,~] = fileparts(timeseriesFile);

    % Create a new folder
    runDir = fullfile(workingDir, runName);
    if ~exist(runDir, 'dir')
        mkdir(runDir);
    end

    % empty struct for all data to go to
    runStruct = struct();
    
    disp('Generating pre-MARSS Motion Parameters...'); pause(eps); drawnow;
    % Generate motion parameters for that run and save them
    preMARSS_MPpath = MARSS_getMPs(timeseriesFile, MB, runDir);
    
    % Load motion parameter file and save in struct
    motionParams = load(preMARSS_MPpath);
    runStruct.preMARSS.motionParams = motionParams;

    disp('Generating pre-MARSS Slice Correlations...'); pause(eps); drawnow;
    % Generate slice correlation matrix and save in struct
    preMARSS_corrs = MARSS_mbCorrPlot(timeseriesFile,motionParams,MB);
    runStruct.preMARSS.sliceCorrelations.Z_avgSimulSliceCorr_motionRegressed = preMARSS_corrs.rdasimulSlicesZ;
    runStruct.preMARSS.sliceCorrelations.Z_avgAdjacentSliceCorr_motionRegressed = preMARSS_corrs.rdaAdjacentSlicesZ;
    runStruct.preMARSS.sliceCorrelations.R_avgSliceCorrDifference = zToR(preMARSS_corrs.rdasimulSlicesZ - preMARSS_corrs.rdaAdjacentSlicesZ);
    runStruct.preMARSS.sliceCorrelations.corrMat_motionRegressed = preMARSS_corrs.corrMat_motionRegressed;
    
    disp('Performing MARSS Procedure...'); pause(eps); drawnow;
    % Estimate and remove artifact function
    [postMARSS_fname, postMARSS_avgSlcArt_fname] = MARSS_removeSliceArtifact(timeseriesFile,MB,motionParams,runDir);
    
    disp('Generating post-MARSS Motion Parameters...'); pause(eps); drawnow;
    % Use function to generate motion parameters for new artifact-removed data
    %use za or something else to track down file name
    postMARSS_MPpath  =  MARSS_getMPs(postMARSS_fname, MB,runDir);
    
    % Load new motion parameters and save in struct
    postmotionParams = load(postMARSS_MPpath);
    runStruct.postMARSS.motionParams = postmotionParams;
    
    disp('Generating post-MARSS Slice Correlations...'); pause(eps); drawnow;
    % Generate slice correlation matrix for artifact-removed run and save in struct
    postMARSS_corrs = MARSS_mbCorrPlot(postMARSS_fname,postmotionParams,MB);
    runStruct.postMARSS.sliceCorrelations.Z_avgSimulSliceCorr_motionRegressed = postMARSS_corrs.rdasimulSlicesZ;
    runStruct.postMARSS.sliceCorrelations.Z_avgAdjacentSliceCorr_motionRegressed = postMARSS_corrs.rdaAdjacentSlicesZ;
    runStruct.postMARSS.sliceCorrelations.R_avgSliceCorrDifference = zToR(postMARSS_corrs.rdasimulSlicesZ - postMARSS_corrs.rdaAdjacentSlicesZ);
    runStruct.postMARSS.sliceCorrelations.corrMat_motionRegressed = postMARSS_corrs.corrMat_motionRegressed;

    % Save final struct 
    save(fullfile(runDir, [runName '_MARSS_SliceCorrelations.mat']), 'runStruct');

    % Load avg artifact distribution
    artDistHeader = spm_vol(postMARSS_avgSlcArt_fname);
    artDistDat = spm_read_vols(artDistHeader);
    maxArt = squeeze(max(max(max(artDistDat))));
    stdDat = std(artDistDat(:));
    meanArt = mean(artDistDat(:));
    % color limit for displaying artifact distribution (mean + 3*stddev to control for outlier values)
    colorMax = meanArt + (3.*stdDat);

    %% Generate multipanel summary figure
    subaxis(2, 6, 1:3,'SpacingVert',0,'SpacingHoriz',0.01)
    % preMARSS slice correlation matrix
    imagesc(runStruct.preMARSS.sliceCorrelations.corrMat_motionRegressed,[-1,1]);
    title('Uncorrected Data');
    subtitle(['\DeltaR = ' num2str(runStruct.preMARSS.sliceCorrelations.R_avgSliceCorrDifference)]);
    xlabel('Slice #');
    ylabel('Slice #');
    axis image
    set(gca,'FontSize',12);

    % postMARSS slice correlation matrix
    subaxis(2, 6, 4:6,'SpacingVert',0,'SpacingHoriz',0.01)
    imagesc(runStruct.postMARSS.sliceCorrelations.corrMat_motionRegressed,[-1,1]);
    title('Corrected Data');
    subtitle(['\DeltaR = ' num2str(runStruct.postMARSS.sliceCorrelations.R_avgSliceCorrDifference)]);
    xlabel('Slice #');
    ylabel('Slice #');
    axis image
    origSize = get(gca,'Position');
    % configure colorbar
    c = colorbar('eastoutside');
    ct = ylabel(c,'Pearson''s R');
    set(gca,'Position',origSize);
    set(gca,'FontSize',12);
    subaxis(2, 6, 7:8,'SpacingVert',0.05,'SpacingHoriz',0.01)

    % get orthviews of artifact distribution
    %saggital
    imagesc(imrotate(squeeze(artDistDat(round(size(artDistDat,1)/2)-2,end:-1:1,end:-1:1,:)),-90),[0 colorMax]);
    axis image
    set(gca,'YTick',[],'XTick',[]);
    set(gca,'FontSize',16);
    %coronal
    subaxis(2, 6, 9:10,'SpacingVert',0.05,'SpacingHoriz',0.01)
    imagesc(squeeze(artDistDat(end:-1:1,round(size(artDistDat,2)/2),end:-1:1,:))',[0 colorMax]);
    axis image
    set(gca,'YTick',[],'XTick',[]);
    origSize = get(gca,'Position');
    artC = colorbar('southoutside');
    ylabel(artC,'Signal Intensity');
    set(gca,'FontSize',12);
    set(gca,'Position',origSize);    
    %axial
    subaxis(2, 6, 11:12,'SpacingVert',0.22,'SpacingHoriz',0.008)
    imagesc(fliplr(imrotate(squeeze(artDistDat(end:-1:1,end:-1:1,round(size(artDistDat,3)/2))),-90)),[0 colorMax]);
    axis image
    set(gca,'YTick',[],'XTick',[]);
    set(gcf,'Position',[595   216   883   733]);
    set(gca,'FontSize',12);

    figTitle = annotation('textbox', [0 0.9 1 0.1], ...
    'String', runName, ...
    'EdgeColor', 'none', ...
    'HorizontalAlignment', 'center','Interpreter','none','fontweight', 'bold');
    figTitle.FontSize = 12;

    % export figure
    figureName = fullfile(runDir,['MARSS_' runName '.png']);
    saveas(gcf,figureName);
    close(gcf);
    disp(['Completed MARSS for ' runName '.']); pause(eps); drawnow;
end