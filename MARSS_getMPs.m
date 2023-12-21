function MPpath = MARSS_getMPs(fn,MB, workingDir)
% This function generates 6 rigid-body Motion parameters via SPM12
% Inputs:
% fn = full path to volume timeseries
% MB = multiband acceleration factor
% workingDir = directory to save motion parameters to
%
% Outputs:
% MPpath = full path to which motion parameter text file was saved
% 
% Motion Parameter text file is named with prefix 'rp'
%-----------------------------------------------------------------------------
% Philip Tubiolo, John C. Williams, Mahika Gupta, & Jared Van Snellenberg 2023
%-----------------------------------------------------------------------------

V = spm_vol(fn);

if mod(V(1).dim(3),MB)
    error(['# of slices and MB factor are incompatible for ' fn '.']);
end

[p,f,~] = fileparts(fn);

% check workingDir for MPs
ex = exist([workingDir filesep 'rp_' f '.txt'],'file');

if ~ex
    % prepare spm job batch
    a = spm5_image_list(length(V),{fn});
    matlabbatch{1}.spm.spatial.realign.estimate.data = {a{1}'};

    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.quality = 0.9000;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.sep = 4;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.fwhm = 5;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.rtm = 1;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.interp = 2;
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estimate.eoptions.weight = '';

    spm_jobman('run',{matlabbatch});
% Delete .mat file output by SPM (prevents actual volume realignment from
% being performed
    if isempty(p)
        delete([f '.mat']);
    else
        delete([p filesep f '.mat']);
    end
end
% MP text file is automatically generated to the same directory as the
% input timeseries. This ensures that the file is also in the MARSS working
% directory
if ~exist([workingDir filesep 'rp_' f '.txt'],'file')
    copyfile([p filesep 'rp_' f '.txt'],workingDir)
end
MPpath = fullfile(workingDir,['rp_' f '.txt']);
end


