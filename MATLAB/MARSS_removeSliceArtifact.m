function [postMARSS_fname, postMARSS_avgSlcArt_fname] = MARSS_removeSliceArtifact(filename,MB,MPs,workingDir)
% This function performs the full MARSS procedure on a single run and
% outputs a corrected timeseries, artifact signal timeseries, and average
% spatial artifact distribution
%
% Inputs:
% filename = full path to 4D timeseries file
% MB = multiband acceleration factor
%
% Outputs:
% postMARSS_fname = full path to corrected timeseries file
% postMARSS_avgSlcArt_fname = full path to average artifact spatial
% distribution file (3D volumetric data)
%-----------------------------------------------------------------------------
% Philip Tubiolo, John C. Williams, Mahika Gupta, & Jared Van Snellenberg 2023

% When used, please CITE:  
%-----------------------------------------------------------------------------

if ~iscell(filename)
    filename = {filename};
end

for i = 1:length(filename)
    V = spm_vol(filename{i});
    Y = spm_read_vols(V);
    for j = 1:length(V)
        V(j).dt = [16 0];
    end
    Y = double(Y); 


    if mod(size(Y,3),MB)
        warning(['# of slices and MB factor are incompatible for ' filename{i} '. Skipping.']);
        continue
    end


    MPs = zscore(MPs);
    % nuisance parameter design matrix
    Xest = [MPs MPs.^2 [zeros(1,6); diff(MPs)] [zeros(1,6); diff(MPs)].^2]; 

    % estimate artifact signal in each slice
    [artifactEstimate,nonsliceGlobalSignal] = MARSS_estimateSliceArtifact(Y,MB,Xest);


    [Ya,Yart] = deal(zeros(size(Y)));
    for j = 1:size(artifactEstimate,2)

        % final MARSS design matrix
        Xcalc = [ones(size(artifactEstimate,1),1) zscore([artifactEstimate(:,j) nonsliceGlobalSignal(:,j) Xest])];

        Yt = reshape(squeeze(Y(:,:,j,:)),[],size(Y,4))';
        % perform regression
        B = Xcalc\Yt;

        art = Xcalc(:,2)*B(2,:);
        % subtract artifact estimation from original timeseries data
        Yta = reshape(Yt' - art',size(Y,1),size(Y,2),1,size(Y,4));
        Yartt = reshape(art',size(Y,1),size(Y,2),1,size(Y,4));
        Ya(:,:,j,:) = Yta;
        Yart(:,:,j,:) = Yartt;
    end
    % create fname
    [~,f,x] = fileparts(filename{i});

    for j = 1:size(Y,4)
        V(j).fname = [workingDir filesep 'za' f x]; %Corrected Data
        spm_write_vol(V(j),Ya(:,:,:,j));

        V(j).fname = [workingDir filesep f '_slcart' x]; %Isolated artifact timeseries
        spm_write_vol(V(j),Yart(:,:,:,j));
    end

    Yaavg = mean(abs(Yart),4);

    V(1).fname = [workingDir filesep f '_AVGslcart' x]; %Average artifact distribution

    spm_write_vol(V(1),Yaavg);

end
postMARSS_fname = [workingDir filesep 'za' f x];
postMARSS_avgSlcArt_fname = [workingDir filesep f '_AVGslcart' x];

end
