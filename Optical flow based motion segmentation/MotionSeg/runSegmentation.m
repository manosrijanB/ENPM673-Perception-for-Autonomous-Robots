function [] = runSegmentation( video ) 
%        video      name of the video.
%                   Example: video   'forest'
%                            when frames are named with forest_xxx.png'
%        dataset    name of the dataset. Example: dataset 'cbg'
%
%        Do you want to run the code including RANSAC initialization? 
%        if yes set RANSAC true
         RANSAC = true;
%        Do you want to use precomputed results of the RANSAC initialization?
%        if yes set RANSACcomputed true
         precomputedVal = false;
%--------------------------------------------------------------------------

if (RANSAC == false)
    precomputedVal = false;
end

mkdir('../results', sprintf('%s', video));
dirResult = sprintf('../results/%s', video);

%find start and end index of opticalflow
dirFlow = sprintf('../%s/%s/%s', 'data', video, 'opticalflow');
sprintf('%s/%s', dirFlow, 'OF*');
listFlowName = dir(sprintf('%s/%s', dirFlow, 'OF*'));

numStart = sscanf(listFlowName(1).name, 'OF%d.mat', [1 Inf]);
numEnd = sscanf(listFlowName(length(listFlowName)).name, 'OF%d.mat', [1 Inf]);

clear listFlowName

listFramesName = dir (sprintf('../data/%s/frames/*%s*', video, video));
[~, ~, frame_fileExtension] = fileparts(sprintf('../data/%s/frames/%s', video, listFramesName(1).name));

clear listFramesName

if((precomputedVal == false) && (RANSAC == true))
    %RANSAC initialization
    initialize( video, frame_fileExtension,  numStart, dirFlow );
    %get initial camera motion 
    [ TransAF_ideal_bg, RotadjustedOF, RotadjustedAF, pE ] = getInitialCameraMotion( video, RANSAC, numStart, dirFlow );
    %motion segmentation
    motionSegmentation( dirResult, video, numStart, numEnd, TransAF_ideal_bg, RotadjustedOF, RotadjustedAF, pE, frame_fileExtension );
else
    %get initial camera motion 
    [ TransAF_ideal_bg, RotadjustedOF, RotadjustedAF, pE ] = getInitialCameraMotion( video, RANSAC, numStart, dirFlow );
    %motion segmentation
    motionSegmentation( dirResult, video, numStart, numEnd, TransAF_ideal_bg, RotadjustedOF, RotadjustedAF, pE, frame_fileExtension );
end

end