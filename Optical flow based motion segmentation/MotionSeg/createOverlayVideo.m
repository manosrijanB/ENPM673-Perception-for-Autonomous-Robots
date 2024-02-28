function [ ] = createOverlayVideo(SegmentationCell, firstidxOF, lastidxOF, video, imageformate, dirVideo, height, width)
% INPUT: SegmentationCell segmentations (numx1 cell array)     
%        firstidxOF       number of first frame (mostly 1)    
%        lastidxOF        number of last frame-1 
%        video            video name, for example 'forest'
%        imageformate     formate of the frame, for example '.png'
%        dirVideo         directory where video will be saved

fnameFormat = '%s/%s%03d%s';
prefix_Error = [video, '_'];
dirImg = sprintf('../%s/%s/%s', 'data', video, 'frames');
fileExt = imageformate;

videoname  = sprintf('%s.%s', video,  'avi');
outputVideo = VideoWriter(fullfile(dirVideo, videoname));
open(outputVideo);

alpha = 0.3;

for i = firstidxOF:lastidxOF
    
    origImg = imread(sprintf(fnameFormat, dirImg, prefix_Error, i, fileExt));

    segmentation = cell2mat(SegmentationCell(i,1));

    %create red mask
    mask(:,:,1) = segmentation ;
    mask(:,:,2) = segmentation .* 0.01;
    mask(:,:,3) = segmentation .* 0.01;

    Img(:,:,1) = rgb2gray(origImg);
    Img(:,:,2) = rgb2gray(origImg);
    Img(:,:,3) = rgb2gray(origImg);

    %create overlay
    overlay = im2double(Img);

    for n = 1 : width
        for m = 1:height

            if (mask(m,n,1)>0 && mask(m,n,2)>0  && mask(m,n,3)>0)
            
                overlay(m,n,1) = alpha .* overlay(m,n,1) + (1 - alpha) .* mask(m,n,1);
                overlay(m,n,2) = alpha .* overlay(m,n,2) + (1 - alpha) .* mask(m,n,2);
                overlay(m,n,3) = alpha .* overlay(m,n,3) + (1 - alpha) .* mask(m,n,3);

            end
        end
    end
    
    %origImg = insertText(origImg, [1 20], 'original video','FontSize', 22, 'BoxColor', 'black', 'BoxOpacity', 0.6, 'TextColor', 'white');
    %overlay = insertText(overlay, [1 20], 'our Method','FontSize', 22, 'BoxColor', 'black', 'BoxOpacity', 0.6, 'TextColor', 'white');

    frame = [im2uint8(origImg) im2uint8(overlay)];
    writeVideo(outputVideo, frame);

end

close(outputVideo);

end

