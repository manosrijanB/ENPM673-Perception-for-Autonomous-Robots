function [ ] = motionSegmentation( dirResult, video, firstidxOF, lastidxOF, TransAF_ideal_bg_Initial, RotadjustedOF_Initial, RotadjustedAF_Initial, pE_Initial, imageformate )
% INPUT: video      name of the video.
%                   Example: video   'forest'
%                            when frames are named with forest_xxxxx.png'
%        firstidxOF number of the first optical flow matrix (often 1)                 
%        lastidxOF  number of last optical flow matrix, this is the number
%                   of the last videoframe-1
%        TransAF_ideal_bg_Initial
%                   estimated translational anglefiel based on the first
%                   opticalflow
%        RotadjustedOF_Initial
%                   Transalational opticalflow of first and second frame
%        RotadjustedAF_Initial
%                   Anglefield of transalational opticalflow of first and second frame
%        pE         modified Bruss and Horn error


%exponent b and multiplier a to evaluate kappa k=a*r^b
param_multiplier = 4; 
param_exponent = 1;
%minimal effectivness metric
OtsusThresh = 0.6;
%a maximum of 15 motion components can be observed in the first frame
%(will be dynamicaly increased throughout the video)
maxNumMotions = 15;
%a singel motion component must contain at least 10 pixels to get a
%motion estimate
minNumPixels = 10;

rng default

TransAF_ideal_bg  = TransAF_ideal_bg_Initial;
RotadjustedOF  = RotadjustedOF_Initial;
RotadjustedAF = RotadjustedAF_Initial;
pE = pE_Initial;

fnameFormat = '%s/%s%03d%s';
prefix_Error = 'OF';
dirFlow = sprintf('../%s/%s/%s', 'data', video, 'opticalflow');
fileExt = '.mat';
[height, width] = size(pE_Initial);
SegmentationCell = cell(lastidxOF-firstidxOF+1, 1);
newComp = zeros(height, width);
imbin_probability = zeros(height, width);
regionError_current = 0;
RotValue_prev = [0 0 0];
exist_idx = 0;
possibleNewMotionDetectedInPreviousFrame = false;
ImgSegmented = zeros(height, width);
flowed_Posterior = zeros(height, width);
tooSmall = ones(height, width);
likelihood_fg = zeros(height, width, maxNumMotions);
idx_MotionComp = cell(1,1,maxNumMotions);

xmin = floor(-(width-1)/2);
xmax = floor((width-1)/2);
ymin = floor(-(height-1)/2);
ymax = floor((height-1)/2);
x_comp = repmat(xmin:xmax, height, 1);
y_comp = repmat((ymin:ymax).', 1, width);

% -------------------------------------------------------------------------
% Output: Text
% -------------------------------------------------------------------------
text_begin = sprintf('%s%s%s', 'Segmenting video: ', video, '...\n' );
fprintf(text_begin);

for i=firstidxOF:lastidxOF

    tic;
    
    OF = load(sprintf(fnameFormat, dirFlow, prefix_Error, i, fileExt));
    OF = double(OF.uv);
       
    if (i == firstidxOF)

        % -------------------------------------------------------------------------
        % Segmetnation using Otsus method (first frame only)
        % -------------------------------------------------------------------------
        [thresh, effectiveness] = multithresh(pE, 1);
        imbin = (pE > thresh(1));
        
        n = 1;
        zeroMask = zeros(height, width);
     
        while (effectiveness > OtsusThresh && n<=maxNumMotions)

            %find region with largest average error
            CC = bwconncomp(imbin);

            for k=1:CC.NumObjects
                
                PixelList = CC.PixelIdxList{k};
                numPixel = length(PixelList);
            
                if numPixel > minNumPixels                    
                    regionError = sum(pE(PixelList))/numPixel;           
                    if regionError > regionError_current
                        regionError_current = regionError;
                        idx = k;
                        exist_idx = 1;
                    end                   
                else
                    tooSmall(CC.PixelIdxList{k}) = 0;                   
                end
                
            end
            
            %make sure that idx exists - component containing more than minNumPixels
            %pixels is found
            if(exist_idx == 1)      
                zeroMask(CC.PixelIdxList{idx}) = 1;
                newComp(CC.PixelIdxList{idx}) = 1;
                newMotionComp(:,:,n) = newComp.*effectiveness;
                idx_MotionComp(:,:,n) = {[CC.PixelIdxList{idx}]};
                imbin_probability = imbin_probability + newComp.*effectiveness;
                imbin = abs(zeroMask-1);
                pE = pE.*imbin;          
            end
            
            pE = pE.*tooSmall;
            [B, C, ~] = find(1-tooSmall);
            idx_tooSmallTOremove = sub2ind([height, width], B, C);
            clearvars B C wert
            [B, C, ~] = find(zeroMask);
            idx_MotionCompTOremove =  sub2ind([height, width], B, C);
            clearvars B C wert
            PixelList_toRemove = cat(1, idx_tooSmallTOremove, idx_MotionCompTOremove);
   
            Adif_likelihood_vec = reshape(pE, [height*width, 1]);
            A = zeros(height*width, 2);
            A(:,1) = 1:height*width;
            A(:,2) = Adif_likelihood_vec;
            [A_removedComp, ~] = removerows(A, 'ind',  PixelList_toRemove);
        
            [thresh, effectiveness] = multithresh( A_removedComp(:,2), 1);
            imbin = (pE > thresh(1));
        
            newComp = zeros(height, width);
            n = n+1;
        
            CC = [];
            exist_idx = 0;
            regionError_current = 0;
            clearvars idx
        end
        
        if(effectiveness<=0.6 && n==1)
            newMotionComp = zeros(height, width, 1);
        end
        
    else
            % -------------------------------------------------------------------------
            % computation of the camera motion given the optical flow of frame i and i+1 and a
            % mask (1-ImgSegmented), which is an estimate for static background.
            % -------------------------------------------------------------------------
            flow_mask(:,:,1) = OF(:,:,1).*bg_mask;
            flow_mask(:,:,2) = OF(:,:,2).*bg_mask;
    
            [~, ~, u] = find(flow_mask(:,:,1));
            [row, col, v] = find(flow_mask(:,:,2));
 
            [ TransAF_ideal_bg, RotadjustedOF, RotadjustedAF, ~, RotValue_prev] = CameraMotion( OF, u, v, row, col, bg_mask, height, width, RotValue_prev, x_comp, y_comp );

    end
    
    % computation of kappa, which dependent of the flow magnitude
    % kappa k=a*r^b
    magn = sqrt(RotadjustedOF(:,:,1).^2+RotadjustedOF(:,:,2).^2);
    
    kappa_Angle = (param_multiplier.*(magn.^param_exponent));
    kappa_Vec = reshape(kappa_Angle, 1, height*width);
    bessel = besseli(0, kappa_Vec);

    idx_MotionComp = idx_MotionComp(~cellfun('isempty',idx_MotionComp)) ;
    n = length(idx_MotionComp)+1;
    TransOF_ideal = zeros(height, width, 2, n);
    imbin_probability = abs(imbin_probability -1);
  
    % -------------------------------------------------------------------------
    % ith component likelihood: 
    % based on the rotation adjusted flowfield find the best fitting
    % translational motion for the ith motion component. 
    % Computation of the likelihood is using the von Mises
    % distribution with kappa, which is dependent on the flow magnitude
    % -------------------------------------------------------------------------
    for s = 1:(n-1)
         
         idx_MotionComp_i = cell2mat(idx_MotionComp(:,:,s));
         
         RotadjustedOF_MotionComp1 = RotadjustedOF(:,:,1);
         RotadjustedOF_MotionComp2 = RotadjustedOF(:,:,2);
         RotadjustedOF_MotionComp(:,:,1) = RotadjustedOF_MotionComp1(idx_MotionComp_i);
         RotadjustedOF_MotionComp(:,:,2) = RotadjustedOF_MotionComp2(idx_MotionComp_i);
         
         [y, x] = ind2sub([height, width], idx_MotionComp_i);
         x = x - width/2 - 1;
         y = y - height/2 - 1;
         
         % translational motion of ith component
         [ U, V, W] = Translation( RotadjustedOF_MotionComp(:,:,1), RotadjustedOF_MotionComp(:,:,2), x, y);
         TransOF_ideal(:,:,1,s) = -U+x_comp.*W;
         TransOF_ideal(:,:,2,s) = -V+y_comp.*W;
         [ Angle_MotionComp_ideal, ~] = chooseTrans( idx_MotionComp_i, TransOF_ideal(:,:,:,s), RotadjustedAF );
         
         % Computation of the likelihood of ith component based on
         Dif_Angle_fg = RotadjustedAF - Angle_MotionComp_ideal;
         [ Adif_conditioned_magn_fg, ~ ] = motionLikelihood( Dif_Angle_fg, bessel, kappa_Vec, height, width);

         likelihood_fg( :, :, s) = Adif_conditioned_magn_fg;
         
         clearvars RotadjustedOF_MotionComp 
    
    end  
    
    % -------------------------------------------------------------------------
    % Static background likelihood:  
    % Computation of the likelihood is using the von Mises
    % distribution with kappa, which is dependent on the flow magnitude
    % -------------------------------------------------------------------------
    Dif_Angle_bg = RotadjustedAF - TransAF_ideal_bg;
    
    [ Adif_conditioned_magn_bg, ~ ] = motionLikelihood( Dif_Angle_bg, bessel, kappa_Vec, height, width );
    likelihood_bg = Adif_conditioned_magn_bg;
    
    % -------------------------------------------------------------------------
    % likelihood of a newMotion 
    % -------------------------------------------------------------------------
    likelihood_newMotion = (1/(2*pi))*ones(height, width);
    
    % -------------------------------------------------------------------------
    % Segmentation using Bays rule (function updatePosterior)
    % -------------------------------------------------------------------------
    
    if (i == firstidxOF)
        bgPrior = imbin_probability;
        fgPrior = newMotionComp;
    end
    
    % -------------------------------------------------------------------------
    %Add a newMotion component only if a possible motion was observed in previous frame. 
    % -------------------------------------------------------------------------
    if (possibleNewMotionDetectedInPreviousFrame == false)
        newPrior = (1/(n+1)).*ones(height, width); 
        
        [bgMaskPosterior, fgMaskPosterior, newMaskPosterior, bgPrior, fgPrior, newMotionPrior] = computePosterior( likelihood_bg, likelihood_fg, likelihood_newMotion, OF, bgPrior, fgPrior, n-1, newPrior);
        
        AllPosteriors = cat(3, newMaskPosterior, fgMaskPosterior, bgMaskPosterior);
        [~, ind] = sort(AllPosteriors, 3);
        newMaskPosteriorPrev =  (ind(:,:,n+1)==1);
        
        if(sum(sum(ind(:,:,n+1)==1))>0)
            possibleNewMotionDetectedInPreviousFrame = true;
        
            newPrior = zeros(height, width)+eps;
            [bgMaskPosterior, fgMaskPosterior, newMaskPosterior, bgPrior, fgPrior, newMotionPrior] = computePosterior( likelihood_bg, likelihood_fg, likelihood_newMotion, OF, bgPrior, fgPrior, n-1, newPrior);
            AllPosteriors = cat(3, newMaskPosterior, fgMaskPosterior, bgMaskPosterior);
            [~, ind] = sort(AllPosteriors, 3);
        end
        
    else
        newPrior = Posterior2Prior(newMaskPosteriorPrev, OF);
        newPrior = newPrior./max(max(newPrior)).*(1/(n+1));
        possibleNewMotionDetectedInPreviousFrame = false;
        
        [bgMaskPosterior, fgMaskPosterior, newMaskPosterior, bgPrior, fgPrior, newMotionPrior] = computePosterior( likelihood_bg, likelihood_fg, likelihood_newMotion, OF, bgPrior, fgPrior, n-1, newPrior);
    
        AllPosteriors = cat(3, newMaskPosterior, fgMaskPosterior, bgMaskPosterior);
        [~, ind] = sort(AllPosteriors, 3);
    end
      
    idx_MotionComp = cell(1,1,n-1);
    
    %remove motionComponents smaller than minNumPixels pixels
    for t = 2:n
        bin_current = (ind(:,:,n+1)==n+2-t);
        CC = bwconncomp(bin_current);
        
        for k=1:CC.NumObjects    
            PixelList = CC.PixelIdxList{k};
            numPixel = length(PixelList);
            if numPixel <= minNumPixels
                bin_current(PixelList) = 0;
            end
        end
        
        pos = find(bin_current==1);
        %flowing binary motionComponent     
        if (~isempty(pos) && length(pos)>minNumPixels)
            flowed_Posterior(pos) = 1;
            AllPosteriors_flowed = flowingBin(flowed_Posterior, OF);
            [row,col] = find(AllPosteriors_flowed);
            if length(row)>minNumPixels 
                idx_MotionComp(:,:,n+2-t-1) = {sub2ind([height width], row, col)};            
                flowed_Posterior = zeros(height, width);
            else
                flowed_Posterior = zeros(height, width);
                idx_MotionComp(:,:,n+2-t-1) = [];
                fgPrior(:,:,n+2-t-1) = [];
                fgMaskPosterior(:,:,n+2-t-1) = [];
            end
        else
            idx_MotionComp(:,:,n+2-t-1) = [];
            fgPrior(:,:,n+2-t-1) = [];
            fgMaskPosterior(:,:,n+2-t-1) = [];
        end
    end
    
    numMotions = size(fgMaskPosterior, 3);
    
    newMotion = ind(:,:,n+1)==1;
    
    CC = bwconncomp(newMotion);
    numNewMotion = CC.NumObjects;
    
    % add a new found motion component and increase number of motion
    % components (numMotions)
    for k=1:numNewMotion               
         PixelList = CC.PixelIdxList{k};
         numPixel = length(PixelList);
            
          if numPixel > minNumPixels 
             flowed_Posterior(PixelList) = 1;
             AllPosteriors_flowed = flowingBin(flowed_Posterior, OF);
             [row,col] = find(AllPosteriors_flowed);
             if length(row)>minNumPixels
                numMotions = numMotions+1;
                idx_MotionComp(:,:,numMotions) = {sub2ind([height width], row, col)};
                flowed_Posterior = zeros(height, width);
                fgPrior(:,:,numMotions) = newMotionPrior;
             else
                flowed_Posterior = zeros(height, width);
             end
          end               
    end
    
    idx_MotionComp = idx_MotionComp(~cellfun('isempty',idx_MotionComp)) ;  

    ImgSegmented(ind(:,:,n+1)==(n+1)) = 1;
    
    SegmentationCell(i) = {1-ImgSegmented};

    bg_mask = flowingBin(SegmentationCell{i}, OF);
    bg_mask = 1-bg_mask;
    % -------------------------------------------------------------------------
    % Output: Text
    % -------------------------------------------------------------------------
    elapsed = toc;
    text = sprintf('%s%d/%d%s', 'frame ', i, lastidxOF-firstidxOF+1, ' computed in %gsec\n');
    fprintf(text, elapsed);
    n = numMotions+1;
    ImgSegmented = zeros(height, width);
    imbin = zeros(height, width);
    likelihood_fg = zeros(height, width, maxNumMotions);
    imbin_probability = zeros(height, width);
end

% -------------------------------------------------------------------------
% Save segmentation and segmented video in './results/videoname'
% -------------------------------------------------------------------------
saveSequencesFilename  = sprintf('%s/%s', dirResult,  'Segmentation.mat');
                save(saveSequencesFilename, 'SegmentationCell');
                
fprintf( 'Creating segmentation video...\n' );
createOverlayVideo(SegmentationCell, firstidxOF, lastidxOF, video, imageformate, dirResult, height, width);

fprintf( 'Segmentation finished\n' );

clearvars -except numexperiments dataset video imageformate firstidxOF lastidxOF param_multiplier param_exponent param_a param_k param_e param_m r_matrix segments RotadjustedAF_Initial RotadjustedOF_Initial TransAF_ideal_bg_Initial AngleDif_Initial pE_Initial;

end
    


