function [] = initialize( video, imageformate, firstidxOF, dirFlow )

rng default

average_pixelError = realmax;

OF = load(sprintf('%s/%s%03d.%s', dirFlow, 'OF', firstidxOF, 'mat'));
    OF = OF.uv;

[height, width, ~] = size(OF);

%create superpixels
Img = imread(sprintf('%s/%s%03d%s', sprintf('../%s/%s/%s', 'data', video, 'frames'), [video, '_'], firstidxOF, imageformate));
regionSize = 20 ;
regularizer = 0.5 ;
segments = vl_slic(im2single(Img), regionSize, regularizer) ;

%outlier threshold
errorThresh = 0.1;

% create bucket matrix (all four corners of the image)
bucket_height = floor(height./5);
bucket_width = floor(width./5);

bucket_topL = segments(1:bucket_height, 1:bucket_width);
bucket_topR = segments(1:bucket_height, (width-bucket_width+1):width);
disp(width)
bucket_bottomL = segments((height-bucket_height+1):height, 1:bucket_width);
bucket_bottomR = segments((height-bucket_height+1):height, (width-bucket_width+1):width);

bucket_matrix = zeros(bucket_height, bucket_width, 4);
bucket_matrix(:,:,1) = bucket_topL;
bucket_matrix(:,:,2) = bucket_topR;
bucket_matrix(:,:,3) = bucket_bottomL;
bucket_matrix(:,:,4) = bucket_bottomR;

%create meshgrid used in camera motion estimation
xmin = floor(-(width-1)/2);
xmax = floor((width-1)/2);
ymin = floor(-(height-1)/2);
ymax = floor((height-1)/2);
x_comp = repmat((xmin:xmax), height, 1);
y_comp = repmat((ymin:ymax).', 1, width);

disp('Initialization...');

clearvars bucket_topL bucket_topR bucket_bottomL bucket_bottomR bucket_width bucket_height regularizer regionSize Img imageformate firstidxOF dirFlow 
tic;    
for loopRANSAC = 1:5000
        patch = bucketing( segments, bucket_matrix );
        
        flow_mask(:,:,1) = OF(:,:,1).*patch;
        flow_mask(:,:,2) = OF(:,:,2).*patch;
    
        [~, ~, u] = find(flow_mask(:,:,1));
        [row, col, v] = find(flow_mask(:,:,2));
    
        [TransAF_ideal_bg_current, RotadjustedOF_current, RotadjustedAF_current, dif_current, RotValue_prev_current, pE_current] = CameraMotion( OF, u, v, row, col, patch, height, width, [0 0 0], x_comp, y_comp );
            
        average_pixelError_current = (pE_current>errorThresh);
        average_pixelError_current =  sum(sum(average_pixelError_current));
             
        if average_pixelError_current < average_pixelError
            average_pixelError = average_pixelError_current;
            TransAF_ideal_bg = TransAF_ideal_bg_current;
            RotadjustedOF = RotadjustedOF_current;
            RotadjustedAF = RotadjustedAF_current;  
            dif = dif_current;
            pE = pE_current;
            selectedPatches = patch;
            RotValue_prev = RotValue_prev_current;
        end
        
        if (mod(loopRANSAC,100) == 0 )
            disp(sprintf('%s%d/%d', 'Iteration ', loopRANSAC, 5000));
        end
end
elapsed = toc;
text = sprintf('%s', 'Initialization finished in %gsec\n');
fprintf(text, elapsed);
    
%save best result after x iterations
mkdir(sprintf('../%s/%s/%s', 'results',  video, 'Initialization_RANSAC'));
dirInitial = sprintf('../%s/%s/%s/%s', 'results',  video, 'Initialization_RANSAC', 'initialParam');
    save(dirInitial, 'TransAF_ideal_bg', 'RotadjustedOF', 'RotadjustedAF', 'dif', 'pE', 'selectedPatches', 'RotValue_prev');

end

