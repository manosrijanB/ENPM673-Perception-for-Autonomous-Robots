function [ TransAF_ideal_bg, RotadjustedOF, RotadjustedAF, pE ] = getInitialCameraMotion( video, RANSAC, numStart, dirFlow )

if (RANSAC == false)
    %compute CameraMotion based on entire opticalflow
    OF = load(sprintf('%s/%s%03d.%s', dirFlow, 'OF', numStart, 'mat'));
    OF = OF.uv;
    
    [height, width, ~] = size(OF);
    
    [~, ~, u] = find(OF(:,:,1));
    [row, col, v] = find(OF(:,:,2));
    
    %create meshgrid used in camera motion estimation
    xmin = floor(-(width-1)/2);
    xmax = floor((width-1)/2);
    ymin = floor(-(height-1)/2);
    ymax = floor((height-1)/2);
    x_comp = repmat((xmin:xmax), height, 1);
    y_comp = repmat((ymin:ymax).', 1, width);

    [ TransAF_ideal_bg, RotadjustedOF, RotadjustedAF, ~, ~, pE] = CameraMotion( OF, u, v, row, col, ones(height, width), height, width, [0 0 0], x_comp, y_comp );
    clear OF u v row col bg_mask height width
else
    %load precomputed CameraMotion with RANSAC
    dirInitial = sprintf('../%s/%s/%s', 'results',  video, 'Initialization_RANSAC');
    
    init = load(sprintf('%s/%s.%s', dirInitial, 'initialParam', 'mat'));
    RotadjustedAF = init.RotadjustedAF;
    RotadjustedOF = init.RotadjustedOF;
    TransAF_ideal_bg = init.TransAF_ideal_bg;
    pE = init.pE;
end

end

