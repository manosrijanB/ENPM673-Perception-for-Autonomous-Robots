function [ TransAngle, RotadjustedOF, RotadjustedAF, dif, RotValue_prev, pE] = CameraMotion( OF,u, v, row, col, mask, height, width, RotValue_prev, x_comp, y_comp )
% INPUT: OF             optical flow (heightxwidthx2 matrix)
%        u, v           values of the optical flow OF(:,:,1) and OF(:,:,2), 
%                       which belong to a single motion like background.
%                       Pixels of a moving object for example a car are not
%                       included.
%        row, col       position
%                       Example: pixel position of u is (row,col, 1)
%        mask           binary (heightxwidth)-matrix. 
%                       1 if optical flow at this position belongs
%                       to a single motion like backround otherwise 0.
%        height, width  size of optical flow
%
% OUTPUT: TransAngle    ideal translational anglefield, which discribes
%                       the pure camera motion (motion of the static
%                       background)
%         RotadjustedOF observed translational flow field 
%                       (observed optical flow - ideal rotational flow)          
%         RotadjustedAF anglefield of RotadjustedOF  
%         dif           angledifference between anglefield of RotadjustedOF
%                       TransAngle
%         pE            pseudo projection error

    x_comp_idx = col - width/2 - 1;
    y_comp_idx = row - height/2 - 1;
    
    focallength=5*width/6.16; %519.48; %focal length in pixel = (f in [mm]) * (imagewidth in pixel) / (CCD width in mm)
    
    %--------------------------------------------------------------------------
    % gradient decent over rotations
    %--------------------------------------------------------------------------
    f = @(x)fcn_to_minimize(x, double(u), double(v), x_comp_idx, y_comp_idx, focallength);
    options = optimset('LargeScale','off', 'Display', 'off');
    [RotValue, ~] = fminunc( f, RotValue_prev, options); 
    RotValue_prev = RotValue;

    [RotOF] = getRotofOF( RotValue, x_comp, y_comp, focallength);
    RotadjustedOF = OF - RotOF;
    magn = sqrt(RotadjustedOF(:,:,1).^2+RotadjustedOF(:,:,2).^2);
    RotadjustedAF = anglefield(RotadjustedOF(:,:,1), RotadjustedOF(:,:,2), magn);

    flow(:,:,1) = RotadjustedOF(:,:,1) .* mask;
    flow(:,:,2) = RotadjustedOF(:,:,2) .* mask;
    
    % find best fitting translational anglefield to anglefieldTransOF 
    [ U, V, W] = Translation( flow(:,:,1), flow(:,:,2), x_comp, y_comp);
    TransOF_ideal(:,:,1) = -U+x_comp.*W;
    TransOF_ideal(:,:,2) = -V+y_comp.*W;
    
    idx = find(mask);
    [TransAngle, dif] = chooseTrans( idx, TransOF_ideal, RotadjustedAF );
    
    pE = projectionError_new( TransAngle, RotadjustedAF, magn );

end


