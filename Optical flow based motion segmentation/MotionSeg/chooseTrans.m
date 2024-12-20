function [ TransAngle, dif] = chooseTrans( idx, TransOF_ideal, RotadjustedAF )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    %select translational anglefield with smalles difference in Angle
    magn = sqrt(TransOF_ideal(:,:,1).^2+TransOF_ideal(:,:,2).^2);
    TransAngle_1 = anglefield(TransOF_ideal(:,:,1), TransOF_ideal(:,:,2), magn);
    TransAngle_2 = mod(TransAngle_1+180, 360);

    dif_1 = abs( RotadjustedAF(idx) - TransAngle_1(idx) );
    dif_1 = min( dif_1, abs(360-dif_1));

    dif_2 = abs( RotadjustedAF(idx) - TransAngle_2(idx) );
    dif_2 = min( dif_2, abs(360-dif_2));

    if sum(sum(dif_1)) < sum(sum(dif_2))    
        TransAngle = TransAngle_1;
        dif = dif_1;
    else     
        TransAngle = TransAngle_2;
        dif = dif_1;
    end


end

