function [ projectionError ] = projectionError_new( idealTransAnglefield, RotadjustedAF, magnitude_RotadjustedFOF )
%--------------------------------------------------------------------------
% pseudo projection Error
%--------------------------------------------------------------------------
theta = abs( RotadjustedAF - idealTransAnglefield );
theta = min( theta, abs(360-theta));

case_A = magnitude_RotadjustedFOF.*sind(theta);
case_B = magnitude_RotadjustedFOF;

case_A = case_A.* (theta<=90);
case_B = case_B.* (theta>90);
projectionError = case_A + case_B;


end

