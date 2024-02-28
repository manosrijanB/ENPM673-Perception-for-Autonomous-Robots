function [ sum_Error ] = fcn_to_minimize( RotVec, u, v, x, y, f)

RotOF = getRotofOF( RotVec, x, y, f);
TransOF(:,:,1) = u - RotOF(:,:,1);
TransOF(:,:,2) = v - RotOF(:,:,2);

%--------------------------------------------------------------------------
%translational component (Berthold Horn, Robot Vision, p.409)
%--------------------------------------------------------------------------
[ U, V, W ] = Translation(TransOF(:,:,1), TransOF(:,:,2), x, y);
TransOFideal(:,:,1) = -U+x.*W;
TransOFideal(:,:,2) = -V+y.*W;

%--------------------------------------------------------------------------
%projection Error
%--------------------------------------------------------------------------
pError = projectionError(TransOF, TransOFideal);

sum_Error = sum(sum(pError));

end

