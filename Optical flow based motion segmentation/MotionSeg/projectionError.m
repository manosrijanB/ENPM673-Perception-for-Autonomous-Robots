function [ projectionError ] = projectionError( TransOF, TransOFideal)

magn = sqrt(TransOF(:,:,1).^2+TransOF(:,:,2).^2);
magn2 = sqrt(TransOFideal(:,:,1).^2+TransOFideal(:,:,2).^2);

cosBeta = min(dot(TransOF, TransOFideal, 3)./(magn.*magn2),1);

projectionError = magn.*sqrt(abs(1-cosBeta.^2));

end

