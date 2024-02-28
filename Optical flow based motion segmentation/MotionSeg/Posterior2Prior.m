function [ PseudoPrior ] = Posterior2Prior(Posterior, flow)

%starttime = tic; 
[H,W] = size(Posterior);
sD = 0;  % Is this too small?%12
sDtot = sD*2+1;
spatialWeight = fspecial('gaussian', [sDtot sDtot], 1);

Padding=100;
a = 1+Padding;
b = H+Padding;
c = W+Padding;

% The SIZE of X and Y is H by W. However, X and Y index into the PADDED array.
[X,Y] = meshgrid(a:c, a:b);   
gridXY = cat(3, X, Y);
newPos = round(flow+gridXY);   % newPos indexes into PADDED array.
flipNewPos=flip(newPos,3);     % Make y come first.
reshapeNewPos=reshape(flipNewPos,[W*H 2]);
mappedPriors=accumarray(reshapeNewPos,Posterior(:),[H+2*Padding,W+2*Padding]);
PseudoPrior=conv2(mappedPriors(a:b,a:c),spatialWeight,'same');

end