function [ angleImage ] = anglefield( OF1, OF2, magn)

mask = OF2<0;
angleImage = (acos(OF1./magn).*180/pi) .* (mask.*2-1) + abs(mask-1)*360;

end