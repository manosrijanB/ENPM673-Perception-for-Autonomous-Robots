function [ Adif_likelihood, Dif_Angle ] = motionLikelihood( Dif_Angle, Jo, kappa_Vec, height, width)

%Angle difference containing positiv and negativ Values
case_A = Dif_Angle > 180;
case_B = Dif_Angle < -180;
case_C = abs((case_A + case_B) - 1);
Dif_Angle_groesser180 = (Dif_Angle - 360) .* case_A;
Dif_Angle_kleinerNeg180 = (Dif_Angle + 360) .* case_B;
Dif_Angle_kleiner180 = Dif_Angle .* case_C;

Dif_Angle = (Dif_Angle_groesser180 + Dif_Angle_kleiner180 + Dif_Angle_kleinerNeg180)*(pi/180);

%von Mises distributed angle differences
Dif_Angle_Vec = reshape(Dif_Angle, 1, height*width);
Adif_vonMises = (exp(kappa_Vec.*cos(Dif_Angle_Vec))) ./ (2.*pi.*Jo);
Adif_likelihood = reshape(Adif_vonMises, height, width);

Adif_likelihood(isnan(Adif_likelihood)) = 0.01;

end