function [ U, V, W] = Translation( u, v, x, y)

a1 = x.*v - y.*u;

a = sum(sum(v.^2));
b = sum(sum(u.^2));
c = sum(sum(a1.^2));
d = -sum(sum( u.*v ));
e = sum(sum( u.*a1 ));
f = - sum(sum( v.*a1 ));

G = [a d f; d b e; f e c];

eigVal = eig(G);

mineigVal = min(eigVal);

U = (b-mineigVal)*(c-mineigVal)-f*(b-mineigVal)-d*(c-mineigVal)+e*(f+d-e);
V = (c-mineigVal)*(a-mineigVal)-d*(c-mineigVal)-e*(a-mineigVal)+f*(d+e-f);
W = (a-mineigVal)*(b-mineigVal)-e*(a-mineigVal)-f*(b-mineigVal)+d*(e+f-d);

end

