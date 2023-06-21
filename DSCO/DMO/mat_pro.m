function [Ex, Ey, Gxy, NUxy, NUyx] = mat_pro(rate,Ef,Em,Nuf,Num)
% rate    : Volume fraction of the fiber
% Ef,Em   : Elastic modulus of matrix and fiber
% Nuf,Num : Poisson's ratio of matrix and fiber
Gf = Ef/2/(1+Nuf);
Gm = Em/2/(1+Num);
Ex = rate*Ef + (1-rate)*Em;
Ey = (Ef*Em)/(rate*Em+(1-rate)*Ef);
Gxy = (Gf*Gm)/(rate*Gm+(1-rate)*Gf);
NUxy = rate*Nuf + (1-rate)*Num;
NUyx = Ey*NUxy/Ex;
end