% Calculation element stiffness
function cand_Ke = calcu_candidate_Ke(angle,D0)
coors=[-0.5 -0.5;0.5 -0.5;0.5 0.5;-0.5 0.5];
GS_point = 1/sqrt(3)*[-1,-1;1,-1;1,1;-1,1];
cand_Ke = cell(length(angle),1);
for i = 1:length(angle)
    t = angle(i)/180*pi;
    s = sin(t); c = cos(t); s2 = s^2; c2 = c^2; cs = c*s;
    Tmat = [c2, s2, -2*cs; s2, c2, 2*cs; cs, -cs, c2-s2];
    Ke=zeros(8,8);
    for j=1:4
        s = GS_point(j,1); t = GS_point(j,2);
        J = [-(1-t) 1-t 1+t -(1+t);-(1-s) -(1+s) 1+s 1-s]/4;
        J0=J*coors;
        DxDy = J0\J;
        B1 = [DxDy(1,1), 0; 0, DxDy(2,1); DxDy(2,1), DxDy(1,1)];
        B2 = [DxDy(1,2), 0; 0, DxDy(2,2); DxDy(2,2), DxDy(1,2)];
        B3 = [DxDy(1,3), 0; 0, DxDy(2,3); DxDy(2,3), DxDy(1,3)];
        B4 = [DxDy(1,4), 0; 0, DxDy(2,4); DxDy(2,4), DxDy(1,4)];
        B = [B1 B2 B3 B4];
        Ke = Ke+B'*Tmat*D0*Tmat'*B*det(J0);
    end
    cand_Ke{i,1} = Ke;
end
end