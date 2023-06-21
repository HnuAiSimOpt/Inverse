function [T, TPhys] = innerFiberTO(nelx, nely, rmin, penal, T, XPhys, model, loopmax,caseind)
% Preparation for FEA
F = model.F;
D0 = model.D0;
iK = model.iK;
jK = model.jK;
freedofs = model.freedofs;
edofMat = model.edofMat;
coors=[-0.5 -0.5;0.5 -0.5;0.5 0.5;-0.5 0.5];
GS_point = 1/sqrt(3)*[-1,-1;1,-1;1,1;-1,1];
U = zeros(2*(nelx+1)*(nely+1),1);
nele = nelx*nely;  Ersatz = 1e-9;  Tmin = -90;  Tmax = 90;   loop = 0;   change = 1;
% Preparation for filter
[H,Hs] = filter2d(rmin,nelx,nely);
TPhys = (H*(XPhys.*T))./Hs;
while change > 0.1
    loop = loop +1;
    if(loop>loopmax); break; end
    % Modeling
    sK = zeros(64*nele,1);
    dKe_dt = cell(nele,1);
    T_rad = (pi/180)*TPhys;
    for i = 1:nele
        if(XPhys(i)==0)
            A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
            A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
            B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
            B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
            nu = 0.3; Ke = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
            dKe=ones(8,8);
        else
            s = sin(T_rad(i)); c = cos(T_rad(i)); s2 = s^2; c2 = c^2; cs = c*s;
            Tmat = [c2, s2, -2*cs; s2, c2, 2*cs; cs, -cs, c2-s2];
            dT_dt = [-2*cs, 2*cs, 2*s2-2*c2; 2*cs, -2*cs, 2*c2-2*s2; c2-s2, s2-c2, -4*cs];
            Ke=zeros(8,8);  dKe=zeros(8,8);
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
                dKe = dKe + B'*dT_dt*D0*Tmat'*B*det(J0) + B'*Tmat*D0*dT_dt'*B*det(J0);
            end
        end
        sK((i-1)*64+1:i*64,1) = (Ersatz+(1-Ersatz)*XPhys(i)^penal)*Ke(:);
        dKe_dt{i,1} = (Ersatz+(1-Ersatz)*XPhys(i)^penal)*dKe;
    end
    K = sparse(iK,jK,sK);  K = (K+K')/2;
    % FEA
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    % Objective function and sensitivity analysis
    if caseind ==3
        comp(loop) = 2*F'*U;
    else
        comp(loop) = F'*U;
    end
    dc = zeros(nele,1);
    for i = 1:nele
        Ue = U(edofMat(i,:));
        dc(i)= -Ue'*dKe_dt{i,1}*Ue;
    end
    dc = (H*dc)./Hs;
    % Steepest descent optimization with conjugate mapping
    Told = T;  eta = 3;  sigma = 5;
    % dc = dc.*exp(1-abs(dc));
    T = max(Tmin,max(T-sigma,min(Tmax,min(T+sigma,T-eta*dc))));
    %TPhys = (H*T)./Hs;
    TPhys = (H*(XPhys.*T))./Hs;
    change = max(abs(Told-T));
        
    %%
%     Map = make_bitmap_unconvergent(angle,nelx,nely,0.95,xPhys);
%     figure(7)
%     image([flip(Map ,2), Map]); axis image off; drawnow;
end