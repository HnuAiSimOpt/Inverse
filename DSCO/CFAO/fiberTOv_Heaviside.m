function comp = fiberTOv_Heaviside(nelx,nely,x,T,rmin,penal,volfrac,model)
nele = nelx*nely;  Ersatz = 1e-9;
% Preparation for FEA
D0 = model.D0;
coors=[-0.5 -0.5;0.5 -0.5;0.5 0.5;-0.5 0.5];
GS_point = 1/sqrt(3)*[-1,-1;1,-1;1,1;-1,1];
edofMat = model.edofMat;
iK = model.iK;
jK = model.jK;
% Boundary conditions
F = model.F;
freedofs = model.freedofs;
U = zeros(2*(nely+1)*(nelx+1),1);
% Preparation for filter
[H,Hs] = filter2d(rmin,nelx,nely);

% Initialize iteration
TPhys = T;  xPhys = x;
loop = 0;   change = 1;  loopMax = 80;  comp = zeros(loopMax,1);
while change > 1e-4  && loop < loopMax
    loop = loop +1;
    % Modeling
    sK = zeros(nele,1);
    KE = cell(nele,1);
    T_rad = (pi/180)*TPhys;
    for i = 1:nele
        if(xPhys(i)==0)
            A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
            A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
            B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
            B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
            nu = 0.3; Ke = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
        else
            s = sin(T_rad(i)); c = cos(T_rad(i)); s2 = s^2; c2 = c^2; cs = c*s;
            Tmat = [c2, s2, -2*cs; s2, c2, 2*cs; cs, -cs, c2-s2];
            Ke = zeros(8,8);
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
                Ke = Ke + B'*Tmat*D0*Tmat'*B*det(J0);
            end
        end
        KE{i,1} = Ke;
        sK((i-1)*64+1:i*64,1) = (Ersatz+(1-Ersatz)*xPhys(i)^penal)*Ke(:);
    end
    K = sparse(iK,jK,sK);  K = (K+K')/2;
    % FEA
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    % Objective function and sensitivity analysis
    comp(loop) = F'*U;
    dc = zeros(nely,nelx);
    for i = 1:nele
        Ue = U(edofMat(i,:));
        dKe_dx = penal*(1-Ersatz)*(xPhys(i)^(penal-1))*KE{i};
        dc(i) = -Ue'*dKe_dx*Ue;
    end
    dv = ones(nele,1);
    % Filter
    dc = H*(dc(:)./Hs);
    dv = H*(dv(:)./Hs);
    l1 = 0; l2 = 1e9; move = 0.1;
    while (l2-l1)/(l1+l2) > 1e-3
        lmid = 0.5*(l2+l1);
        xnew = max(0,max(x-move,min(1,min(x+move,x.*sqrt(-dc./dv/lmid)))));
        xPhys(:) = (H*xnew(:))./Hs;
        if sum(xPhys(:)) > volfrac*nelx*nely, l1 = lmid; else l2 = lmid; end
    end
    change = max(abs(xnew(:)-x(:)));
    x = xnew;
    % inner loop
    [T, TPhys] = innerFiberTO(nelx, nely, rmin, penal, T, xPhys, model, 5);
    % Print result
    fprintf(' It.:%5i Obj.:%11.4f Vol.:%7.3f ch.:%7.3f\n',loop,comp(loop), ...
        mean(xPhys(:)),change);
    % Plot
    figure(1);colormap(gray); imagesc(1-reshape(xPhys,nely,nelx)); caxis([0 1]); axis equal; axis off; drawnow;
end
xPhys = reshape(xPhys,nely,nelx);
T = reshape(T,nely,nelx);
display_result(comp,xPhys,T);
end