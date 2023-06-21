function [xPhys,model,angle,obj] = DMO_nMat(nelx,nely,volfrac,rmin,caseind)
global convergenceRate_all singleangleRate_all loop_global
% if(nargin == 0); nelx =100; nely = 60; volfrac = 0.5; rmin = 4; caseind = 3; end
nele = nelx*nely;   ndof = 2*(nelx+1)*(nely+1);  move = 0.2;
epsilon = 1e-9;  showgap = 5;
penallist = [3];  % ³õÊ¼³Í·£Ö¸Êý
% Material properties and design variables
rate = 0.5;  Ef = 1;  Em = 1/15;  Nuf = 0.22;  Num = 0.38;                 % ²Î¿¼ 10.1016/j.compstruct.2018.06.020
[Ex, Ey, Gxy, NUxy, NUyx] = mat_pro(rate,Ef,Em,Nuf,Num);
% Ex=165; Ey=9, Gxy=6, NUxy=0.25, NUyx=0.1;
% Preparation for candidate material constitutive matrix
D0 = [Ex/(1-NUxy*NUyx), NUyx*Ex/(1-NUxy*NUyx), 0; NUxy*Ey/(1-NUxy*NUyx), Ey/(1-NUxy*NUyx), 0; 0, 0, Gxy];
% D0 = [2 0.3 0;0.3 1 0; 0 0 0.25];   %https://doi.org/10.1016/j.compstruct.2020.111900
% angle = [0,90]; 
% angle = [0,45];
% angle = [-30,30]; 
%  angle = [0,-45,45,90]; 331.63
%  angle = [0,-30,30,90]; 
 angle = [0,-60,60,90]; 
% angle = [0,-45,45,90,-30,-60,30,60]; 
% angle = [0,-45,45,90,-30,30]; 
% angle = [0,-45,45,90,30,60]; 
% angle = [0,-30,-60,30,60,90];
% angle = [0,-45,45,90,-30,-60];
nAng = length(angle);
cand_Ke = calcu_candidate_Ke(angle,D0);
% Preparation for FEA
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx);
edofVec = reshape(2*nodenrs(1:end-1,1:end-1)+1,nelx*nely,1);
edofMat = repmat(edofVec,1,8)+repmat([0 1 2*nely+[2 3 0 1] -2 -1],nelx*nely,1);
iK = reshape(kron(edofMat,ones(8,1))',64*nelx*nely,1);
jK = reshape(kron(edofMat,ones(1,8))',64*nelx*nely,1);
switch caseind
    case 1  % Ðü±ÛÁº
        loaddofs = 2*(nely+1)*nelx+nely+2;
        fixind= 1:(nely+1);
        fixeddofs = [2*fixind(:)-1,2*fixind(:)];
    case 2  % ¼òÖ§Áº
        loaddofs = 2;
        fixeddofs = [1:2:2*(nely+1),ndof-1,ndof];
    case 3
        loaddofs = [2*(nely+1),2*(nely+1)*(nelx/2)+2];
        fixeddofs = [1:2:2*(nely+1),ndof-1,ndof];
    case 4
        loaddofs = [2*(nely+1)*(nelx/4)+2,2*(nely+1)*(nelx/2+1),2*(nely+1)*(3*nelx/4)+2];
        fixeddofs = [2*nely+1,2*(nely+1),ndof-1,ndof];
    otherwise
        disp('Case identifier error');
end
alldofs = [1:2*(nely+1)*(nelx+1)];
freedofs = setdiff(alldofs,fixeddofs);
NF = length(loaddofs);
if caseind ==3
    F = sparse(loaddofs,ones(1,NF),[-1 -1],ndof,1);
else
    F = sparse(loaddofs,ones(1,NF),-1*ones(1,NF),ndof,1);
end
U = zeros(ndof,1);

% Preparation for filter
[H,Hs] = filter2d(rmin,nelx,nely);

% Initialization design cycle
x = repmat(0.5,nele,nAng);  xPhys = x;
change = 20; loop = 0; Pind = 1;
% MMA
m = 1;  n = nele*nAng;
xold1 = zeros(n,1);  xold2 = xold1;
xmin  = max(zeros(n,1),x(:)-move);
xmax  = min(ones(n,1),x(:)+move);
low   = xmin;  upp   = xmax;
a0 = 1;  a = zeros(m,1);  c = 10000*ones(m,1);  d = zeros(m,1);
while change > 0.05 
    loop = loop + 1;
    if(loop>1000);break;end
    penal = penallist(Pind);
    w = (1-epsilon)*(xPhys.^penal);
    w_ind = [1:nAng];
    for k = 1:nAng-1
        k_w = 1-xPhys.^penal;
        w_ind = [w_ind(end),w_ind(1:end-1)];
        w = w.*k_w(:,w_ind);
    end
    % Element stiffness matrix interpolation
    sK = zeros(64*nele,1);
    for i = 1:nele
        tempK = zeros(8,8);
        for j = 1:nAng
            tempK = tempK + (epsilon+w(i,j))*cand_Ke{j,1};
        end
        sK((i-1)*64+1:i*64,1) = sK((i-1)*64+1:i*64,1)+tempK(:);
    end
    % FE-ANALYSIS
    K = sparse(iK,jK,sK); K = (K+K')/2;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    dwdxp = ((1-epsilon)*penal)*xPhys.^(penal-1);
    w_ind = [1:nAng];
    for k = 1:nAng-1
        dwdxp_ = 1-xPhys.^penal;
        w_ind = [w_ind(end),w_ind(1:end-1)];
        dwdxp = dwdxp.*dwdxp_(:,w_ind);
    end
    ce = zeros(nele,nAng);
    for i = 1:nAng
        ce(:,i) = -(1-epsilon)*sum((U(edofMat)*cand_Ke{i,1}).*U(edofMat),2);
    end
    if caseind ==3
        obj(loop) = 2*F'*U;
    else
        obj(loop) = F'*U;
    end
    dcdx = zeros(nele,nAng);
    dvdx = ones(nele,nAng);
    for i = 1:nAng
        dcdxp = ce(:,i).*dwdxp(:,i);
        dcdx(:,i) = H*(dcdxp./Hs);
        dvdx(:,i) = H*((ones(nele,1))./Hs);
    end
    % MMA
    xval = x(:);
    xmin  = max(zeros(n,1),xval-move);
    xmax  = min(ones(n,1),xval+move);
    f0val = obj(loop);
    df0dx = dcdx(:);
    fval = zeros(m,1);
    dfdx = zeros(nAng*nele,m);
    fval(1,1) = sum(xPhys(:))-volfrac*nele;
    dfdx(:,1) = repmat(H*((ones(nele,1)./Hs)),nAng,1);
%     for i = 1:nAng
%         maxvol = 1e-3;
%         fval(i+1,1) = maxvol-sum(xPhys(:,i));
%         dfdx(nele*(i-1)+1:nele*i,i+1) = H*((repmat(-1,nele,1))./Hs);
%     end
    dfdx = dfdx';
    [xmma,~,~,~,~,~,~,~,~,low1,upp1] = mmasub(m,n,loop,xval,xmin,xmax, ...
        xold1,xold2,f0val,df0dx,fval,dfdx,low,upp,a0,a,c,d);
    xold2 = xold1;  xold1 = xval;  low = low1;     upp = upp1;
    x = reshape(xmma,nele,nAng);
    for i = 1:nAng
        xPhys(:,i) = (H*x(:,i))./Hs;
    end
    threshold = 1e-6;
    for i = 1:nele
        if(length(find(xPhys(i,:)>0.95))>1)
            id = find(xPhys(i,:)>0.95);
            mid = setdiff([1:nAng]',id);
            xPhys(i,id) = mean(xPhys(i,id))/nAng;
            xPhys(i,mid) = mean(xPhys(i,mid))/nAng;
        end
    end
    xPhys(find(xPhys<threshold)) = 0;
    xPhys(find(xPhys>(1-threshold))) = 1;
    vol_sum = sum(xPhys(:))/nele;
    if loop>5
        change = var(obj(loop-4:loop));
    end
    fprintf('It.:%-4i   Obj.:%-.4e   vol_sum.:%-.4e   rmin.:%-.4e   ch1.:%-.4e \n', ...
        loop,obj(loop),vol_sum,rmin,change);
    volmat = zeros(nAng,1);
    for i = 1:nAng; volmat(i) = sum(xPhys(:,i))/nele; end

    % display result
    resultShow(loop,showgap,angle,nelx,nely,xPhys,obj,volmat,vol_sum,caseind,0.95);
    %%
    if  loop==5 || loop == 28 || loop == 56 || rem(loop,20)==0 
%         picturename = stecat(DMO',num2str(loop),'.fig')
        savefig(figure(1),['./FIG/case5-2/DMO_',num2str(loop),'.fig']);
        saveas(figure(1),['./FIG/case5-2/DMO_',num2str(loop),'.png']);
        saveas(figure(9),['./FIG/case5-2/DMO_all_',num2str(loop),'.png']);
    end
    %%

    % Para update
    if(rem(loop,25)==0); Pind = min(Pind+1,length(penallist)); end
    if(rem(loop,25)==0)
        oldrmin = rmin;  rmin = max(oldrmin-1,1.5);
        if(oldrmin~=rmin); [H,Hs] = filter2d(rmin,nelx,nely); end
    end
    
    %%
    loop_global = loop_global+1;
    [nCon1,~] = conCri(0.95,xPhys);  %[0.95,0.99]
    convergenceRate1 = nCon1/nele;
    convergenceRate_all = [convergenceRate_all convergenceRate1];
    if (convergenceRate1>0.985)
        break;
    end
    unconv = []; 
    for i = 1:nele
        if length(find(xPhys(i,:)>0.05 & xPhys(i,:)<0.95))>=2
            unconv = [unconv i];
        end
    end
    singleangleRate = 1-length(unconv)/nele;
    singleangleRate_all = [singleangleRate_all singleangleRate];
    figure(5)
    plot(1:loop_global,convergenceRate_all,1:loop_global,singleangleRate_all,'r');
    
    %%
    Map = make_bitmap_unconvergent(angle,nelx,nely,0.95,xPhys);
    figure(8)
    if caseind==3
        image([flip(1-Map ,2), 1-Map]); axis image off; drawnow;
    else
        image(1-Map); axis image off; drawnow;
    end
%     image( [1-Map]); axis image off; drawnow;
    if loop==5 || loop == 28 || loop == 56 || rem(loop,20)==0
        savefig(figure(8),['./FIG/case5-2/DMO_unconverged_',num2str(loop),'.fig']);
        saveas(figure(8),['./FIG/case6-2singleload/DMO_unconverged_',num2str(loop),'.png']);
    end
end
model.D0 = D0;
model.cand_Ke = cand_Ke;
model.iK = iK;
model.jK = jK;
% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
model.F = F;
model.freedofs = freedofs;
model.edofMat = edofMat;
end

