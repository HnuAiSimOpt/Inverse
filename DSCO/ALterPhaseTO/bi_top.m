% MODIFIED BINARY-PHASE TOPOLOGY OPTIMIZATION SOLVER
function [obj,alpha] = bi_top(ph_a,ph_b,nelx,nely,penal,alpha_old,rmin,sumvol,vollimit,iter_max_in,model,angle,caseind,act)
global convergenceRate_all singleangleRate_all loop_global loopAAP
alpha = alpha_old;   [nele,nP] = size(alpha);
% 初始化过滤模型
[H,Hs] = filter2d(rmin,nelx,nely);
% PREPARE FINITE ELEMENT ANALYSIS
edofMat = model.edofMat;
iK = model.iK;
jK = model.jK;
% DEFINE LOADS AND SUPPORTS (HALF MBB-BEAM)
F = model.F;
freedofs = model.freedofs;
U = zeros(2*(nely+1)*(nelx+1),1);
% INNER ITERATIONS
loop = 0; move = 0.1;  epsilon = 1e-9;  change = 1;
while change > 1e-2 && loop < iter_max_in
    loop = loop + 1; loopAAP = loopAAP + 1;
    % FE-ANALYSIS
    sK = reshape(model.cand_Ke{1,1}(:)*((1-epsilon)*(alpha(:,1).^penal))',64*nele,1);
    for i = 2:nP
        sK = sK + reshape(model.cand_Ke{i,1}(:)*(epsilon+alpha(:,i).^penal)',64*nele,1);
    end
    K = sparse(iK,jK,sK); K = (K+K')/2;
    U(freedofs) = K(freedofs,freedofs)\F(freedofs);
    % OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS
    if caseind ==3
        obj(loop) = 2*F'*U;
    else
        obj(loop) = F'*U;
    end
    ce = sum((U(edofMat)*model.cand_Ke{ph_a,1}).*U(edofMat),2);
    dc = -(((1-epsilon)*penal)*alpha(:,ph_a).^(penal-1)).*ce;
    dv = ones(nele,1);
    % FILTERING OF SENSITIVITIES
    %dc = min(0,H*(dc./Hs));
    dc = min(0,H*(alpha(:,ph_a).*dc)./Hs./max(1e-3,alpha(:,ph_a)));
%     dv = H*(dv./Hs);
    % MMA
    r = ones(nele,1);
    for k = 1:nP
        if (k ~= ph_a) && (k ~= ph_b)
            r = r - alpha(:,k);
        end
    end
    xmin = max(0,alpha(act,ph_a)-move);
    xmax = min(r(act),alpha(act,ph_a)+move);
    l1 = 0; l2 = 1e9;
    alpha_a = alpha(:,ph_a);
    while (l2-l1)/(l1+l2) > 1e-3  && l2>1e-40
        lmid = 0.5*(l2+l1);
        alpha_a(act) = max(xmin,min(xmax,alpha(act,ph_a).*sqrt(-dc(act)./dv(act)/lmid)));
%         temp = (H*alpha_a)./Hs;
%         alpha(act,ph_a) = temp(act);
        alpha(act,ph_a) = alpha_a(act);
        if sum(alpha(:,ph_a)) > nele*vollimit(ph_a)
            l1 = lmid;
        else
            l2 = lmid;
        end
    end
    alpha(act,ph_b) = r(act)-alpha(act,ph_a);
    % change
    if loop>5
        change = var(obj(loop-4:loop));
    end
    % display result
    volmat = [sum(alpha,1)/nele]';
    if loopAAP==200
        aaaa=1;
        resultShow(loop,1,angle,nelx,nely,alpha(:,1:end-1),obj,volmat(1:end-1),sumvol,caseind,0.95);
        saveas(figure(9),['./FIG/case5-2/AAP_all_',num2str(loopAAP),'.png']);
    end
    %resultShow(loop,1,angle,nelx,nely,alpha,obj,volmat,sumvol,caseind);
    resultShow(loop,1,angle,nelx,nely,alpha(:,1:end-1),obj,volmat(1:end-1),sumvol,caseind,0.95);
    %%
    if loopAAP==60 || loopAAP == 180 || loopAAP == 227 || rem(loopAAP,20)==0
        savefig(figure(1),['./FIG/case5-2/AAP',num2str(loopAAP),'.fig']);
        saveas(figure(1),['./FIG/case5-2/AAP',num2str(loopAAP),'.png']);
        saveas(figure(9),['./FIG/case5-2/AAP_all_',num2str(loopAAP),'.png']);
    end
    %%
    fprintf('It.:%-4i   Phase.:%-2i--%-2i   Obj.:%-.4e \n',loop,ph_a,ph_b,obj(loop));
    % 参数更新
    if(rem(loop,25)==0)
        oldrmin = rmin;  rmin = max(oldrmin-1,1.5);
        if(oldrmin~=rmin); [H,Hs] = filter2d(rmin,nelx,nely); end
    end
    %%
    loop_global = loop_global+1;
    [nCon1,~] = conCri_AAP(0.95,alpha);  %[0.95,0.99]
    convergenceRate1 = nCon1/nele;
    convergenceRate_all = [convergenceRate_all convergenceRate1];
        %%
    Map = make_bitmap_unconvergent(angle,nelx,nely,0.95,alpha(:,1:end-1));
    figure(8)
    if caseind==3
        image([flip(1-Map ,2), 1-Map]); axis image off; drawnow;
    else
        image(1-Map); axis image off; drawnow;
    end
    if loopAAP==60 || loopAAP == 180 || loopAAP == 227 || rem(loopAAP,20)==0
        savefig(figure(8),['./FIG/case5-2/AAP_unconverged_',num2str(loopAAP),'.fig']);
        saveas(figure(8),['./FIG/case5-2/AAP_unconverged_',num2str(loopAAP),'.png']);
    end
%     unconv = []; 
%     for i = 1:nele
%         if length(find(alpha(i,:)>0.05 & alpha(i,:)<0.95))>=2
%             unconv = [unconv i];
%         end
%     end
%     singleangleRate = 1-length(unconv)/nele;
%     singleangleRate_all = [singleangleRate_all singleangleRate];
    figure(5)
    plot(1:loop_global,convergenceRate_all,'r');
%     plot(1:loop_global,convergenceRate_all,1:loop_global,singleangleRate_all,'r');
    %%
%     [nCon,~] = conCri(0.95,alpha(:,1:end-1));  %[0.95,0.99]
%     convergenceRate = nCon/nele;
%     if (convergenceRate>0.99)
%         convergenceRate
%         break;
%     end
    %%
%     Map = make_bitmap_unconvergent(angle,nelx,nely,0.95,alpha(:,1:end-1));
%     figure(7)
%     image([1-flip(Map ,2), 1-Map]); axis image off; drawnow;
end
end