function [alpha,comp] = AlterPhasetop(nelx,nely,alpha,tol_out,iter_max_in,iter_max_out,penal,rmin,volfrac,model,angle,caseind)
global loopAAP
[nele,nP] = size(alpha);
% 增加空相
void_phase = max(0,ones(nele,1)-sum(alpha,2));
nP = nP+1;
alpha = [alpha,void_phase];
A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
nu = 0.3; KE = 1e-9/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
model.cand_Ke{nP,1} = KE;
vollimit = sum(alpha,1)/nele;
change = 1; iter = 1;
comp = [];
loopAAP = 0;
while change > tol_out && iter < iter_max_out
    for a = 1:nP-1
        for b = 1:nP
            if(a~=b)
                temp_phase = alpha(:,[a,b]);
                act = ActiveEleSelec(temp_phase);
                [obj,alpha] = bi_top(a,b,nelx,nely,penal,alpha,rmin,volfrac,vollimit,iter_max_in,model,angle,caseind,act);
                comp = [comp;obj(:)];
                % a,b 材料编号;  nP 材料种类数;  volfrac 总体积分数;  vollimit 各材料体积分数
%                 [nCon,ind] = conCri(0.95,alpha(:,1:end-1));  %[0.95,0.99]
%                 convergenceRate = nCon/nele;
%                 if (convergenceRate>0.99)
%                     convergenceRate
%                     break;
%                 end
            end
        end
    end
    [nCon,ind] = conCri(0.95,alpha(:,1:end-1));  %[0.95,0.99]
    convergenceRate = nCon/nele;
    if (convergenceRate>0.99)
        convergenceRate
        break;
    end
    figure(4)
    plot(comp);
end