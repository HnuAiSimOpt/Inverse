clear all
clc
close all;
addpath('./show')
comp = [];
%%
global convergenceRate_all singleangleRate_all loop_global caseind
convergenceRate_all = []; singleangleRate_all =[]; loop_global=0;
%%
addpath('./DMO')
nelx = 60; nely = 40; volfrac = 0.5; rmin = 1.5; caseind = 3;
[xPhys,model,angle,objDMO] = DMO_nMat(nelx,nely,volfrac,rmin,caseind);
comp = [comp;objDMO(:)];
objDMO = objDMO';
figure(4)
plot(comp);
print(figure(1),'-dpng','-r300','./FIG/DMO.png');
savefig(figure(1),'./FIG/case5-2/DMO.fig');
saveas(figure(9),['./FIG/case5-2/DMO_all','.png']);
addpath('./ALterPhaseTO')
% xPhys = xPhys./max(1e-9,sum(xPhys,2));
tol_out = 1e-2; 
% tol_out = 0.5;
iter_max_in = 40;  iter_max_out = 20;  penal = 3; rmin = 1.5;
[alpha,objBi] = AlterPhasetop(nelx,nely,xPhys,tol_out,iter_max_in,iter_max_out,penal,rmin,volfrac,model,angle,caseind);
comp = [comp;objBi(:)];
figure(4)
plot(comp);
print(figure(1),'-dpng','-r300','./FIG/Bi.png');
savefig(figure(1),'./FIG/case5-2/Bi.fig');
saveas(figure(9),['./FIG/case5-2/Bi_all','.png']);
convergenceRate_all = convergenceRate_all';
addpath('./CFAO')
[x,T] = selectX(alpha,angle);
save all;
rmin = 1.5;
objCFAO = fiberTOv(nelx,nely,x,T,rmin,penal,volfrac,model,caseind);
comp = [comp;objCFAO(:)];
figure(4)
plot(comp);
print(figure(1),'-dpng','-r300','./FIG/CFAO.png');
savefig(figure(1),'./FIG/CFAO.fig');

print(figure(4),'-dpng','-r300','./FIG/comp.png');
savefig(figure(1),'./FIG/comp.fig');
save comp convergenceRate_all;