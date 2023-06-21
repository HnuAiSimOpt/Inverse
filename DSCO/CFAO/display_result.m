function display_result(comp,den,T)
global caseind
% clf(figure(1));
clf(figure(2));
[nely,nelx] = size(den);

figure(2)
plot(comp,'.-'); xlabel('Iteration'); ylabel('objective function');

figure(6)
imagesc([1/2,nelx-1/2],[1/2,nely-1/2],1-den);colormap(gca,'gray'); axis equal; axis tight;
hold on
if caseind==3
    imagesc([-(nelx-1/2),-1/2],[1/2,nely-1/2],flip(1-den,2));colormap(gca,'gray'); axis equal; axis tight;
    hold on
end
[xcoor,ycoor] = meshgrid(1/2:nelx-1/2,1/2:nely-1/2);
x1 = xcoor - 0.4.*cosd(T);  x2 = xcoor + 0.4.*cosd(T);
y1 = ycoor + 0.4.*sind(T);  y2 = ycoor - 0.4.*sind(T);  % 因为matlab Y轴正方向图示向下
for i=1:nely
    for j=1:nelx
        if(den(i,j)>0.01)
            line ([x1(i,j) x2(i,j)],[y1(i,j) y2(i,j)],'Color','w','linewidth',1.5);
            hold on
            line ([-x2(i,j) -x1(i,j)],[y2(i,j) y1(i,j)],'Color','w','linewidth',1.5);
            hold on
        end
    end
end
if caseind==3
    axis equal; set(gca,'xLim',[-nelx nelx]); set(gca,'YLim',[0 nely]);
else
    axis equal; set(gca,'xLim',[0 nelx]); set(gca,'YLim',[0 nely]);
end
% axis equal; set(gca,'xLim',[0 nelx]); set(gca,'YLim',[0 nely]);
% axis equal; set(gca,'xLim',[-nelx nelx]); set(gca,'YLim',[0 nely]);
set( gca, 'xTick', [], 'YTick', [] );  
% set(gca,'Box','on');
box off;
axis off;
hold off;
figure(7)
set(gca,'YDir','reverse');
for i=1:nely
    for j=1:nelx
        if(den(i,j)>0.01)
            line ([x1(i,j) x2(i,j)],[y1(i,j) y2(i,j)],'Color','r','linewidth',1.4);
            hold on
            line ([-x2(i,j) -x1(i,j)],[y2(i,j) y1(i,j)],'Color','r','linewidth',1.4);
            hold on
        end
    end
end
if caseind==3
    axis equal; set(gca,'xLim',[-nelx nelx]); set(gca,'YLim',[0 nely]);
else
    axis equal; set(gca,'xLim',[0 nelx]); set(gca,'YLim',[0 nely]);
end
% axis equal; set(gca,'xLim',[0 nelx]); set(gca,'YLim',[0 nely]);
% axis equal; set(gca,'xLim',[-nelx nelx]); set(gca,'YLim',[0 nely]);
set( gca, 'xTick', [], 'YTick', [] );  
% set(gca,'Box','on');
box off;
axis off;
hold off;
savefig(figure(6),'./FIG/case5-2/结构纤维布局图.fig');
saveas(figure(6),'./FIG/case5-2/结构纤维布局图.png');
savefig(figure(7),'./FIG/case5-2/纤维图.fig');
saveas(figure(7),'./FIG/case5-2/纤维图.png');
end