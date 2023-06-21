% display result
function resultShow(loop,gap,angle,nelx,nely,xPhys,obj,volmat,vol_sum,caseind,epsilon)
if(rem(loop,gap)==0)
    [Map,name] = make_bitmap (angle,nelx,nely,xPhys);
    figure(1)
    switch caseind
        case 1  % 悬臂梁
            image(1-Map); axis image off; drawnow;
        case 2  % 简支梁
            image([flip(1-Map ,2) 1-Map]); axis image off; drawnow;
        case 3  % 三点载荷简支梁
            image([flip(1-Map ,2) 1-Map]); axis image off; drawnow;
        case 4
            image(1-Map); axis image off; drawnow;
        otherwise
            disp('Case identifier error');
    end
%     title(name);
    figure(9)
    [Map_fig9,~] = make_bitmap_fig9 (angle,nelx,nely,xPhys,epsilon);
    switch caseind
        case 1  % 悬臂梁
            image(1-Map_fig9); axis image off; drawnow;
        case 2  % 简支梁
            image([flip(1-Map_fig9 ,2) 1-Map_fig9]); axis image off; drawnow;
        case 3  % 三点载荷简支梁
            image([flip(1-Map_fig9 ,2) 1-Map_fig9]); axis image off; drawnow;
        case 4
            image(1-Map_fig9); axis image off; drawnow;
        otherwise
            disp('Case identifier error');
    end
    
    %%
%     if (rem(loop,30)==0)
%         saveas(figure(1),['E:\all_code\code\先DMO后CFAO\先DMO后CFAO\FIG\',num2str(loop),'.jpg']);
%     end

    %%
    figure(2)
    plot(obj,'r-o');
    
    figure(3)
    bar([volmat;vol_sum]);
    grid on;
    xlabel('材料种类');  ylabel('体积分数');
    Nangle = length(angle);
    switch Nangle
        case 2
            set(gca,'XTickLabel',{'mat1','mat2','all'});
            title(['VM1:',num2str(volmat(1),'%-.6f'),'    VM2:',num2str(volmat(2),'%-.6f')]);
        case 3
            set(gca,'XTickLabel',{'mat1','mat2','mat3','all'});
            title(['VM1:',num2str(volmat(1),'%-.6f'),'    VM2:',num2str(volmat(2),'%-.6f'), ...
                '    VM3:',num2str(volmat(3))]);
        case 4
            set(gca,'XTickLabel',{'mat1','mat2','mat3','mat4','all'});
            title(['VM1:',num2str(volmat(1),'%-.6f'),'    VM2:',num2str(volmat(2),'%-.6f'), ...
                '    VM3:',num2str(volmat(3),'%-.6f'),'    VM4:',num2str(volmat(4),'%-.6f')]);
        case 5
            set(gca,'XTickLabel',{'mat1','mat2','mat3','mat4','mat5','all'});
            title(['VM1:',num2str(volmat(1),'%-.6f'),'    VM2:',num2str(volmat(2),'%-.6f'), ...
                '    VM3:',num2str(volmat(3),'%-.6f'),'    VM4:',num2str(volmat(4),'%-.6f'), ...
                '    VM5:',num2str(volmat(5),'%-.6f')]);
        case 6
            set(gca,'XTickLabel',{'mat1','mat2','mat3','mat4','mat5','mat6','all'});
            title(['VM1:',num2str(volmat(1),'%-.6f'),'    VM2:',num2str(volmat(2),'%-.6f'), ...
                '    VM3:',num2str(volmat(3),'%-.6f'),'    VM4:',num2str(volmat(4),'%-.6f'), ...
                '    VM5:',num2str(volmat(5),'%-.6f'),'    VM6:',num2str(volmat(6),'%-.6f')]);
        otherwise
    end
end
end