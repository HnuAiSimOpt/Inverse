% MAKE BITMAP IMAGE OF MULTIPHASE TOPOLOGY
function Map = make_bitmap_unconvergent(angle,nx,ny,epsilon,x)
% [1 0 0] 红色 <------对应------> 青蓝
% [1 1 0] 黄色 <------对应------> 蓝色
% [0 0 1] 蓝色 <------对应------> 黄色
% [0 1 1] 青蓝 <------对应------> 红色
% [1 0 1] 品红 <------对应------> 绿色
% [0 1 0] 绿色 <------对应------> 品红
% [0 0 0] 黑色 <------对应------> 白色
% [1 1 1] 白色 <------对应------> 黑色
color = [1 1 1;0 0 0;1-242/255 1-115/255 1-41/255;];
p = length(angle);
Map = zeros(nx*ny,3);
[nele,nP] = size(x);
for i = 1:nele
    sum2 = 0;
%     [val,~] = max(x(i,:));
    for j = 1:nP
        sum2 = sum2+x(i,j)^2;
    end
    if (length(find(x(i,:)<0.05))==p)
       Map(i,1:3) = color(2,:);
    else
        if(max(x(i,:))>epsilon*sqrt(sum2))
            Map(i,1:3) = color(1,:);
        else
            Map(i,1:3) = color(3,:);
        end
    end 
end
Map = reshape(Map,ny,nx,3);