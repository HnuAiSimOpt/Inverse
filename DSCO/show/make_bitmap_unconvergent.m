% MAKE BITMAP IMAGE OF MULTIPHASE TOPOLOGY
function Map = make_bitmap_unconvergent(angle,nx,ny,epsilon,x)
% [1 0 0] ��ɫ <------��Ӧ------> ����
% [1 1 0] ��ɫ <------��Ӧ------> ��ɫ
% [0 0 1] ��ɫ <------��Ӧ------> ��ɫ
% [0 1 1] ���� <------��Ӧ------> ��ɫ
% [1 0 1] Ʒ�� <------��Ӧ------> ��ɫ
% [0 1 0] ��ɫ <------��Ӧ------> Ʒ��
% [0 0 0] ��ɫ <------��Ӧ------> ��ɫ
% [1 1 1] ��ɫ <------��Ӧ------> ��ɫ
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