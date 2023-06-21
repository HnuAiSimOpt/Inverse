function [nCon,ind] = conCri(epsilon,x)
[nele,nP] = size(x);
nCon = 0;
ind = [];
for i = 1:nele
    sum2 = 0;
    for j = 1:nP
        sum2 = sum2+x(i,j)^2;
    end
    if(max(x(i,:))>epsilon*sqrt(sum2) || max(x(i,:))==0)
        nCon = nCon+1;
    else
        ind = [ind;i];
    end
end