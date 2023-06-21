function [x,T] = selectX(alpha,angle)
[nele,~] = size(alpha);
n = length(angle);
x = zeros(nele,1);
T = zeros(nele,1);
for i = 1:nele
    [value,ind] = max(alpha(i,:));
    if(ind<n+1)
        T(i) = angle(ind);
        x(i) = value;
    else
        T(i) = 0;
        x(i) = 0;
    end
end