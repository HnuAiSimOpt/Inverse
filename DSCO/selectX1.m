function [x,T] = selectX1(alpha,angle)
[nele,~] = size(alpha);
n = length(angle);
x = zeros(nele,1);
T = zeros(nele,1);
for i = 1:nele
    [value,ind] = max(alpha(i,:));
    if value < 0.05
        T(i) = 0;
        x(i) = 0;
    else
        T(i) = angle(ind);
        x(i) = value;
    end
end