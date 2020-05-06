function [ym] = dropout(y, ratio)
[m, n] = size(y);
ym = zeros(m, n);
%计算不为零的节点数
num = round(m * n * (1 - ratio)); % round([0.49, 0.5, 0.51]) =  0     1     1
%在m*n的范围内产生四个唯一整数（没有重复元素）
idx = randperm(m * n, num); % randperm(10, 5) =  3     2    10     9     5
%idx代表选取的位置s
ym(idx) = 1 / (1 - ratio);

end