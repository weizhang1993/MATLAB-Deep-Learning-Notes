function [y] = softmax( x )
%SOFTMAX 此处显示有关此函数的摘要
%   此处显示详细说明
ex = exp(x);
y = ex / sum(ex);

end

