function y = pool( x )
%POOL 此处显示有关此函数的摘要
%
% 2x2 mean pooling
%
[xrow, xcol, numFilters] = size(x)

y = zeros(xrow/2, xcol/2, numFilters);
for k = 1:numFilters
    filter = ones(2)/2*2; % for mean
%     C=conv2(A,B,shape);        %卷积滤波
%     A:输入图像，B:卷积核
%     假设输入图像A大小为ma x na，卷积核B大小为mb x nb，则
%     当shape=full时，返回全部二维卷积结果，即返回C的大小为（ma+mb-1）x（na+nb-1）
%     shape=same时，返回与A同样大小的卷积中心部分
%     shape=valid时，不考虑边界补零，即只要有边界补出的零参与运算的都舍去，返回C的大小为（ma-mb+1）x（na-nb+1）
    image = conv2(x(:, :, k), filter, 'valid')
%    pool其实就是卷积的一种操作，本来求出的是3*3，将其转换成2*2，去掉中间移动一步的那些结果
    y(:, :, k) = image(1:2:end, 1:2:end); % 把四个边上的中间的点去掉
end
end

