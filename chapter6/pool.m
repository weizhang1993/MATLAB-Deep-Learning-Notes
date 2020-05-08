function y = pool( x )
%POOL �˴���ʾ�йش˺�����ժҪ
%
% 2x2 mean pooling
%
[xrow, xcol, numFilters] = size(x)

y = zeros(xrow/2, xcol/2, numFilters);
for k = 1:numFilters
    filter = ones(2)/2*2; % for mean
%     C=conv2(A,B,shape);        %����˲�
%     A:����ͼ��B:�����
%     ��������ͼ��A��СΪma x na�������B��СΪmb x nb����
%     ��shape=fullʱ������ȫ����ά��������������C�Ĵ�СΪ��ma+mb-1��x��na+nb-1��
%     shape=sameʱ��������Aͬ����С�ľ�����Ĳ���
%     shape=validʱ�������Ǳ߽粹�㣬��ֻҪ�б߽粹�������������Ķ���ȥ������C�Ĵ�СΪ��ma-mb+1��x��na-nb+1��
    image = conv2(x(:, :, k), filter, 'valid')
%    pool��ʵ���Ǿ����һ�ֲ����������������3*3������ת����2*2��ȥ���м��ƶ�һ������Щ���
    y(:, :, k) = image(1:2:end, 1:2:end); % ���ĸ����ϵ��м�ĵ�ȥ��
end
end

