function y = conv(x, W)

[wrow, wcol, numFilters] = size(W); 
[xrow, xcol, ~] = size(x);                

yrow = xrow - wrow + 1;
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numFilters);

for k = 1 : numFilters
    filter = W(:, :, k);
    filter = rot90(squeeze(filter), 2); 
    % squeeze:��ʾ����һ�����飬��Ԫ������������ A ��ͬ����ɾ���˳���Ϊ 1 ��ά��
    % ���磬��� A �� 3��1��2 ���飬�� squeeze(A)���� 3��2 ����
    % B=rot90(A,k)��ʾ������A��ʱ����ת(90��k)���Ժ󷵻�B��kȡ����ʱ��ʾ˳ʱ����ת
    y(:, :, k) = conv2(x, filter, 'valid');
end

end
%        x: ����ͼ��W: �����kernel, ����˲���filter
%        ��������ͼ��x��СΪma x na�������W��СΪmb x nb����
%        ��shape=fullʱ������ȫ����ά��������������C�Ĵ�СΪ��ma+mb-1��x��na+nb-1��
%       shape=sameʱ��������Aͬ����С�ľ�����Ĳ���
%        shape=validʱ�������Ǳ߽粹�㣬��ֻҪ�б߽粹�������������Ķ���ȥ������C�Ĵ�СΪ��ma-mb+1��x��na-nb+1��
% x = [1, 1, 1, 3; 4, 6, 4, 8; 30, 0, 1, 5; 0, 2, 2, 4]; W = [1, 0; 0, 1];
% y = conv(x, W)


