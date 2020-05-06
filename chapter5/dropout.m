function [ym] = dropout(y, ratio)
[m, n] = size(y);
ym = zeros(m, n);
%���㲻Ϊ��Ľڵ���
num = round(m * n * (1 - ratio)); % round([0.49, 0.5, 0.51]) =  0     1     1
%��m*n�ķ�Χ�ڲ����ĸ�Ψһ������û���ظ�Ԫ�أ�
idx = randperm(m * n, num); % randperm(10, 5) =  3     2    10     9     5
%idx����ѡȡ��λ��s
ym(idx) = 1 / (1 - ratio);

end