function [y] = softmax( x )
%SOFTMAX �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
ex = exp(x);
y = ex / sum(ex);

end

