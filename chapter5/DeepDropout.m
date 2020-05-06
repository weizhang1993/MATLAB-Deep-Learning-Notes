function [W1, W2, W3, W4 ] = DeepDropout( W1, W2, W3, W4, X, D)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
alpha = 0.01;

N = 5;
for k = 1:N
    x = reshape(X(:,:,k), 25, 1);
    v1 = W1*x;
    y1 = sigmoid(v1);
    y1 = y1.*dropout(y1,0.2);
    
    v2 = W2*y1;
    y2 = sigmoid(v2);
    y2 = y2.*dropout(y2,0.2);
    
    v3 = W3*y2;
    y3 = sigmoid(v3);
    y3 = y3.*dropout(y3,0.2);
    
    v = W4*y3;
    y = softmax(v);
    
    d = D(k,:)';
    e = d -y;
    delta = e;
    
    e3 = W4'*delta;
    delta3 = y3.*(1-y3).*e3;
    
    e2 = W3'*delta3;
    delta2 = y2.*(1-y2).*e2;
    
    e1 = W2'*delta2;
    delta1 = y1.*(1-y1).*e1;
    
    dW4 = alpha*delta*y3';
    W4 = W4 + dW4;

    dW3 = alpha*delta3*y2';
    W3 = W3 + dW3;
    
    dW2 = alpha*delta2*y1';
    W2 = W2 + dW2;

    dW1 = alpha*delta1*x';
    W1 = W1 + dW1;
end

end

