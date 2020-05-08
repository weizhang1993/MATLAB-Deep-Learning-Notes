function [W1, W5, Wo] = MinstConv( W1, W5, Wo, X, D)
%MINSTCONV 此处显示有关此函数的摘要
%   此处显示详细说明
alpha = 0.01;
beta = 0.95;

momentum1 = zeros(size(W1));
momentum5 = zeros(size(W5));
momentumo = zeros(size(Wo));

N = length(D);
%输入是10000组数据，其中8000组训练，2000组测试
%然后8000组数据每一次训练的时候随机选取100组，循环80次
bsize = 100; 
blist = 1:bsize:(N-bsize+1);%代表的是blist= [ 1, 101, 201, 301, ..., 7801, 7901 ]
%One epoch loop
for batch = 1:length(blist)
    dW1 = zeros(size(W1));
    dW5 = zeros(size(W5));
    dWo = zeros(size(Wo));
    %Mini-batch loop
    begin = blist(batch);
    for k = begin:begin + bsize-1
        %Forward pass
        x = X(:, :, k);
        y1 = conv(x,W1);
        y2 = ReLU(y1);
        y3 = pool(y2);
        y4 = reshape(y3, [], 1);
        v5 = W5 * y4;
        y5 = ReLU(v5);
        v = Wo * y5;
        y = softmax(v);
       % One-hot encoding
       d = zeros(10,1);
       d(sub2ind(size(d), D(k), 1)) = 1;
       
       % Backpropagation
       e = d - y; %Output layer
       delta = e;
       
       e5 = Wo' * delta; % Hidden(ReLU) layer
       delta5 = y5>0 .*e5;
       
       e4 = W5' * delta5; % Pooling layer
       
       e3 = reshape(e4, size(y3));
       
       e2 = zeros(size(y2));
       W3 = ones(size(y2))/(2*2);
       for c = 1:20
           %张量计算，pool层是缩减了2倍，所以这个增加了2倍
          e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
       end
       
       delta2 = (y2>0) .* e2;    %ReLU layer
       
       delta1_x = zeros(size(W1));       % Convolutional layer
       for c = 1:20
            delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2),'valid');
       end
        
       dW1 = dW1+ delta1_x;
       dW5 = dW5 + delta5*y4';
       dWo = dWo + delta *y5';

    end
    % Update weights
    %
      dW1 = dW1 / bsize;
      dW5 = dW5 / bsize;
      dWo = dWo / bsize;
      momentum1 = alpha*dW1 + beta*momentum1;
      W1        = W1 + momentum1;
      momentum5 = alpha*dW5 + beta*momentum5;
      W5        = W5 + momentum5;
      momentumo = alpha*dWo + beta*momentumo;
      Wo        = Wo + momentumo;
    
    
end 
end

