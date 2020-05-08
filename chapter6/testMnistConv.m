clear all
Images = loadMNISTImages('t10k-images-idx3-ubyte');
Images = reshape(Images, 28, 28, [ ]);
Labels = loadMNISTLabels('t10k-labels-idx1-ubyte');
Labels(Labels == 0) = 10; % 0 --> 10
rng(1);

% Learning
% 
W1 = 1e-2 * randn([9, 9, 20]);
W5 = (2 * rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
Wo = (2 * rand(10, 100) - 1) * sqrt(6) / sqrt(10 + 100);

X = Images(:, :, 1:8000);
D = Labels(1 : 8000);
for epoch = 1 : 3
    epoch
    [W1, W5, Wo] = MinstConv(W1, W5, Wo, X, D);
end

save('MnistConv.mat');

% Test
%
X = Images(:, :, 8001 : 10000);
D = Labels(8001 : 10000);

acc = 0;
N = length(D);
for k = 1:N
    x = X(:, :, k); % Input, 28x28
    
    y1 = conv(x, W1); % Convolution, 20x20x20
    y2 = ReLU(y1); %
    y3 = pool(y2); % Pool, 10x10x20
    y4 = reshape(y3, [ ], 1); % 2000
    v5 = W5 * y4; % ReLU, 360
    y5 = ReLU(v5); %
    v = Wo * y5; % Softmax, 10
    y = softmax(v); %
    %For this example, the code wants to work with the index of the maxmum value, not the value itself, so the minimum value that is returned is discarded and only the index is retained
    [~, i] = max(y);
    if i == D(k)
        acc = acc + 1;
    end
end
acc = acc / N;
fprintf('Accuracy is %f\n', acc);
