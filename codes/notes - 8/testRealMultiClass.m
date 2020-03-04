clear

testMultiClass; % train weight1, weight2

X = zeros(5, 5, 5);

X(:, :, 1) = [ 0 0 1 1 0;
                  0 0 1 1 0;
                  0 1 0 1 0;
                  0 0 0 1 0;
                  0 1 1 1 0
                 ];
X(:, :, 2) = [ 1 1 1 1 0;
                   0 0 0 0 1;
                   0 1 1 1 0;
                   1 0 0 0 1;
                   1 1 1 1 1
                 ];
X(:, :, 3) = [ 1 1 1 1 0;
                   0 0 0 0 1;
                   0 1 1 1 0;
                   1 0 0 0 1;
                   1 1 1 1 0
                 ];
X(:, :, 4) = [ 0 1 1 1 0;
                  0 1 0 0 0;
                  0 1 1 1 0;
                  0 0 0 1 0;
                  0 1 1 1 0
                 ];
X(:, :, 5) = [ 0 1 1 1 1;
                   0 1 0 0 0;
                   0 1 1 1 0;
                   0 0 0 1 0;
                   1 1 1 1 0
                 ];
             
N = 5; % inference
for k = 1:N
x = reshape(X(:, :, k), 25, 1);
v1 = weight1 * x;
y1 = sigmoid(v1);
v = weight2 * y1;
y = softmax(v)
end