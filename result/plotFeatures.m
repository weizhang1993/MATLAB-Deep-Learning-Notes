clear

load('MnistConv.mat')

k = 2; % figure of the number 2
x = X( :, :, 2) % Input,           28x28
y1 = conv( x, W1 ); % Convolution,  20x20x20
y2 = ReLU(y1);
y3 = pool(y2); % Pool,         10x10x20
y4 = reshape( y3, [ ], 1);
v5 = W5 * y4;
y5 = ReLU(v5);
v = Wo * y5;
y = softmax(v);

figure
% 调用display_network函数，以网格的形式，随机显示多个扣取的图像块儿
display_network(x( : ));
title('Input Image')

convFilters = zeros( 9*9, 20 );
for i = 1 : 20
    filter = W1( :, :, i );
    convFilters( :, i ) = filter( : );
end
figure
display_network(convFilters);
title('Convolution Filters')

fList = zeros( 20 * 20, 20 );
for i = 1 : 20
    feature = y1( :, :, i );
    fList( :, i ) = feature( : );
end
figure
display_network(fList);
title('Features [Convolution]')

fList = zeros( 20 * 20, 20 );
for i = 1 : 20
    feature = y2( :, :, i );
    fList( :, i ) = feature( : );
end
figure
display_network(fList);
title('Features [Convolution + ReLU]')

fList = zeros( 10 * 10, 20 );
for i = 1 : 20
    feature = y3( :, :, i );
    fList( :, i ) = feature( : );
end
figure
display_network(fList);
title('Features [Convolution + ReLU + MeanPool]')






