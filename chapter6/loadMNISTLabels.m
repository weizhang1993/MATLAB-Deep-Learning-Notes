function [labels] = loadMNISTLabels(filename)
% loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
% the labels for the MNIST images
   % after compress the fliter is binary
    fp = fopen(filename, 'rb');
    %判断表达是否成立assert(expression) evaluates expression and, if it is false, generates an exception.
    assert(fp ~= -1, ['Could not open ', filename, '']);
    %A = fread(fileID,sizeA,precision,skip,machinefmt) 
    % additionally specifies the order for reading bytes or bits in the file. The sizeA and skip arguments are optional.
    magic = fread(fp, 1, 'int32', 0, 'ieee-be');
    assert(magic == 2049, ['Bad magic number in ', filename, '']);
    numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
    labels = fread(fp, inf, 'unsigned char');
    assert(size(labels,1) == numLabels, 'Mismatch in label count');
    fclose(fp);
end