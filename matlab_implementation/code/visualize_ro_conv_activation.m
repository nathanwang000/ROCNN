function [activations, index] = visualize_ro_conv_activation(train_data, filter)
% train_data is of the shape width, height, 1, image_num
% filter is 2d, now it should have same width and height as the data
% assum
assert(size(filter,1) == size(train_data,1) && size(filter,2) == size(train_data,2) && size(train_data,3) == 1, 'incorrect dimension\n');
activations = zeros([1 size(train_data,4)]);
for i=1:size(train_data,4)
    im = train_data(:,:,1,i);
    activations(i) = sum(sum(flip(flip(filter.*im,1),2)));
end
[activations, index] = sort(activations,'descend');

data_size = 10;
figure; 
for i=1:data_size
    subplot(1,data_size,i);
    imshow(train_data(:,:,1,index(i)));
end
end

function ro_conv_toy_visualization()
    % visualization for the 1 layer ro_conv net
    load model.mat;
    model = model_info.model;
    num_angles = 8;
    % load dataset
    % load_MNIST_rot_data;
    % load 'small_mnist.mat' % ones vs rotated ones
    % load 'mnist_ones_modified.mat' % ones vs rotated ones + other digits
    % load 'one_vs_ten.mat'
    load 'small_one_vs_ten.mat'
    % load 'small_mnist_ones.mat'
    % load data
    train_data = data.train_data;
    test_data = data.test_data;
    val_data = data.val_data;
    train_label = data.train_label;
    test_label = data.test_label;
    val_label = data.val_label;

    
    angles = 0:2*pi/num_angles:(2*pi-0.001); % -0.001 for not double counting 360
    for i=1:2
        i=1
        filter = model.layers(1).params.W(:,:,1,1,i);
        for j=1:num_angles
            j=4
            rot_filter=circular_rotate(filter,angles(j));
            activations = visualize_ro_conv_activation(train_data(:,:,:,1:10), rot_filter)
        end        
    end
    
    
end