% load MNIST_rot data
addpath ../data/mnist_rotation_new;

% global test_data;
% global test_label;
if ~exist('MNIST_rot_loaded') || ~MNIST_rot_loaded
	% Load training data
    train_data = fscanf(fopen('mnist_all_rotation_normalized_float_train_valid.amat','r'),'%f',[28*28+1 Inf])';
	train_label = train_data(:,end);
    train_data = train_data(:,1:(end-1));
    train_data = reshape(train_data',[28 28 1 size(train_data,1)]);
 	train_label(train_label == 0) = 10; % Remap 0 to 10    

    % split train (10000) val (2000)
    val_data = train_data(:,:,:,10001:end);
    val_label = train_label(10001:end);
    train_data = train_data(:,:,:,1:10000);
    train_label = train_label(1:10000);
    
    % Load testing data
    test_data = fscanf(fopen('mnist_all_rotation_normalized_float_test.amat','r'),'%f',[28*28+1 Inf])';
	test_label = test_data(:,end);
    test_data = test_data(:,1:(end-1));
    test_data = reshape(test_data',[28 28 1 size(test_data,1)]);
 	test_label(test_label == 0) = 10; % Remap 0 to 10    

    data = struct('train_label',train_label, 'train_data', train_data, 'test_label', test_label, 'test_data', test_data, 'val_label',val_label, 'val_data',val_data);
    save('mnist_rot.mat','data')
    MNIST_rot_loaded = true;
end

