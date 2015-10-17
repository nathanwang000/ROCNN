function [acc, loss] = pipeline(params)
train_fn = @train;%@train_stochastic;
fn_loss = @loss_crossentropy;
addpath('layers');
debug = false;
validation = true;

% grid search parameters
lrs = [0.01];%[0.01 0.1];%[0.1, 0.2]; next to try: [0.0001 0.001 0.1] % for 2 layers 0.01 is fine
batch_sizes = [200];%[50,100];%[10,50,100];
rot_priors = [0.00005]; %[0.001 10];%[0.01,0.1,1]; next to try: [0.0001
                  %0.001 10 100]
wds = [0.0005]; %[0.0005,0.005,0.05,0.5];
                  %filter_sizes = [5,10,15,20,25]; % see gen_model
                  %n_filters = [20,50,80];
                  %pool_sizes = [2,3,4];
mu = 0.5;
% destroy momentum matrix
if exist('momentum.mat', 'file') delete 'momentum.mat'; end

% grid search
for lr=lrs
    for bs=batch_sizes
        for rp=rot_priors
            for wd=wds

%save_file = sprintf('model_%0.4d_%0.4d_%0.4d_%0.4d.mat', lr, bs, rp, wd);
%save_file = sprintf('model_2layer.mat');
save_file = 'model.mat'
if nargin < 1
    % sanity check: can overfit a small amount of data
    if debug params = struct('lr', 0.01, 'wd', 0.0000, 'df', 1.0, 'mu', 0, 'rot_prior',0.00, 'batch_size', 100, 'save_file', save_file, 'fn_loss', fn_loss);
    elseif validation params = struct('lr', lr, 'wd', wd, 'df', 1, 'mu', mu, 'rot_prior',rp, 'batch_size', bs, 'save_file', save_file, 'fn_loss', fn_loss);
    % real params
    else params = struct('lr', 0.01, 'wd', 0.0005, 'df', 1, 'mu', 0.98, 'rot_prior',0.1,'save_file', save_file, 'fn_loss', fn_loss); end
end

if ~exist(save_file, 'file') fprintf('gen new model\n'); model = gen_model;
else fprintf('use existing model\n'); 
    load(save_file); 
    model = model_info.model; 
end

%% load dataset
load mnist_rot.mat
% load mnist_rot_ones_threes.mat
% load small_mnist_rot_ones_threes.mat
% load 'small_mnist.mat' % ones vs rotated ones
% load 'mnist_ones_modified.mat' % ones vs rotated ones + other digits
% load 'one_vs_ten.mat'
% load 'small_one_vs_ten.mat'
% load 'small_mnist_ones.mat'
% load 'one_unbalanced_orientation.mat'
%% load data
train_data = data.train_data;
test_data = data.test_data;
val_data = data.val_data;
train_label = data.train_label;
test_label = data.test_label;
val_label = data.val_label;

% pos=train_data(:,:,:,train_label == 1);
% neg=train_data(:,:,:,train_label == 2);
% m1 = mean(pos,4); m2 = mean(neg,4);
% model.layers(2).params.W(1,:) = m1(:);
% model.layers(2).params.W(2,:) = m2(:);
%% resize data
% data_size = 10;
% new_train_data = zeros([data_size, data_size, 1, size(train_data,4)]);
% new_test_data = zeros([data_size, data_size, 1, size(test_data,4)]);
% new_val_data = zeros([data_size, data_size, 1, size(val_data,4)]);
% for i=1:size(train_data,4) new_train_data(:,:,1,i) = imresize(train_data(:,:,1,i),[data_size data_size]); end
% for i=1:size(test_data,4) new_test_data(:,:,1,i) = imresize(test_data(:,:,1,i),[data_size data_size]); end
% for i=1:size(val_data,4) new_val_data(:,:,1,i) = imresize(val_data(:,:,1,i),[data_size data_size]); end
% train_data = new_train_data;
% test_data = new_test_data;
% val_data = new_val_data;
% data = struct('train_label',train_label, 'train_data', train_data, 'test_label', test_label, 'test_data', test_data, 'val_label',val_label, 'val_data',val_data);
% save('one_unbalanced_orientation.mat','data');

%% use a really small dataset of size 400
% data_size = 400;
% new_train_data = train_data(:,:,:,1:data_size);
% new_val_data = val_data(:,:,:,1:data_size);
% new_train_label = train_label(1:data_size);
% new_val_label = val_label(1:data_size);
% step = floor(data_size/10);
% assert(step == data_size/10, 'check_dimension');
% for i=1:2
%     data_i = train_data(:,:,1,train_label==i);
%     val_data_i = val_data(:,:,1,val_label==i);
%     new_train_data(:,:,1,(i-1)*step+1:i*step) = data_i(:,:,1,1:step);
%     new_val_data(:,:,1,(i-1)*step+1:i*step) = val_data_i(:,:,1,1:step);
%     new_train_label((i-1)*step+1:i*step) = i;
%     new_val_label((i-1)*step+1:i*step) = i;
% end
% train_data = new_train_data;
% val_data = new_val_data;
% train_label = new_train_label;
% val_label = new_val_label;

plot = false;
if plot
    figure;
    for i = 1:100
        subplot(10,10,i);
        imshow(train_data(:,:,1,i));
        title(sprintf('%d',train_label(i)))
    end
    % title('train data plot');    
    figure;
    for i = 1:100
        subplot(10,10,i);
        imshow(val_data(:,:,1,i));
        title(sprintf('%d',val_label(i)))
    end
    %title('val data plot');
end

% stupid = true;
% if stupid
%    for i=1:10
%       model.layers(1).params.W(:,:,1,1,i) = train_data(:,:,1,(i-1)*step+1);       
%    end
% end

if debug [model, loss] = train_fn(model, train_data(:,:,:,1:100), train_label(1:100), params, 100);
elseif validation 
    try
        [model, loss] = train_fn(model,train_data,train_label,params,5);
    catch
        fprintf('%s is not valid\n',save_file);
        continue;
    end
else [model, loss] = train_fn(model,[train_data; val_data],[train_label; val_label],params,10); end

acc = struct('tr_acc', [], 'val_acc', [], 'te_acc', []);
% output test result
[acc.tr_acc, ~] = test_conv(model, train_data, train_label);
fprintf('train accuracy is %.5f\n', acc.tr_acc);
[acc.val_acc, ~] = test_conv(model, val_data, val_label);
fprintf('validation accuracy is %.5f\n', acc.val_acc);
% [acc.te_acc, ~] = test_conv(model, test_data, test_label);
% fprintf('test accuracy is %.5f\n', acc.te_acc);
if exist(save_file, 'file')
    loss.tr_loss = cat(1,model_info.loss.tr_loss,loss.tr_loss);
    loss.te_loss = cat(1,model_info.loss.te_loss,loss.te_loss);
end
model_info = struct('model', model, 'loss', loss, 'val_acc', acc.val_acc, 'tr_acc', acc.tr_acc);
save(save_file, 'model_info');
        end 
    end
end
end