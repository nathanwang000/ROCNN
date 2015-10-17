% other datasets: createNewDataset.m, createDiscriminative dataset
MNIST_loaded = 0;
load_MNIST_data;
normal_ones = train_data(:,:,:,train_label==1); % 6000 positive examples
normal_ones = normal_ones(:,:,:,1:6000);
rotated_ones = zeros(size(normal_ones));

for i=1:size(normal_ones,4)
    rotated_ones(:,:,:,i) = imrotate(normal_ones(:,:,:,i),90);
end

neg = train_data(:,:,:,train_label==3); % pick an easy class
neg = neg(:,:,:,1:6000);

% randomize ones
normal_ones = normal_ones(:,:,:,randperm(size(normal_ones,4)));
rotated_ones = rotated_ones(:,:,:,randperm(size(rotated_ones,4)));
neg = neg(:,:,:,randperm(size(neg,4)));
% in total 12000, split into train, valid, test: 8000:2000:2000
train_data = cat(4,normal_ones(:,:,:,1:4000),neg(:,:,:,1:4000));
train_label = ones([8000,1]); train_label(4001:end) = 2; % 2 to make sure cross entropy works correctly
val_data = cat(4,rotated_ones(:,:,:,4001:5000),neg(:,:,:,4001:5000));
val_label = ones([2000,1]); val_label(1001:end) = 2;
test_data = cat(4,rotated_ones(:,:,:,5001:6000),neg(:,:,:,5001:6000));
test_label = ones([2000,1]); test_label(1001:end) = 2;

tr_sample = randperm(size(train_data,4));
train_data = train_data(:,:,:,tr_sample);
train_label = train_label(tr_sample);

val_sample = randperm(size(val_data,4));
val_data = val_data(:,:,:,val_sample);
val_label = val_label(val_sample);

te_sample = randperm(size(test_data,4));
test_data = test_data(:,:,:,te_sample);
test_label = test_label(te_sample);

figure
for i=1:100
    subplot(10,10,i);
    imshow(train_data(:,:,1,i));
    title(val_label(i));
end

figure
for i=1:100
    subplot(10,10,i);
    imshow(val_data(:,:,1,i));
    title(val_label(i));
end

data = struct('train_label',train_label, 'train_data', train_data, 'test_label', test_label, 'test_data', test_data, 'val_label',val_label, 'val_data',val_data);
data_normalization(data);
save('one_vs_ten.mat','data')
