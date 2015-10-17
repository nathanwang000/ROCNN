% positive are ones, negtive are rotated 90 degree ones plus other 
MNIST_loaded = 0;
load_MNIST_data;
pos = train_data(:,:,:,train_label==1); % 6742 positive examples
sample = randsample(size(train_data,4),size(pos,4)); % sample negtive examples
neg = zeros(size(pos));

assert(size(neg,4)==size(sample,1), sprintf('neg and sample size not compatible, %d vs %d', size(neg,4),size(sample)));
% generate neg examples
for i=1:size(neg,4)
   curr_data = train_data(:,:,:,sample(i));
   if train_label(sample(i)) == 1
       neg(:,:,:,i) = imrotate(curr_data,90); % neg are rotated 90
   else
       neg(:,:,:,i) = curr_data;
   end
end
% randomize pos and neg
pos = pos(:,:,:,randperm(size(pos,4)));
neg = neg(:,:,:,randperm(size(neg,4)));
% in total 13484, split into train, valid, test: 8000:2000:3484
train_data = cat(4,pos(:,:,:,1:4000),neg(:,:,:,1:4000));
train_label = ones([8000,1]); train_label(4001:end) = 2; % 2 to make sure cross entropy works correctly
val_data = cat(4,pos(:,:,:,4001:5000),neg(:,:,:,4001:5000));
val_label = ones([2000,1]); val_label(1001:end) = 2;
test_data = cat(4,pos(:,:,:,5001:6742),neg(:,:,:,5001:6742));
test_label = ones([3484,1]); test_label(1743:end) = 2;

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
    imshow(pos(:,:,1,i));
end

figure
for i=1:100
    subplot(10,10,i);
    imshow(neg(:,:,1,i));
end

data = struct('train_label',train_label, 'train_data', train_data, 'test_label', test_label, 'test_data', test_data, 'val_label',val_label, 'val_data',val_data);
data = data_normalization(data);
save('mnist_ones_modified.mat','data')

