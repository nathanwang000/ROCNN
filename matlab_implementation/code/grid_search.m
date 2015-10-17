wd_list = [0.0001, 0.0005, 0.001, 0.05, 0.01];
te_losses = zeros(size(wd_list));
te_acc = zeros(size(wd_list));
tr_losses = zeros(size(wd_list));
tr_acc = zeros(size(wd_list));
index = 1;
for wd=wd_list
    delete('model.mat');
    params = struct('lr', 1, 'wd', wd, 'df', 1.0);
    [acc, loss] = pipeline(params);
    tr_losses(index) = loss.tr_loss(end) / 128; % assume the batch is 128
    te_losses(index) = loss.te_loss(end) / 10000; % assume test size 10000
    tr_acc(index) = acc.tr_acc;
    te_acc(index) = acc.te_acc;
    index = index + 1;
end
    