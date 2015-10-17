function [model, loss] = train(model,input,label,params,numIters)
addpath('layers');
% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% loss to use
if isfield(params,'fn_loss') fn_loss = params.fn_loss;
else fn_loss = @loss_crossentropy; end
% Learning rate
if isfield(params,'lr') lr = params.lr;
else lr = .01; end
% Weight decay
if isfield(params,'wd') wd = params.wd;
else wd = .0005; end
% rot prior
if isfield(params,'rot_prior') rp = params.rot_prior;
else rp = 0; end
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 10; end % was 128
% Decay factor (each epoch)
if isfield(params,'df') df = params.df;
else df = 0.95; end
% momentum
if isfield(params,'mu') mu = params.mu;
else mu = 0.9; end

% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);

input_size = size(input);
update_list = 1:batch_size:input_size(end);
% epoch size
epoch_size = length(update_list)-1;
tr_loss = ones([numIters*epoch_size 1]);
te_loss = ones([numIters*epoch_size 1]);
addpath pcode;% for debug
best_loss = 2.302; % the random loss for 10 classes -ln(0.1) = 2.302
index = 1;
for i = 1:numIters
	% Training code
    % minibatch training
    fprintf('epoch %d start\n', i)
    shuffled = randperm(input_size(end));
    for j= update_list(1:end)
        % select data
        if (j+batch_size-1) > input_size(end) continue; end
        selection = shuffled(j:j+batch_size-1); %randsample(input_size(end),batch_size);
        batch = input(:,:,:,selection); % or do it the minibatch way by epoch, also the input size might be a trouble
        batch_label = label(selection);
        % run inference
        [output, activations] = inference(model, batch);
        if (isequal(fn_loss,@loss_sigmoid))
            [~,prediction] = max([output; 1-output]);
        else
            [~, prediction] = max(output);
        end
        prediction = prediction';
        fprintf('current accuracy for index %d is %.5f\n', index, sum(prediction==batch_label)/length(batch_label));
        % calc loss
        [curr_loss, dv_output] = fn_loss(output, batch_label,[], true);
        tr_loss(index) = curr_loss/batch_size;
        fprintf('training loss for iteration %d index %d is %.5f\n', i, index, tr_loss(index));
        % calc grad
        [grad] = calc_gradient(model,batch,activations,dv_output);
        % update weight
        model = update_weights(model,grad,struct('learning_rate',lr,'weight_decay',wd, 'batch_size', batch_size, 'rot_prior', rp, 'mu', mu));
        % save best model so far
%         if tr_loss(index) < best_loss
%             save(save_file, 'model');
%             best_loss = tr_loss(index);
%         end
        % see test_result
%         global test_data;
%         global test_label;
%         [~, te_loss(index)] = test_conv(model, test_data, test_label);
        
        index = index + 1;
        
    end

    lr = df*lr;
end
loss = struct('tr_loss', tr_loss, 'te_loss', te_loss);
