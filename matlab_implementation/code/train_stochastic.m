function [model, loss] = train_stochastic(model,input,label,params,numIters)
addpath('layers');
% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

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
else save_file = 'model_stochastic.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);

numIters = floor(size(input,4) / batch_size);

input_size = size(input);
tr_loss = ones([numIters 1]);
te_loss = ones([numIters 1]);
best_loss = 2.302; % the random loss for 10 classes -ln(0.1) = 2.302
for i = 1:numIters
    	% Training code
        % minibatch training
        fprintf('epoch %d start\n', i)
        % select data
        selection = randsample(input_size(end),batch_size);
        batch = input(:,:,:,selection); % or do it the minibatch way by epoch, also the input size might be a trouble
        batch_label = label(selection);
        % run inference
        [output, activations] = inference(model, batch);
        [~, prediction] = max(output);
        prediction = prediction';
        fprintf('current accuracy is %.5f\n', sum(prediction==batch_label)/length(batch_label));
        % calc loss
        [curr_loss, dv_output] = loss_crossentropy(output, batch_label,[], true);
        tr_loss(i) = curr_loss/batch_size;
        fprintf('training loss for iteration %d is %.5f\n', i, tr_loss(i));
        % calc grad
        [grad] = calc_gradient(model,batch,activations,dv_output);
        % update weight
        model = update_weights(model,grad,struct('learning_rate',lr,'weight_decay',wd, 'batch_size', batch_size, 'rot_prior', rp, 'mu', mu));
        % save best model so far
%         if tr_loss(i) < best_loss
%             save(save_file, 'model');
%             best_loss = tr_loss(i);
%         end
        % see test_result
%         global test_data;
%         global test_label;
%         [~, te_loss(index)] = test_conv(model, test_data, test_label);


    lr = df*lr;
end
loss = struct('tr_loss', tr_loss, 'te_loss', te_loss);
