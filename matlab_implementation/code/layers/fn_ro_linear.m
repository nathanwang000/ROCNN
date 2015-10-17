% ======================================================================
% Matrix size reference:
% note: num_out * num_in see init_layer for detail
% input: in_height, in_width, in_num_angle, num_channels (num_filters in the previous layer), batch_size
% output: num_out * batch_size
% hyper_params:
% params.W: filter_height, filter_width, filter_depth (in_num_angle), num_channels, num_filters
% params.b: num_out * 1
% dv_output: same as output
% dv_input: same as input
% grad: same as params
% ======================================================================

function [output, dv_input, grad] = fn_ro_linear(input, params, hyper_params, backprop, dv_output)
% rotation optional linear layer
num_dims = 5;
in_dim = size(input);
batch_size = size(input,num_dims); 
input = reshape(input,[],batch_size); % new input

[num_in,batch_size] = size(input);
assert(num_in == hyper_params.num_in,...
	sprintf('Incorrect number of inputs provided at linear layer.\nGot %d inputs expected %d.',num_in,hyper_params.num_in));

output = zeros(hyper_params.num_out, batch_size);
% FORWARD CODE
if isfield(hyper_params,'dropout') && hyper_params.dropout % is actually not dropout but droput like
    num_channels = size(params.W,4);
    num_filters = size(params.W,5);
    assert(num_channels==num_filters,'dropout for ro_conv cannot be applied, make sure num_in and num_out are the same');
    % weights from other channel is treated as 0
    mask = zeros(size(params.W));
    for i=1:num_channels
        mask(:,:,:,i,i) = 1; % num_channel == num_filter
    end
end

if isfield(hyper_params,'dropout') && hyper_params.dropout % is actually not dropout but droput like
    params.W = params.W .* mask;
    W = reshape(params.W,[],hyper_params.num_filters)';    
else
    W = reshape(params.W,[],hyper_params.num_filters)';
end
    
output = W * input + repmat(params.b, [1, batch_size]);
grad = struct('W',[],'b',[]);
dv_input = [];
% backward
if backprop
    W_in_dim = size(params.W);
	dv_input = zeros(size(input));
	gradW = zeros(size(W));
	grad.b = zeros(size(params.b));
	% BACKPROP CODE
    gradW = dv_output * input';
    dv_input = reshape(W' * dv_output,in_dim); % correct for dropout
    grad.b = sum(dv_output,2); % correct for dropout
    grad.W = reshape(gradW',W_in_dim); % correct for dropout but to pass gradient check, grad.W should be again be filtered by mask
    if isfield(hyper_params,'dropout') && hyper_params.dropout % is actually not dropout but droput like
        grad.W = grad.W .* mask;
    end
end
end
