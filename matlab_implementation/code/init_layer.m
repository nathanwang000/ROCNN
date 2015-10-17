function layer = init_layer(type, info)
% Given a layer name, initializes the layer structure properly with the
% weights randomly initialized.
% 
% Input:
%	type - Layer name (options: linear, conv, pool, softmax, flatten, relu)
%	info - Structure holding hyper parameters that define the layer
%
% Examples: init_layer('linear',struct('num_in',18,'num_out',10)
% 			init_layer('softmax',[])
% to better initialize the weight, use the recommendation in http://cs231n.github.io/neural-networks-2/#init

% Parameters for weight initialization
addpath pcode;
weight_init = @randn;
if isfield(info,'weight_scale') ws = info.weight_scale;
else ws = 0.1; end
if isfield(info,'bias_scale') bs = info.bias_scale;
else bs = 0.1; end

params = struct('W',[],'b',[]);
switch type
    case 'sigmoid'
        fn = @fn_sigmoid;
	case 'linear'
		% Requires num_in, num_out
        fn = @fn_linear;		
        ws = sqrt(2.0/info.num_in); % following suggestion in http://arxiv.org/abs/1502.01852
		W = weight_init(info.num_out, info.num_in)*ws;
		b = weight_init(info.num_out, 1)*bs;
		params.W = W;
		params.b = b;
	case 'conv'
		% Requires filter_size, filter_depth, num_filters
		fn = @fn_conv;		
		W = weight_init(info.filter_size, info.filter_size, info.filter_depth, info.num_filters)*ws;
		b = weight_init(info.num_filters, 1)*bs;
		params.W = W;
		params.b = b;
    case 'ro_conv'
		% Requires filter_height, filter_width, filter_depth, num_channels, num_filters, num_angle
		fn = @fn_ro_conv;
        % params.W: filter_height, filter_width, filter_depth (in_num_angle), num_channels, num_filters
		W = weight_init(info.filter_height, info.filter_width, info.filter_depth, info.num_channels, info.num_filters)*ws;
		b = weight_init(info.num_filters, 1)*bs;
		params.W = W;
		params.b = b;
        info.angles = 0:2*pi/info.num_angle:(2*pi-0.001); % -0.001 for not double counting 360
	case 'pool'
		% Requires filter_size and optionally stride (default stride = 1)
		fn = @fn_pool;	
    case '3dpool'
        % Requires filter_size, optionally stride (default to filter_size)
        fn = @fn_3dpool;
    case '3dAvgPool'
        fn = @fn_3dAvgPool;
	case 'softmax'
		fn = @fn_softmax;
	case 'flatten'
		% Requires the number of dimensions of the output of the previous layer.
		% The parameter should be defined by info.num_dims
		fn = @fn_flatten;
	case 'relu'
		fn = @fn_relu;
    case 'ro_linear'
		% Requires filter_height, filter_width, filter_depth, num_channels, num_filters, num_angle
		fn = @fn_ro_linear;
        info.num_in = info.filter_height*info.filter_width*info.filter_depth*info.num_channels;
        info.num_out = info.num_filters;
        ws = sqrt(2.0/info.num_in); % following suggestion in http://arxiv.org/abs/1502.01852
		params.W = weight_init(info.filter_height, info.filter_width, info.filter_depth, info.num_channels, info.num_filters)*ws;
		params.b = weight_init(info.num_out, 1)*bs;
        % put a mask on W
        if isfield(info,'dropout') && info.dropout % is actually not dropout but droput like
            num_channels = size(params.W,4);
            num_filters = size(params.W,5);
            assert(num_channels==num_filters,'dropout for ro_conv cannot be applied, make sure num_in and num_out are the same');
            % weights from other channel is treated as 0
            mask = zeros(size(params.W));
            for i=1:num_channels
                mask(:,:,:,i,i) = 1; % num_channel == num_filter
            end
            params.W = params.W .* mask;
        end
    case 'addDim'
        fn = @fn_add_dim;
    case 'minusDim'
        fn = @fn_minus_dim;
end

layer = struct('fwd_fn', fn, 'type', type, 'params', params, 'hyper_params', info);
