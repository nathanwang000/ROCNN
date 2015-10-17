function debug_ro_output()
    ro_layer = init_layer('ro_conv',struct('num_angle',1,'filter_height',3,'filter_width',3,'filter_depth',1, 'num_channels',1,'num_filters',3,'circular',false)); % 26*26*1*3
    conv_layer = init_layer('conv',struct('filter_size',3,'filter_depth',1,'num_filters',3)); % 26*26*3

    % get some input and data
    load_MNIST_data; % get train_data, train_label, test_data, test_label
    % get a batch of input
    el = randsample(size(train_data,4),10);
    input = train_data(:,:,:,el);
    label = train_label(el);
    
    params = ro_layer.params;
    [ro_output,dv_input,grad] = ro_layer.fwd_fn(input, params, ro_layer.hyper_params, false, []);
    [conv_output,dv_input,grad] = conv_layer.fwd_fn(input, permute(params,[1,2,4,5,3]), conv_layer.hyper_params, false, []);
    
    % adjust output size for ro_output
    ro_output = permute(ro_output,[1,2,4,5,3]);
    
end