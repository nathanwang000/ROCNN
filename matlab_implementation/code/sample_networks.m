% the following are sample networks that can be used to replace the one in gen_model
    conv_net = [
        init_layer('conv',struct('filter_size',3,'filter_depth',1,'num_filters',3)) % 26*26*3
        init_layer('pool',struct('filter_size',2,'stride',2)) % 13*13*3
        % init_layer('conv',struct('filter_size',4,'filter_depth',3,'num_filters',5)) % 10*10*5
        % init_layer('pool',struct('filter_size',2,'stride',2)) % 5*5*5
        init_layer('flatten',struct('num_dims',4))
        init_layer('linear',struct('num_in',13*13*3,'num_out',30))
        init_layer('relu', []);
        init_layer('linear',struct('num_in',30,'num_out',10))
        init_layer('softmax',[])];

    sig_net = [init_layer('flatten',struct('num_dims',4))
        init_layer('linear',struct('num_in',28*28*1,'num_out',30))
        init_layer('sigmoid', [])
        init_layer('linear',struct('num_in',30,'num_out',10))
        init_layer('softmax',[])
        ];
    
    relu_net = [init_layer('flatten',struct('num_dims',4))
        init_layer('linear',struct('num_in',28*28*1,'num_out',30))
        init_layer('relu', [])
        init_layer('linear',struct('num_in',30,'num_out',10))
        init_layer('softmax',[])
        ];
