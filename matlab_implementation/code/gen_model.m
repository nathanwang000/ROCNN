function model=gen_model()
    addpath layers;
    tic
    data_size = 28;
    % networks
    ro_conv_net = [
        init_layer('ro_conv', struct('num_angle',8,'filter_height',data_size,'filter_width',data_size,...
                   'filter_depth',1,'num_channels',1,'num_filters',10,'circular',false)) % 1*1*8*10
        init_layer('relu', []) % 1*1*8*10
        init_layer('ro_linear', struct('filter_height',1,'filter_width',1,...
                   'filter_depth',8,'num_channels',10,'num_filters',10,'dropout',true))
        init_layer('softmax',[])
    ];

    mnist_rot_ro_conv_net = [
        init_layer('ro_conv',struct('num_angle',8,'filter_height',11,'filter_width',11,...
            'filter_depth',1, 'num_channels',1,'num_filters',20,'circular',false))
        init_layer('relu', []) % 18*18*8*20
        init_layer('3dAvgPool',struct('filter_size',2)) % 9*9*8*20, filter_size is also the stride size        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
%         init_layer('flatten',struct('num_dims',5))
%         init_layer('linear',struct('num_in',9*9*8*20,'num_out',500)) % 500
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ro_conv version of linear layer
        init_layer('ro_linear', struct('filter_height',9,'filter_width',9,...
             'filter_depth',8, 'num_channels',20,'num_filters',500))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        init_layer('linear',struct('num_in',500,'num_out',10)) % 10
        init_layer('softmax',[])
    ];

    mnist_rot_ro_conv_net_one_three = [ % result stored in model_one_three.mat
        init_layer('ro_conv',struct('num_angle',8,'filter_height',11,'filter_width',11,...
            'filter_depth',1, 'num_channels',1,'num_filters',20,'circular',false))
        init_layer('relu', []) % 18*18*8*20
        init_layer('3dAvgPool',struct('filter_size',2)) % 9*9*8*20, filter_size is also the stride size        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ro_conv version of linear layer
        init_layer('ro_linear', struct('filter_height',9,'filter_width',9,...
             'filter_depth',8, 'num_channels',20,'num_filters',100))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        init_layer('linear',struct('num_in',100,'num_out',2)) % 2
        init_layer('softmax',[])
    ];

    mnist_rot_ro_conv_net_one_three_extended = [ % too slow
        init_layer('ro_conv',struct('num_angle',8,'filter_height',5,'filter_width',5,...
            'filter_depth',1, 'num_channels',1,'num_filters',10,'circular',false))
        init_layer('relu', []) % 24*24*8*10
        init_layer('3dAvgPool',struct('filter_size',2)) % 12*12*8*10, filter_size is also the stride size        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        init_layer('ro_conv',struct('num_angle',8,'filter_height',7,'filter_width',7,...
            'filter_depth',8, 'num_channels',10,'num_filters',5,'circular',true))
        init_layer('relu', []) % 6*6*8*5
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ro_conv version of linear layer
        init_layer('ro_linear', struct('filter_height',6,'filter_width',6,...
             'filter_depth',8, 'num_channels',5,'num_filters',2))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        init_layer('softmax',[])
    ];

    mnist_rot_ro_conv_net_one_three_extended_small = [
        init_layer('ro_conv',struct('num_angle',8,'filter_height',3,'filter_width',3,...
            'filter_depth',1, 'num_channels',1,'num_filters',5,'circular',false))
        init_layer('relu', []) % 8*8*8*5
        init_layer('3dAvgPool',struct('filter_size',2)) % 4*4*8*10, filter_size is also the stride size        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        init_layer('ro_conv',struct('num_angle',8,'filter_height',4,'filter_width',4,...
            'filter_depth',8, 'num_channels',5,'num_filters',3,'circular',true))
        init_layer('relu', []) % 1*1*8*3
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ro_conv version of linear layer
        init_layer('ro_linear', struct('filter_height',1,'filter_width',1,...
             'filter_depth',8, 'num_channels',3,'num_filters',2))
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%        
        init_layer('softmax',[])
    ];


    toy_ro_conv_net_1 = [ % use to show circular rotation works
        init_layer('ro_conv', struct('num_angle',8,'filter_height',data_size,'filter_width',data_size,...
                   'filter_depth',1,'num_channels',1,'num_filters',1,'circular',false)) % 1*1*8*1
        init_layer('relu', []) % 1*1*8*1
        init_layer('ro_conv', struct('num_angle',8,'filter_height',1,'filter_width',1,'filter_depth',8,...
                   'num_channels',1,'num_filters',1,'circular',true)) % 1*1*8*1
        init_layer('flatten',struct('num_dims',5))
    ];
    
    toy_ro_conv_net_2 = [ % binary classification (most useful!!!)
        init_layer('ro_conv', struct('num_angle',8,'filter_height',data_size,'filter_width',data_size,...
                   'filter_depth',1,'num_channels',1,'num_filters',2,'circular',false)) % 1*1*8*1
        init_layer('relu', []) % 1*1*8*2
        init_layer('ro_linear', struct('filter_height',1,'filter_width',1,...
                   'filter_depth',8,'num_channels',2,'num_filters',2,'dropout',true))
        init_layer('softmax',[])        
    ];
 
    toy_ro_conv_net_3 = [ % not really working
        init_layer('ro_conv', struct('num_angle',8,'filter_height',data_size,'filter_width',data_size,...
                   'filter_depth',1,'num_channels',1,'num_filters',1,'circular',false)) % 1*1*8*1
        init_layer('sigmoid', []) % 1*1*8*1
        init_layer('flatten',struct('num_dims', 5)) % 8
        init_layer('linear',struct('num_in',8,'num_out',1))
        init_layer('sigmoid',[])
    ];
    
    toy_linear_net = [
        init_layer('flatten',struct('num_dims',4))
        init_layer('linear',struct('num_in',data_size*data_size,'num_out',2))
        init_layer('softmax',[]);
    ];

    toy_conv_net = [
        init_layer('conv', struct('filter_size',10, 'filter_depth',1, 'num_filters',2))
        init_layer('relu',[])
        init_layer('flatten',struct('num_dims',4))
        init_layer('softmax',[])
    ];

    debug_net = [
        init_layer('ro_conv', struct('num_angle',8,'filter_height',5,'filter_width',5,...
                   'filter_depth',1,'num_channels',1,'num_filters',10,'circular',false)) % 24*24*8*10
        init_layer('ro_conv',struct('num_angle',8,'filter_height',24,'filter_width',24,...
            'filter_depth',8, 'num_channels',10,'num_filters',1,'circular',true)) % 1*1*8*1
        init_layer('flatten',struct('num_dims',5)) % 8
    ];

   mnistRotBig1 = [
        init_layer('ro_conv',struct('num_angle',8,'filter_height',11,'filter_width',11,...
                                    'filter_depth',1, 'num_channels',1,'num_filters',40,'circular',false))
        init_layer('relu', []) % 18*18*8*20
        init_layer('3dAvgPool',struct('filter_size',2)) % 9*9*8*20, filter_size is also the stride size        
    %%%%%%%%%%%%%%%%%%%%%%%%%% ro_conv version of linear layer                                                                                                                                               
        init_layer('ro_linear', struct('filter_height',9,'filter_width',9,...
                                       'filter_depth',8, 'num_channels',40,'num_filters',500))
    %%%%%%%%%%%%%%%%%%%%%%%%%%                                                                                                                                                                               
        init_layer('linear',struct('num_in',500,'num_out',10)) % 10
        init_layer('softmax',[])
    ];

    mnistRotBig2 = [
        init_layer('ro_conv',struct('num_angle',8,'filter_height',11,'filter_width',11,...
            'filter_depth',1, 'num_channels',1,'num_filters',20,'circular',false))
        init_layer('relu', []) % 18*18*8*20
        init_layer('3dAvgPool',struct('filter_size',2)) % 9*9*8*20, filter_size is also the stride size    
        init_layer('ro_conv',struct('num_angle',8,'filter_height',9,'filter_width',9,...
            'filter_depth',8, 'num_channels',20,'num_filters',10,'circular',true)) % 1*1*8*10        
        init_layer('relu', [])
        init_layer('ro_linear', struct('filter_height',1,'filter_width',1,...
                   'filter_depth',8,'num_channels',10,'num_filters',10,'dropout',true))
        init_layer('softmax', [])
    ];

    mnistRotBig3 = [ % too slow to run
        init_layer('ro_conv',struct('num_angle',8,'filter_height',11,'filter_width',11,...
                                    'filter_depth',1, 'num_channels',1,'num_filters',20,'circular',false))
        init_layer('relu', []) % 18*18*8*20
        init_layer('3dAvgPool',struct('filter_size',2)) % 9*9*8*20, filter_size is also the stride size        
        init_layer('ro_conv',struct('num_angle',8,'filter_height',5,'filter_width',5,...
                                    'filter_depth',8, 'num_channels',20,'num_filters',20,'circular',true))
    %%%%%%%%%%%%%%%%%%%%%%%%%% ro_conv version of linear layer                                                                                                                                               
        init_layer('ro_linear', struct('filter_height',5,'filter_width',5,...
                                       'filter_depth',8, 'num_channels',20,'num_filters',500))
    %%%%%%%%%%%%%%%%%%%%%%%%%%                                                                                                                                                                               
        init_layer('linear',struct('num_in',500,'num_out',10)) % 10
        init_layer('softmax',[])
    ];

   mnistRotRef = [ % too slow to run
        init_layer('ro_conv',struct('num_angle',8,'filter_height',11,'filter_width',11,...
            'filter_depth',1, 'num_channels',1,'num_filters',20,'circular',false))
        init_layer('relu', []) % 18*18*8*20
        init_layer('3dAvgPool',struct('filter_size',2)) % 9*9*8*20, filter_size is also the stride size    
        init_layer('ro_conv',struct('num_angle',8,'filter_height',9,'filter_width',9,...
            'filter_depth',8, 'num_channels',20,'num_filters',10,'circular',true)) % 1*1*8*10        
        init_layer('relu', [])
        init_layer('ro_linear', struct('filter_height',1,'filter_width',1,...
                   'filter_depth',8,'num_channels',10,'num_filters',10,'dropout',true))    
        init_layer('softmax', [])
    ];


    % layers, input_size (h*w*prev_num_angle*depth), output_size, visualize_each_layer
    % model = init_model(toy_ro_conv_net_2,[data_size data_size 1],2,true);
    %model = init_model(mnist_rot_ro_conv_net,[data_size data_size 1],10,true);
    model = init_model(mnistRotRef,[data_size data_size 1],10,false);
    %model = init_model(debug_net,[data_size data_size 1],8,true);
toc
end
