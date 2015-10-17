% ======================================================================
% Matrix size reference:
% input: in_height, in_width, in_num_angle, num_channels (num_filters in the previous layer), batch_size
% output: out_height, out_width, out_num_angle, num_filters, batch_size
% hyper parameters: circular (whether apply circular shift, false for first layer, defaut true), ... 
% ======================================================================
function [output, dv_input, grad]=fn_3dAvgPool(input, params, hyper_params, backprop, dv_output)
    % three dimensional average pooling
    % default: only pool along translation with stride of hyper_params.filter_size
    
    debug = false;
    if debug
        input = [1 0 3 0;
                 0 1 0 3;
                 0 2 0 0;
                 0 2 4 4];
        dim = [1 1 2 3 2];
        input = repmat(input, dim);
        stride = 2;
        filter_size = 2;
        expected_output = repmat([0.5 3/2; 1 2], dim);

        backprop = true;
        dv_output = repmat([3 1; 1 9], dim); %ceil(rand(2)*10); % randint(1,10)
        expected_dv_input = [3/4 3/4 1/4 1/4;
                             3/4 3/4 1/4 1/4;
                             1/4 1/4 9/4 9/4;
                             1/4 1/4 9/4 9/4];
        expected_dv_input = repmat(expected_dv_input, dim);
    else
        filter_size = hyper_params.filter_size;
        stride = filter_size; %hyper_params.stride; % note that this can be buggy
    end
    
    [in_height, in_width, in_num_angle, num_channels, batch_size] = size(input);
    out_height = floor((in_height - filter_size)/stride) + 1;
    out_width = floor((in_width - filter_size)/stride) + 1;
    out_num_angle = in_num_angle; % case for not pooling along the rotation dimension

    % forward
    output = zeros(out_height, out_width, out_num_angle, num_channels, batch_size);
    for i=1:batch_size
       for j=1:num_channels
           for k=1:out_num_angle
               A = input(:,:,k,j,i);
               tmp = conv2(A,ones(filter_size),'valid');
               output(:,:,k,j,i) = tmp(1:stride:end,1:stride:end);
           end
       end
    end
    output = output/(filter_size^2);
   
    dv_input = [];
    grad = struct('W',[],'b',[]);    
    % backward
    if backprop
        dv_input = zeros(size(input));
        % temperary solution for non overlapping pooling
        for i=1:batch_size
            for j=1:num_channels
                for k=1:out_num_angle
                    dv_input(:,:,k,j,i) = kron(dv_output(:,:,k,j,i), ones(filter_size)) / (filter_size^2); 
                end
            end
        end
        
    end
    
    if debug
        if isequal(output, expected_output) disp('pool forward is working properly');
        else disp('plz check pool forward code'); end
        if isequal(dv_input, expected_dv_input) disp('pool backward is working properly')
        else disp('plz check pool backward code'); end
    end
    
end