% ======================================================================
% Matrix size reference:
% input: in_height, in_width, in_num_angle, num_channels (num_filters in the previous layer), batch_size
% output: out_height, out_width, out_num_angle, num_filters, batch_size
% hyper parameters: circular (whether apply circular shift, false for first layer, defaut true), ... 
% ======================================================================
function [output, dv_input, grad]=fn_3dpool(input, params, hyper_params, backprop, dv_output)
    % three dimensional max pooling
    % default: only pool along translation with stride of hyper_params.filter_size
    % wrong!!!!!!!!!!!!!!!!!!!
    
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
        expected_output = repmat([1 3; 2 4], dim);

        backprop = true;
        dv_output = repmat([3 1; 1 9], dim); %ceil(rand(2)*10); % randint(1,10)
        expected_dv_input = [3 0 1 0;
                             0 3 0 1;
                             0 1 0 0;
                             0 1 9 9];
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
               tmp = ordfilt2(A,filter_size^2,ones(filter_size));
               output(:,:,k,j,i) = tmp(1:stride:end-filter_size+1,1:stride:end-filter_size+1);
           end
       end
    end
   
    dv_input = [];
    grad = struct('W',[],'b',[]);    
    % backward
    if backprop
        dv_input = zeros(size(input));
        % temperary solution for non overlapping pooling
        for i=1:batch_size
            for j=1:num_channels
                for k=1:out_num_angle
                    tmp = kron(output(:,:,k,j,i), ones(filter_size));
                    mask = (tmp == input(:,:,k,j,i));

%                     tmp = kron(output(:,:,k,j,i), ones(filter_size));
%                     [h w] = size(tmp);
%                     tmp2 = zeros(in_height, in_width); % pad zeros
%                     tmp2(1:h,1:w) = tmp;
%                     mask = (tmp2 == input(:,:,k,j,i));
                    
%                     % fix the mask by using conv
%                     mask = double(mask);
%                     tmp = conv2(mask, ones(filter_size),'valid');
%                     tmp = tmp(1:stride:end,1:stride:end);
%                     tmp = kron(tmp, ones(filter_size));
%                     [h w] = size(tmp);
%                     tmp2 = ones(in_height, in_width); % pad ones
%                     tmp2(1:h,1:w) = tmp;
%                     mask = (mask./tmp2);
                    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                    dv_input(:,:,k,j,i) = kron(dv_output(:,:,k,j,i), ones(filter_size)).*mask ; 
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