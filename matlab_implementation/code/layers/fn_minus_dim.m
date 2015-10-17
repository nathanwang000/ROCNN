% ======================================================================
% Matrix size reference:
% input: in_height, in_width, in_num_angle, num_channels (num_filters in the previous layer), batch_size
% output: out_height, out_width, out_num_angle, num_filters, batch_size
% hyper parameters: circular (whether apply circular shift, false for first layer, defaut true), ... 
% ======================================================================
function [output, dv_input, grad]=fn_minus_dim(input, params, hyper_params, backprop, dv_output)
    % reduce a 5d input to 4d
    assert(size(input,3) == 1); % only reduce if in_num_angle is 1
    output = permute(input,[1 2 4 5 3]);
    dv_input = [];
    grad = struct('W',[],'b',[]);    
    if backprop
       dv_input = permute(dv_output,[1 2 5 3 4]) ;
    end
end