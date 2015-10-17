% ======================================================================
% Matrix size reference:
% input: in_height, in_width, num_channels (num_filters in the previous layer), batch_size
% output: out_height, out_width, out_num_angle, num_filters, batch_size
% hyper parameters: circular (whether apply circular shift, false for first layer, defaut true), ... 
% ======================================================================
function [output, dv_input, grad]=fn_add_dim(input, params, hyper_params, backprop, dv_output)
    % change 4d regular conv input to 5d ro-conv
    output = permute(input, [1 2 5 3 4]);
    dv_input = [];
    grad = struct('W',[],'b',[]);        
    if backprop
        dv_input = permute(dv_output, [1 2 4 5 3]);
    end
end