% ======================================================================
% Matrix size reference:
% input: in_height, in_width, in_num_angle, num_channels (num_filters in the previous layer), batch_size
% or     in_height, in_width,               num_channels (num_filters in the previous layer), batch_size for first roconv layer
% output: out_height, out_width, out_num_angle, num_filters, batch_size
% hyper parameters: circular (whether apply circular shift, false for first layer, defaut true), ... 
% params.W: filter_height, filter_width, filter_depth (in_num_angle), num_channels, num_filters
% params.b: num_filters * 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ======================================================================

function [output, dv_input, grad] = fn_ro_conv(input, params, hyper_params, backprop, dv_output)
% rotation optional convolution layer
% to optimize the code: plz
% 1) record prob_matrix for each orientation, change call to circular_rotate and calc_prob_matrix accordingly: done
% 2) change circular rotate to use the given prob_matrix: done
mfilepath=fileparts(which(mfilename));
addpath(fullfile(mfilepath,'../precomputation'));
% debug, these should go into the hyper_params
test = false;
if test
    out_num_angle = 4;
    angles = 0:2*pi/out_num_angle:(2*pi-0.001);
    hyper_params.circular = true;
    input = 0.7+0.2*randn(ceil(rand([1 5])*5+5)); % 5 dim: randint from 6 to 10, input: mean 0.7, std 0.2
    rn = ceil(rand([1 4])*3+1); % random numbers
    params.W = 0.2*randn([rn(1),rn(2),size(input,3),rn(3),rn(4)]); % 5 dim: randint from 2:4, W: mean 0, std 0.2
    params.b = 0.1*randn([rn(4) 1]); % rn(4) dim, b: mean 0, std 0.1
    backprop = true;
else
    out_num_angle = hyper_params.num_angle;
    angles = hyper_params.angles;
end

if ~hyper_params.circular % deal with 4D first layer input: transform into 5D
    input = permute(input, [1 2 5 3 4]);
end
[in_height,in_width,in_num_angle,num_channels,batch_size] = size(input);
[filter_height,filter_width,filter_depth,num_channels,num_filters] = size(params.W);
assert(filter_depth == in_num_angle, 'Filter depth does not match number of input angles');

% should precompute this and then load this quantity, fixme!!!!!!!!!!!!
if exist(['prob_m_' int2str(filter_width) '_' int2str(filter_height) '_' int2str(out_num_angle) '.mat'])
    load(['prob_m_' int2str(filter_width) '_' int2str(filter_height) '_' int2str(out_num_angle) '.mat']);
else
    fprintf('make sure this never runs in production');
    %assert(test,'make sure this never runs in production');
    prob_matrices = zeros([filter_height, filter_width, filter_height, filter_width, out_num_angle]);
    % 0 degree rotation is no rotation
%     for i=1:filter_height
%         for j=1:filter_width
%             prob_matrices(i,j,i,j,1) = 1;
%         end
%     end
    % 180 degree rotation is flip and flip
%     for i=1:filter_height
%         for j=1:filter_width
%             prob_matrices(filter_height-i+1,filter_width-j+1,i,j,2) = 1;
%         end
%     end
    % else
    for r=1:out_num_angle
        prob_matrices(:,:,:,:,r) = calc_prob_matrix(zeros(filter_height, filter_width), angles(r));
    end
    save(['prob_m_' int2str(filter_width) '_' int2str(filter_height) '_' int2str(out_num_angle) '.mat'], 'prob_matrices');
end

out_height = size(input,1) - size(params.W,1) + 1; % this is for stride 1 change this when needed !
out_width = size(input,2) - size(params.W,2) + 1; % this is for stride 1 change this when needed !
output = zeros(out_height,out_width,out_num_angle,num_filters,batch_size); 

if test
    % infer up and back to one layer above
    dv_output = randn(size(output));
end

% forward
% tic;
% for j = 1:num_filters
%     b = params.b(j,1);
%     parfor i = 1:batch_size % this should be parfor
%         for r = 1:out_num_angle
%             % circular rotate
%             W = circular_rotate(params.W(:,:,:,:,j),angles(r),r-1,prob_matrices(:,:,:,:,r),hyper_params.circular);
%             output(:,:,r,j,i) = b*ones(out_height,out_width); 
%             output(:,:,r,j,i) = output(:,:,r,j,i)+convn(input(:,:,:,:,i),flip(flip(W,3),4),'valid');
%             % note here doesn't pass test b/c filter_depth,num_channel are the same for W and input
%         end
%     end
% end
% t = toc;
% fprintf('output time is %f\n',t);

tic;
parfor j = 1:num_filters
    b = params.b(j,1);
    for i = 1:batch_size % this should be parfor
        for r = 1:out_num_angle
            % circular rotate
            W = params.W(:,:,:,:,j);
            % W = circular_rotate(params.W(:,:,:,:,j),angles(r),r-1,prob_matrices(:,:,:,:,r),hyper_params.circular);
            output(:,:,r,j,i) = b*ones(out_height,out_width); 
            output(:,:,r,j,i) = output(:,:,r,j,i)+convn(input(:,:,:,:,i),flip(flip(W,3),4),'valid');
            % note here doesn't pass test b/c filter_depth,num_channel are the same for W and input
        end
    end
end
t = toc; fprintf('output time is %f\n',t);


dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	gradW = zeros(size(params.W));
    % input: in_height, in_width, in_num_angle, num_channels (num_filters in the previous layer), batch_size    
    % output: out_height, out_width, out_num_angle, num_filters, batch_size    
    % params.b: num_filters * 1
    tic
    parfor i=1:batch_size 
       for r=1:out_num_angle
            for j=1:num_filters
                % can be optimize by using previously computed result
                W = params.W(:,:,:,:,j);
                %W = circular_rotate(params.W(:,:,:,:,j),angles(r),r-1,prob_matrices(:,:,:,:,r),hyper_params.circular);   
                for k=1:num_channels 
                    for k2=1:in_num_angle
                        dv_input(:,:,k2,k,i) = dv_input(:,:,k2,k,i)+conv2(dv_output(:,:,r,j,i), ...
                            rot90(W(:,:,k2,k),2),'full');
                    end
                end
            end
        end
    end
    t = toc; fprintf('dv_input time is %f\n',t);
    
    tic
    % params.W: filter_height, filter_width, filter_depth (in_num_angle), num_channels, num_filters
    dW = zeros([size(params.W,1) size(params.W,2) size(params.W,3), size(params.W,4) size(params.W,5) out_num_angle]);
    parfor r=1:out_num_angle
        for j=1:num_filters
            for k=1:num_channels
                for k2=1:in_num_angle
                    dW(:,:,k2,k,j,r) = zeros(filter_height, filter_width);
                    for i=1:batch_size
                        dW(:,:,k2,k,j,r) = dW(:,:,k2,k,j,r)+conv2(rot90(input(:,:,k2,k,i),2),dv_output(:,:,r,j,i),'valid');
                    end
                end
            end
        end
    end
    
    parfor f=1:num_filters
        for r=1:out_num_angle
            for i=1:filter_height
                for j=1:filter_width
                    if ~hyper_params.circular
%                         grad.W(i,j,:,:,f) = grad.W(i,j,:,:,f) + sum(sum(dW(:,:,:,:,f,r) .* ...
%                             repmat(permute(prob_matrices(i,j,:,:,r),[3 4 1 2 5]), [1 1 size(dW(:,:,:,:,f,r),3) size(dW(:,:,:,:,f,r),4)])...
%                             ,1),2);
                        gradW(i,j,:,:,f) = gradW(i,j,:,:,f) + sum(sum(dW(:,:,:,:,f,r) .* ...
                            repmat(permute(prob_matrices(i,j,:,:,r),[3 4 1 2 5]), [1 1 size(dW(:,:,:,:,f,r),3) size(dW(:,:,:,:,f,r),4)])...
                            ,1),2);

                    else
%                         grad.W(i,j,:,:,f) = grad.W(i,j,:,:,f) + circshift(sum(sum(dW(:,:,:,:,f,r) .* ...
%                             repmat(permute(prob_matrices(i,j,:,:,r),[3 4 1 2 5]), [1 1 size(dW(:,:,:,:,f,r),3) size(dW(:,:,:,:,f,r),4)])...
%                             ,1),2), -r+1, 3); % -r+1 b/c if r=1, it shouldn't rotate
                        gradW(i,j,:,:,f) = gradW(i,j,:,:,f) + circshift(sum(sum(dW(:,:,:,:,f,r) .* ...
                            repmat(permute(prob_matrices(i,j,:,:,r),[3 4 1 2 5]), [1 1 size(dW(:,:,:,:,f,r),3) size(dW(:,:,:,:,f,r),4)])...
                            ,1),2), -r+1, 3); % -r+1 b/c if r=1, it shouldn't rotate
                    end
                end
            end
        end
    end
    grad.W = gradW;
    t = toc; fprintf('grad.W time is %f\n',t);
    tic
    % for b
    % verbose
%     grad.b = zeros(size(params.b));
%     for j = 1:num_filters
%         grad.b(j) = sum(sum(sum(sum(dv_output(:,:,:,j,:)))));
%     end
    % 5xfaster
    grad.b = permute(sum(sum(sum(sum(dv_output,1),2),3),5),[4 1 2 3 5]);
    t = toc; fprintf('grad.b time is %f\n',t);
end

end


