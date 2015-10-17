% rotation of filter for this project
function rot_filter = rotate_monte_carlo(filter, theta, prob_matrix)
    % debug
%      theta = -pi/2;
%      filter = [0 1 0; 0 2 0; 0 3 0];
    dim = length(size(filter));
    
    if ~exist('prob_matrix')
         prob_matrix = calc_prob_matrix(filter, theta);
    end
    
    [h, w] = size(filter);    
    % verbose way
%    rot_filter = zeros(size(filter));
%     for i=1:h
%         for j=1:w
%             rot_filter(i,j) = sum(sum(filter.*prob_matrix(:,:,i,j)));
%         end
%     end
    % 3 times faster
    if dim==2  % filter: filter_height, filter_width
        rot_filter = permute(sum(sum(repmat(filter,[1 1 size(prob_matrix,3) size(prob_matrix,4)]).*prob_matrix,1),2),[3 4 1 2]);    
    elseif dim == 4 % filter: filter_height, filter_width, filter_depth (in_num_angle), num_channels
        filter = reshape(filter,[size(filter,1),size(filter,2),1,1,size(filter,3),size(filter,4)]);
        rot_filter = permute(sum(sum( ...
            repmat(filter,[1 1 size(prob_matrix,3) size(prob_matrix,4) 1 1]).* ...
            repmat(prob_matrix,[1 1 1 1 size(filter,5) size(filter,6)]), ...
            1),2),[3 4 5 6 1 2]);    
    else
        assert(false, 'should not reach here!')
    end
end
