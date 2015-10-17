function rot_filter=circular_rotate(filter,theta,rotate_index,prob_matrix,circular)
    % filter: filter_height, filter_width, filter_depth (in_num_angle), num_channels
    % theta: the angle to rotate
    % rotate_index: the amount of circular shift
    
    % debug
%     filter = reshape(1:27,[3 3 3 1]);
%     theta = pi/2;
%     rotate_index = 2;
     
    if ~exist('prob_matrix') 
        %assert(false, 'should never run!')
        prob_matrix = calc_prob_matrix(filter, theta); 
    end
    if ~exist('circular') circular = true; end

    [filter_height, filter_width, filter_depth, num_channels] = size(filter);
    
    % verbose way
    rot_filter = zeros(size(filter));
    i = 1; j=1;
    rot_filter(:,:,j,i) = rotate_monte_carlo(filter(:,:,j,i), theta, prob_matrix); % reuse prob matrix here
%     for i=1:num_channels
%         for j=1:filter_depth
%             rot_filter(:,:,j,i) = rotate_monte_carlo(filter(:,:,j,i), theta, prob_matrix); % reuse prob matrix here
%         end
%         if circular
%             rot_filter(:,:,:,i) = circshift(rot_filter(:,:,:,i),rotate_index,3);
%         end     
%     end
%     
    % 3 times faster
        %rot_filter = rotate_monte_carlo(filter, theta, prob_matrix);
%     if ~circular
% 
%     else
%         
%     end
    

end