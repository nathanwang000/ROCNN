% =============================================================================================
% matrix size reference:
% filter: filter_width * filter_height * num_channels * num_angles * filter maps
% =============================================================================================
% note: if desirable for each filter map to have different num_angles, then
% change filter to use struct representation
function =first_layer_rotation(filter)
    % this is a vanilla rotation as described in rotate_monte_carlo
    % given a filter, output rotated filters
    % debug
    angles = [0, pi/4, pi/2, 3/4*pi, pi, 5/4*pi, 3/2*pi, 7/4*pi];
    
    [h w c] = size(filter);
    for i=1:c
        rotate_monte_carlo(filter,);    
    end
    
end

function rot_filter=rotate_3d_monte_carlo(filter, theta)
    rot_filter = zeros(size(filter));
    for i = 1:size(filter,3)
        rot_filter(:,:,i) = rotate_monte_carlo(filter(:,:,1),theta);
    end
end