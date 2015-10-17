% test ro_conv with circular rotation
addpath('precomputation')

% 1) circular rotation is working properly
% params.W: filter_height, filter_width, filter_depth (in_num_angle), num_channels, num_filters
filter1 = [0 1 0; 0 1 0; 0 1 0;];
filter1 = cat(3,filter1,[0 1 0; 0 0 0; 0 1 0]);
filter1 = cat(3,filter1,[0 1 0; 0 0 0; 0 0 0]);
filter2 = [0 0 0; 1 1 1; 0 0 0;];
filter2 = cat(3,filter2,[0 0 0; 1 0 1; 0 0 0]);
filter2 = cat(3,filter2,[0 0 0; 0 0 1; 0 0 0]);
filter = cat(4,filter1,filter2);
rotFilter = circular_rotate(filter,4*pi/4,1);

%% filter visualization
for r=1:2
    for c=1:3
        subplot(2,3,(r-1)*3+c);
        imshow(filter(:,:,c,r));
    end
end
figure
for r=1:2
    for c=1:3
        subplot(2,3,(r-1)*3+c);
        imshow(rotFilter(:,:,c,r));
    end
end
