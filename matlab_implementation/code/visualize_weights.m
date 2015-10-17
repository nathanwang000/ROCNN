% a similar function is in visualize_ro_conv_activation.m
function visualize_weights(W)
% eg call: visualize_weights(model, W(:,:,1,1,1))
% W should be 2 d
[w h c] = size(W);
assert(c==1,'W dimension incorrect');
% imagesc(W)
 W = W - min(min(W));
 W = W / max(max(W));
 imshow(W);

end

function filter_visualization()
    % for conv layer
    load model.mat
    model = model_info.model;
    data_size = 10;
    figure
    for i=1:2
        subplot(1,3,i);
        visualize_weights(model.layers(1).params.W(:,:,1,i));
        title(sprintf('filter %d', i));    
    end
    subplot(1,3,3)
    visualize_weights(model.layers(1).params.W(:,:,1,1)-model.layers(1).params.W(:,:,1,2));
    title('filter diff');    
    
    % for linear layer
    load model.mat
    model = model_info.model;
    data_size = 10;
    figure
    for i=1:2
        subplot(1,3,i);
        visualize_weights(reshape(model.layers(2).params.W(i,:),[data_size data_size]));
        title(sprintf('filter %d', i));    
    end
    subplot(1,3,3)
    visualize_weights(reshape(model.layers(2).params.W(1,:) - model.layers(2).params.W(2,:),[data_size data_size]));
    title('filter diff');    
    
    % for ro_conv layer
    load model.mat
    model = model_info.model;
    figure
    for i=1:2
        subplot(1,3,i);
        visualize_weights(model.layers(1).params.W(:,:,1,1,i));
        title(sprintf('filter %d', mod(i,10)));    
        axis off;
    end
    subplot(1,3,3)
    W1 = model.layers(1).params.W(:,:,1,1,1);
    W2 = model.layers(1).params.W(:,:,1,1,2);
    layer2W1 = reshape(model.layers(3).params.W(:,:,:,1,1),[1 8]);
    layer2W2 = reshape(model.layers(3).params.W(:,:,:,2,2),[1 8]);
    num_angle = 8;
    angles = 0:2*pi/num_angle:(2*pi-0.001); % -0.001 for not double counting 360    
    newW1 = zeros(size(W1)); newW2 = zeros(size(W2));
    for i=1:num_angle
        newW1 = newW1 + layer2W1(i) * circular_rotate(W1, angles(i));
        newW2 = newW2 + layer2W2(i) * circular_rotate(W2, angles(i));
    end
    visualize_weights(newW1 - newW2);
    % visualize_weights(model.layers(1).params.W(:,:,1,1,1) - model.layers(1).params.W(:,:,1,1,2));
    title('filter diff');    
    % for final linear layer of ro_conv
    figure
    for i=1:2
        subplot(1,2,i);
        visualize_weights(reshape(model.layers(3).params.W(:,:,:,i,i),[1 8]));
        title(sprintf('filter %d', i));    
    end    
end

function twenty_filter_plot()
    load model.mat
    model = model_info.model;
    figure
    for i=1:5
        subplot(1,5,i);
        visualize_weights(model.layers(1).params.W(:,:,1,1,i));
        title(sprintf('%d', i));    
        axis off;
    end
end

function confusion()
    label = val_label;
    input = val_data;
    [predicted acc loss] = predict_label(model, input, label);
    confusionmat(label, predicted);
end


function data_visualize()
    % see pipeline.m plot for detail
    figure;
    %title(subplot(10,step,1),'train data plot');    
    for i = 1:data_size
        subplot(10,step,i);
        imshow(train_data(:,:,1,i));
    end
    figure;
    for i = 1:data_size
        subplot(10,step,i);
        imshow(val_data(:,:,1,i));
    end
    %title('val data plot');
end

function verify_rotation()
    addpath precomputation;
    filter = train_data(:,:,1,1);
    num_angle = 8;
    angles = 0:2*pi/num_angle:(2*pi-0.001); % -0.001 for not double counting 360
    num_angles = size(angles,2);
    figure
    for i=1:num_angles
        rot_filter=circular_rotate(filter,angles(i));
        subplot(1,num_angles,i);
        imshow(rot_filter);
    end    
    rmpath precomputation;
end

function rotate_input_visualization()
addpath('precomputation');
input = train_data(:,:,:,1);
% assume given input, model
angle = 0;
while angle < 360
    %sample = imresize(imrotate(input,angle),[28,28]);
    sample = rotate_monte_carlo(input,angle/360*2*pi);
    [o,a] = inference(model,sample);
    ylim([-10,10]);
    %plot(a{1}(:));
    plot(o);
    hold on;
    angle = angle + 45;
    pause
end
end

function activation_visualization()
i=16;
sample = val_data(:,:,:,i);
imshow(sample);
[o,a] = inference(model,sample);
 c = permute(a{1},[3 4 1 2])
end

function val_data_visualization()
figure
for i=1:100
    subplot(10,10,i);
    imshow(val_data(:,:,:,i));
    title(sprintf('data %d',i));
end
end

function model_info_vis()
load model.mat;
fprintf('validation acc is %f\n', model_info.val_acc)
plot(model_info.loss.tr_loss);
end

function visualize_experiments_on_one_vs_3()
    load 'one_unbalanced_orientation.mat'
    neg = test_data(:,:,:, test_label==2);
    figure;
    for i=1:36
        subplot(6,6,i);
        pic=imresize(imrotate(neg(:,:,:,1),10*i), [10 10]);imshow(pic); pred = inference(model,pic);
        title(pred(2)*100);
    end

end

