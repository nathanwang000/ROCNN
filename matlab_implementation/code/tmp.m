% assume given input, model
angle = 0
while angle < 360
    sample = imresize(imrotate(input,angle),[28,28]);
    [o,a] = inference(model,sample);
    plot(a{3}(:));
    ylim([0,0.2]);
    hold on;
    angle = angle + 45;
end