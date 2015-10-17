function [accuracy, loss]=test_conv(model, input, label)
    n = 400;
    loss = 0;
    totalCorrect = 0;
    if size(input,4) <= n
        [output, activations] = inference(model, input);
        [~, prediction] = max(output);
        prediction = prediction';
        accuracy = sum(prediction==label)/length(label);
        % calculate test loss
        [loss,~] = loss_crossentropy(output, label,[], false);
        loss = loss / size(input,4);
    else
        % partition input
        fprintf('total %d data\n', length(label));
        partition = 1:n:size(input,4);
        for i=1:length(partition)
            from = partition(i); 
            if i==length(partition)
                to = size(input,4);
            else
                to = partition(i+1)-1;
            end
            [output, activations] = inference(model, input(:,:,:,from:to));
            [~, prediction] = max(output);
            prediction = prediction';
            nCorrect = sum(prediction==label(from:to));
            totalCorrect = totalCorrect + nCorrect;
            fprintf('index %d acc is %f\n',i,nCorrect/(to-from));            
            % calculate test loss
            [l,~] = loss_crossentropy(output, label(from:to),[], false);
            loss = loss + l;
        end
        accuracy = totalCorrect / size(input,4);
        loss = loss / size(input,4);
    end
end