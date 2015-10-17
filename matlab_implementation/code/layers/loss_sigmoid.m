% ======================================================================
% Matrix size reference:
% ----------------------------------------------------------------------
% input: num_nodes * batch_size
% labels: batch_size * 1
% ======================================================================

function [loss, dv_input] = loss_sigmoid(input, labels, hyper_params, backprop)

%assert(max(labels) <= size(input,1));
assert(size(input,1) == 1); % make sure the loss makes sense
[num_nodes ,batch_size] = size(input);

% calculate loss
loss = 0;
% verbose method
% for i=1:batch_size
%    loss = loss + crossentropy(input(:,i), labels(i));
% end
input = [input;1-input]; % augment the input
% simplified version
mask = zeros(size(input));
for i=1:batch_size
    mask(labels(i),i) = 1;
end
mask = logical(mask);
loss = sum(-log(input(mask))); % where mask(i, j) is 1 if lables(j) == i else 0

dv_input = zeros(size(input));
if backprop
    % calculate dL/dx
    dv_input(mask) = -1 ./ input(mask);
    dv_input = dv_input(1,:) - dv_input(2,:);
    assert(size(dv_input,1) == 1 && size(dv_input,2) == batch_size);
end
end

function loss = crossentropy(x, label)
    % label in range 0-9, verbose function
    loss = -log(x(label));
end