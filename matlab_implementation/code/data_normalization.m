function data=data_normalization(data)
    % assume gray image
    data = normalize(center(data));
end

function data = normalize(data)
    sig = std(data.train_data(:));
    data.train_data = data.train_data/sig;
    data.val_data = data.val_data/sig;
    data.test_data = data.test_data/sig;
end

function data=center(data)
    mu = mean(mean(mean(data.train_data)));
    data.train_data = data.train_data - mu;
    data.val_data = data.val_data - mu;
    data.test_data = data.test_data - mu;
end

function data = PCA_white(data)
    data = center(data);
    X = reshape(data.train_data,[],size(data.train_data,4))';
    cov = X*X';
    [U,S,V] = svd(X); % columns of U is the eigen vector of XX'
    Xrot = X*U; % project X onto columns U
    Xwhite = Xrot / sqrt(S+1e-5); % be caution about this 
end