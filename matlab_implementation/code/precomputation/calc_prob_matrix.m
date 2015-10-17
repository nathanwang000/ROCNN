% the only relevent information in filter is filter width and filter heigth
function prob_matrix = calc_prob_matrix(filter, theta)
        
    h = size(filter,1); w = size(filter,2);    

    rot_matrix = get_rot_matrix(-theta);
    prob_matrix = zeros([size(filter), size(filter)]); % 4 d matrix
    
    % draw n samples, either randomly or evenly spaced, from -0.5, 0.5
    n = 100000;
    n_per_bin = n/h/w ;% the expected number of pt draw in each bin
    pts = rand(2,n)-0.5;
    % find the origin of every points
    for i=1:n
       pt = pts(:,i);
       ori_pt = rot_matrix*pt;
       if ori_pt(1) > 0.5 || ori_pt(2) > 0.5 || ori_pt(1) < -0.5 || ori_pt(2) < -0.5; continue; end
       % note the bin of both pt and ori_pt
       pt_bin = [h+1-ceil((pt(2)+0.5)*h), ceil((pt(1)+0.5)*w)]; 
       ori_pt_bin = [h+1-ceil((ori_pt(2)+0.5)*h), ceil((ori_pt(1)+0.5)*w)];
       % update the prob_matrix
       prob_matrix(ori_pt_bin(1), ori_pt_bin(2), pt_bin(1), pt_bin(2)) = ...
       prob_matrix(ori_pt_bin(1), ori_pt_bin(2), pt_bin(1), pt_bin(2)) + 1;
    end
        
    prob_matrix = prob_matrix / n_per_bin;

end

function rot_matrix=get_rot_matrix(theta)
    rot_matrix = [cos(theta), -sin(theta); sin(theta), cos(theta)];
end