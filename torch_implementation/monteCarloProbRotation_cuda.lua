-- gpu version of mcpr
require 'math'
require 'image'
require 'io'

mcpr = {}
opt = opt or {verbose=false}

function mcpr.get_rot_matrix(theta) -- theta should be given in radian and counterclockwise
   if math.abs(theta) > 2*math.pi then io.write("\nmake sure theta is in radian mode") end
   return torch.Tensor({{math.cos(theta),-math.sin(theta)},{math.sin(theta),math.cos(theta)}})
end

function mcpr.calc_prob_matrix(h,w,theta,n_per_bin)
   local rot_matrix = mcpr.get_rot_matrix(theta) ---theta b/c we want to get ori_pt from pt
   local prob_matrix = torch.zeros(h,w,h,w)
   -- draw n sample randomly from -0.5,0.5 (symetric wrt 0 to make rotation correct)
   local n_per_bin = n_per_bin or 3000
   local n = n_per_bin*h*w 
   local pts = torch.rand(n,2) - 0.5 -- contigous memory for each pt
   -- find the origin of every points: try refactor this using matrix multiplication to run faster
   for i=1,n do
      local pt = pts[i];
      local ori_pt = rot_matrix*pt;
      if ori_pt[1]<0.5 and ori_pt[2]<0.5 and ori_pt[1]>-0.5 and ori_pt[2]>-0.5 then
	 -- original pt inside the box
	 -- note the bin of both pt and ori_pt: rotation is counter clockwise
	 local pt_bin = torch.Tensor({math.ceil((pt[2]+0.5)*h), math.ceil((pt[1]+0.5)*w)}); 
	 local ori_pt_bin = torch.Tensor({math.ceil((ori_pt[2]+0.5)*h), math.ceil((ori_pt[1]+0.5)*w)});
	 -- update the prob_matrix
	 prob_matrix[{ori_pt_bin[1], ori_pt_bin[2], pt_bin[1], pt_bin[2]}] = 
	    prob_matrix[{ori_pt_bin[1], ori_pt_bin[2], pt_bin[1], pt_bin[2]}] + 1;
      end
   end
   local prob_matrix = prob_matrix/n_per_bin;
   return prob_matrix
end

function mcpr.rotate_monte_carlo(fast, n)
   return function(filter, theta, prob_matrix, n)
      -- TODO: make this faster by viewing and expanding less!!!!!!!!!!!!!!!!!!!
      -- rotate a filter by theta radian counter clockwise
      -- TODO: make this function work with 2d,3d,4d, and 5d filter
      assert(filter:dim()==2 or filter:dim()==3 or filter:dim()==4 or filter:dim()==5)
      local rot_filter = torch.Tensor();
      if filter:dim()==2 then -- took 0.156s in test -- should never reach this branch
	 print('using filter dim 2 is deprecated, use filter dim 5 instead')
	 local h = filter:size(1); local w = filter:size(2)
	 prob_matrix = prob_matrix or print('\nprob matrix generated inside rotate_monte_carlo!') or mcpr.calc_prob_matrix(h,w,theta,n); 
	 rot_filter:resize(filter:size())
	 local h = filter:size(1); local w = filter:size(2)
	 for i=1,h do
	    for j=1,w do
	       rot_filter[{i,j}] = torch.sum(torch.cmul(filter,prob_matrix[{{},{},i,j}]));
	    end
	 end
	 -- faster implementation: see here https://github.com/torch/torch7/blob/master/doc/tensor.md
	 -- rot_filter = filter:resize(filter:size(1),filter:size(2),1,1):expand()
	 -- rot_filter = filter:repeatTensor(,1,1):sum() ? fixme!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      elseif filter:dim()==3 then
	 error('not implemented !!!')
      elseif filter:dim()==4 then
	 if not fast then
	    if opt.verbose then print('using not fast rotation') end
	    -- filter:    nOutputFilter,nInputFilter,kH,kW
	    -- expandTo:  nOutputFilter,nInputFilter,kH,       kW,        kH,       kW      (can view)
	    -- prob_matrix:                          oriBinH x oriBinW  x rotBinH x rotBinW (can view)
	    local h = filter:size(3); local w = filter:size(4)
	    prob_matrix = prob_matrix or print('\nprob matrix generated inside rotate_monte_carlo!') or mcpr.calc_prob_matrix(h,w,theta,n); 
	    local fExp = filter:view(filter:size(1),filter:size(2),filter:size(3),filter:size(4),1,1)
	       :expand(filter:size(1),filter:size(2),filter:size(3),filter:size(4),filter:size(3),filter:size(4))
	    local pMExp = prob_matrix:view(1,1,prob_matrix:size(1),prob_matrix:size(2),prob_matrix:size(3),prob_matrix:size(4))
	       :expand(filter:size(1),filter:size(2),filter:size(3),filter:size(4),filter:size(3),filter:size(4))
	    rot_filter = torch.cmul(fExp,pMExp):sum(3):sum(4):view(filter:size())
	 else
	    --- the following is 100 times faster, use this when time is critical
	    if opt.verbose then print('using fast rotation: not as exact as it can be') end
	    local f = filter:view(filter:size(1)*filter:size(2),filter:size(4),filter:size(5))
	    rot_filter = image.rotate(f,theta):view(filter:size())
	 end
      elseif filter:dim()==5 then -- used 0.018 s
	 if not fast then
	    if opt.verbose then print('using not fast rotation') end
	    -- filter:    nOutputFilter,nInputFilter,nInputAng,kH,kW
	    -- expandTo:  nOutputFilter,nInputFilter,nInputAng,kH,       kW,        kH,       kW      (can view)
	    -- prob_matrix:                                    oriBinH x oriBinW  x rotBinH x rotBinW (can view)
	    local h = filter:size(4); local w = filter:size(5)
	    prob_matrix = prob_matrix or print('\nprob matrix generated inside rotate_monte_carlo!') or mcpr.calc_prob_matrix(h,w,theta,n); 
	    local fExp = filter:view(filter:size(1),filter:size(2),filter:size(3),filter:size(4),filter:size(5),1,1)
	       :expand(filter:size(1),filter:size(2),filter:size(3),filter:size(4),filter:size(5),filter:size(4),filter:size(5))
	    local pMExp = prob_matrix:view(1,1,1,prob_matrix:size(1),prob_matrix:size(2),prob_matrix:size(3),prob_matrix:size(4))
	       :expand(filter:size(1),filter:size(2),filter:size(3),filter:size(4),filter:size(5),filter:size(4),filter:size(5))
	    rot_filter = torch.cmul(fExp,pMExp):sum(4):sum(5):view(filter:size())
	 else
	    --- the following is 100 times faster, use this when time is critical
	    if opt.verbose then print('using fast rotation: not as exact as it can be') end
	    local f = filter:view(filter:size(1)*filter:size(2)*filter:size(3),filter:size(4),filter:size(5))
	    rot_filter = image.rotate(f,theta):view(filter:size())
	 end
      end
      return rot_filter, prob_matrix
   end
end

function mcpr.circshift(A,k,dim)
   -- circular shift along dim for k steps
   assert(dim<=A:dim() and dim>0, 'wrong dimension')
   k = k % A:size(dim)
   if k==0 then return A end

   return torch.cat(A:narrow(dim,A:size(dim)-k+1,k):double(),A:narrow(dim,1,A:size(dim)-k):double(),dim):cuda()
end

function mcpr.circular_rotate(filter,theta,rotate_index,prob_matrix,circular,fast)
   -- circularly rotate a filter, circular is assume to be true
   -- filter is of size: nInputFilter,nInputAng,kH,kW
   --                or: nOutputFilter,nInputFilter,nInputAng,kH,kW
   -- fast: true of false to use build in rotation
   assert(filter:dim()==4 or filter:dim()==5)
   local rot_filter = torch.Tensor()
   if filter:dim()==4 then -- 0.16 s
      rot_filter:resize(filter:size())
      local nInputFilter=filter:size(1); local nInputAng=filter:size(2); 
      -- verbose version
      for i=1,nInputFilter do
	 for j=1,nInputAng do 
	    rot_filter[{i,j,{},{}}] = mcpr.rotate_monte_carlo(fast)(filter[{i,j,{},{}}], theta, prob_matrix);
	 end
      end
      if circular then rot_filter = mcpr.circshift(rot_filter,rotate_index,2) end
   else  -- used 0.019 s
      local nOutputFilter=filter:size(1); local nInputFilter=filter:size(2); local nInputAng=filter:size(3); 
      -- verbose version
      -- for i=1,nInputFilter do
      -- 	 for j=1,nInputAng do 
      -- 	    for k=1,nOutputFilter do
      -- 	       rot_filter[{k,i,j,{},{}}] = mcpr.rotate_monte_carlo(fast)(filter[{k,i,j,{},{}}], theta, prob_matrix) 
      -- 	    end
      -- 	 end
      -- end
      -- faster
      rot_filter = mcpr.rotate_monte_carlo(fast)(filter, theta, prob_matrix) 
      if circular then rot_filter = mcpr.circshift(rot_filter,rotate_index,3) end
   end
   return rot_filter
end

return mcpr
