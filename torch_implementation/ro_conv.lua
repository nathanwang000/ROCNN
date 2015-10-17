require 'paths'
require 'nn'
require 'monteCarloProbRotation' -- the module has a method mcpr
require 'math'
-- use test_roconv_hard.lua to test this module
local opt = opt or {verbose=false, debug=false}

local Roconv, Parent = torch.class('nn.Roconv', 'nn.Module')

function Roconv:__init(nInputAng,nOutputAng,nInputFilter,nOutputFilter,kH,kW,dH,dW,circular,rp,loss)

   Parent.__init(self)
   -- temporary buffers for unfolding (CUDA)
   self.finput = torch.Tensor()
   self.fgradInput = torch.Tensor()
   -- nInputAng: filter_depth
   -- nOutputAng: num_angles
   -- nInputFilter: num_channels
   -- nOutputFilter: num_filters
   -- kH: filter_height
   -- kW: filter_width
   -- dH: stepH
   -- dW: stepW
   -- circular: circular rotate or not
   -- rp: rotation prior
   nInputAng = nInputAng or 8
   nOutputAng = nOutputAng or 8
   dH = dH or 1
   dW = dW or 1
   -- hyper parameters
   self.nInputAng = nInputAng; self.kT = nInputAng; self.dT=1; -- for compatibility wit 3d conv
   self.nOutputAng = nOutputAng
   self.nInputFilter = nInputFilter; self.nInputPlane = nInputFilter -- for compatibility wit 3d conv
   self.nOutputFilter = nOutputFilter; self.nOutputPlane = nOutputFilter -- for compatibility wit 3d conv
   self.kH = kH
   self.kW = kW
   self.dH = dH
   self.dW = dW
   self.rp = rp or 0.005
   self.loss = loss or 'variance/sum(w^2)'
   -- generate or load probMat
   local basedir = os.getenv('ROCONV_BASEDIR') -- should be absolute path
   if not basedir then error("please config ROCONV_BASEDIR first before executing this file. exiting") end
   local dir = paths.concat(basedir,'data/')
   os.execute('mkdir -p ' .. dir)
   local saveName = string.format('probMat%dx%dx%d.dat',self.kH,self.kW,self.nOutputAng)
   saveName = paths.concat(dir,saveName)
   if paths.filep(saveName) then self.probMat = torch.load(saveName)
   else 
      self.probMat = torch.zeros(nOutputAng,kH,kW,kH,kW)
      for i=1,kH do
	 for j=1,kW do
	    self.probMat[1][{i,j,i,j}] = 1
	 end
      end
      for i=2,nOutputAng do
	 self.probMat[i] = mcpr.calc_prob_matrix(kH,kW,(i-1)/self.nOutputAng*2*math.pi)
      end
      torch.save(saveName,self.probMat) 
   end

   self.circular = circular -- default to nil, false
   assert(circular and true or (nInputAng==1)) -- make sure when no circrot, no input angle should present
   -- parameters: inherited self.output and self.gradInput
   self.W = torch.Tensor(nOutputFilter,nInputFilter,nInputAng,kH,kW)
   self.bias = torch.Tensor(nOutputFilter)
   self.gradW = torch.zeros(self.W:size()); self.gradWeight = torch.Tensor() -- for compatibility wit 3d conv
   self.gradB = torch.zeros(self.bias:size()); self.gradBias = torch.Tensor() -- for compatibility wit 3d conv
   self:reset()
   -- other parameters
   opt.verbose = false
   opt.debug = false
end

function Roconv:debug()
   print('=>enter debug mode: rotation prior disgarded for Roconv')
   opt.debug = true
end

function Roconv:verbose()
   opt.verbose = true
end

function Roconv:__tostring__()
   -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
   return torch.type(self) ..
      string.format('(%d->%d, %dx%d rf, %d->%d ang, %f rp, %s)', self.nInputFilter, self.nOutputFilter,
		    self.kH, self.kW, self.nInputAng, self.nOutputAng, self.rp, self.circular and 'circular' or 'not circular')
end

function Roconv:parameters()
   -- return parameters and gradient 
   if self.W and self.bias and self.gradW and self.gradB then
      return {self.W, self.bias}, {self.gradW, self.gradB}
   else error('parameters not setup correctly') end
end

function Roconv:updateOutput(input) -- reference https://github.com/torch/nn/blob/master/doc/convolution.md#nn.VolumetricConvolution
   assert(input:dim()==4 or input:dim()==5,'4D or 5D (batch-mode) tensor expected for input')
   local rotDim = input:dim()==4 and 2 or 3 -- dimesion of rotation
   -- input: [batch_size] x nInputFilter x time(nInputAng) x height x width
   -- output: [batch_size] x nOutputFilter x otime(nOutputAng) x owidth x oheight  ------------vs----------------
   -- output: out_height x out_width x nOutputAng x nOutputFilter x batch_size
   -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW      ------------vs----------------
   -- weight: kH x kW x nInputAng x nInputFilter x nOutputFilter
   self.weight = self.W
   local output = input.nn.VolumetricConvolution_updateOutput(self, input):clone()

   assert(output:size(rotDim)==1,string.format('the time field must be 1, now is %d', output:size(rotDim)))
   local old_weight
   if opt.verbose then old_weight = self.weight end
   for i=2,self.nOutputAng do 
      -- rotate W:
      self.weight = mcpr.circular_rotate(self.W,(i-1)/self.nOutputAng*2*math.pi,i-1,self.probMat[i],self.circular)
      if opt.verbose then
	 print(string.format('old weights equals new weights: %s', torch.all(torch.eq(self.weight,old_weight)) and "true" or "false"))
	 old_weight = self.weight
      end
      output = torch.cat(output,input.nn.VolumetricConvolution_updateOutput(self, input),rotDim)
   end
   self.output = output
   if opt.verbose then
      print('output is ')
      print(self.output)
   end
   return self.output
end

function Roconv:reset(ws,mode)
   -- ensure equivariance initial activation
   if mode then
      local ws = ws or 2/math.sqrt(self.kW*self.kH*self.nInputAng*self.nInputFilter)
      self.W = torch.randn(self.W:size()) * ws
      self.bias = torch.randn(self.bias:size()) * ws
   else
      ws = ws or 1/math.sqrt(self.kW*self.kH*self.nInputAng*self.nInputFilter)
      self.W:uniform(-ws,ws)
      self.bias:uniform(-ws,ws)
   end
   opt.debug = false
end


function Roconv:updateGradInput(input, gradOutput)
   assert(input:dim()==4 or input:dim()==5,'4D or 5D (batch-mode) tensor expected for input')
   local rotDim = input:dim()==4 and 2 or 3 -- dimesion of rotation
   -- input: [batch_size] x nInputPlane x time(nInputAng) x height x width
   -- output: [batch_size] x nOutputFilter x otime(nOutputAng) x owidth x oheight  ------------vs----------------
   -- output: out_height x out_width x nOutputAng x nOutputFilter x batch_size
   -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW      ------------vs----------------
   -- weight: kH x kW x nInputAng x nInputFilter x nOutputFilter
   -- gradInput: [batch_size] x nInputFilter x time(nInputAng) x height x width
   -- gradOutput: [batch_size] x nOutputFilter x otime(nOutputAng) x owidth x oheight
   self.weight = self.W
   if input:dim()==4 then
      local gI = input.nn.VolumetricConvolution_updateGradInput(self, input, gradOutput[{{},{1},{},{}}]):clone()
      for i=2,self.nOutputAng do 
	 -- rotate W:
	 self.weight = mcpr.circular_rotate(self.W,(i-1)/self.nOutputAng*2*math.pi,i-1,self.probMat[i],self.circular)
	 -- gI = gI + input.nn.VolumetricConvolution_updateGradInput(self, input, gradOutput[{{},{i},{},{}}])
	 gI:add(input.nn.VolumetricConvolution_updateGradInput(self, input, gradOutput[{{},{i},{},{}}]))
      end
      self.gradInput = gI
   else
      local gI = input.nn.VolumetricConvolution_updateGradInput(self, input, gradOutput[{{},{},{1},{},{}}]):clone() -- nOutputAng = 1         
      --print(gradOutput[{{},{},{1},{},{}}])
      for i=2,self.nOutputAng do 
	 -- rotate W:
	 self.weight = mcpr.circular_rotate(self.W,(i-1)/self.nOutputAng*2*math.pi,i-1,self.probMat[i],self.circular)
	 --print(gradOutput[{{},{},{i},{},{}}])
	 --gI = gI + input.nn.VolumetricConvolution_updateGradInput(self, input, gradOutput[{{},{},{i},{},{}}])
	 gI:add(input.nn.VolumetricConvolution_updateGradInput(self, input, gradOutput[{{},{},{i},{},{}}]))
      end
      self.gradInput = gI
   end
   return self.gradInput
end

function Roconv:accGradParameters(input, gradOutput, scale)
   assert(input:dim()==4 or input:dim()==5,'4D or 5D (batch-mode) tensor expected for input')
   local rotDim = input:dim()==4 and 2 or 3 -- dimesion of rotation
   scale = scale or 1
   -- input: [batch_size] x nInputPlane x time(nInputAng) x height x width
   -- output: [batch_size] x nOutputPlane x otime(nOutputAng) x owidth x oheight
   -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
   -- gradInput: [batch_size] x nInputPlane x time(nInputAng) x height x width
   -- gradOutput: [batch_size] x nOutputPlane x otime(nOutputAng) x owidth x oheight
   self.gradW:resizeAs(self.W)
   self.gradWeight = torch.zeros(self.weight:size()); 
   self.gradBias = torch.zeros(self.bias:size())

   if input:dim()==4 then

      input.nn.VolumetricConvolution_accGradParameters(self,input,gradOutput[{{},{1},{},{}}],scale)
      local gW = self.gradWeight:clone(); self.gradWeight:zero(); self.gradBias:zero();
      for i=2,self.nOutputAng do 
	 -- rotate W
	 self.weight = mcpr.circular_rotate(self.W,(i-1)/self.nOutputAng*2*math.pi,i-1,self.probMat[i],self.circular)
	 input.nn.VolumetricConvolution_accGradParameters(self, input, gradOutput[{{},{i},{},{}}], scale)
	 gW = torch.cat(gW,self.gradWeight,6); self.gradWeight:zero(); self.gradBias:zero();
      end
      -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
      -- gW:      nOutputFilter x nInputFilter x nInputAng x [kH      x kW       x] nOutputAng
      -- probMat: nOutputAng    x                            [oriBinH x oriBinW  x] rotBinH x rotBinW
      ----------------------
      -- want:   Expand probMat[{},i,j,{},{}]
      --         nOutputAng x                      rotBinH x rotBinW 
      --         To gW:size()
      --         nOutputFilter x nInputFilter x nInputAng x kH      x kW       x nOutputAng
      -- then cmul gW and sum over (kH, kW, nOutputAng)
      for i=1,self.weight:size(4) do -- kH
	 for j=1,self.weight:size(5) do -- kW
	    local pM = self.probMat[{{},i,j,{},{}}]:permute(2,3,1)
	    pM = pM:reshape(1,1,1,pM:size(1),pM:size(2),pM:size(3)) -- cp memory
	    local pMExpand = pM:expand(gW:size(1),gW:size(2),gW:size(3),pM:size(4),pM:size(5),pM:size(6))
	    if self.circular then
	       local tmp = torch.cmul(pMExpand,gW):sum(4):sum(5)
	       -- tmp: nOutputFilter x nInputFilter x nInputAng x 1 x 1 x nOutputAng
	       for r=2,self.nOutputAng do
		  tmp[{{},{},{},1,1,r}] = mcpr.circshift(tmp[{{},{},{},1,1,r}],-r+1,3)
	       end
	       self.gradW[{{},{},{},i,j}]:add(scale,tmp:sum(6):view(gW:size(1),gW:size(2),gW:size(3)))
	    else
	       self.gradW[{{},{},{},i,j}]:add(scale,torch.cmul(pMExpand,gW):sum(4):sum(5):sum(6):view(gW:size(1),gW:size(2),gW:size(3)))
	    end
	 end
      end
   else -- with optional batch size case: 
      -- gradOutput: [batch_size] x nOutputPlane x otime(nOutputAng) x owidth x oheight
      -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
      self.weight = self.W -- may not need
      input.nn.VolumetricConvolution_accGradParameters(self,input,gradOutput[{{},{},{1},{},{}}],scale)
      local gW = self.gradWeight:clone(); self.gradWeight:zero(); self.gradBias:zero();
      for i=2,self.nOutputAng do 
	 -- rotate W:
	 self.weight = mcpr.circular_rotate(self.W,(i-1)/self.nOutputAng*2*math.pi,i-1,self.probMat[i],self.circular) -- may not need
	 input.nn.VolumetricConvolution_accGradParameters(self, input, gradOutput[{{},{},{i},{},{}}], scale)
	 gW = torch.cat(gW,self.gradWeight,6); self.gradWeight:zero(); self.gradBias:zero();
      end
      -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
      -- gW:      nOutputFilter x nInputFilter x nInputAng x [kH      x kW       x] nOutputAng
      -- probMat: nOutputAng    x                            [oriBinH x oriBinW  x] rotBinH x rotBinW
      ----------------------
      -- want:   Expand probMat[{},i,j,{},{}]
      --         nOutputAng x                      rotBinH x rotBinW 
      --         To gW:size()
      --         nOutputFilter x nInputFilter x nInputAng x kH      x kW       x nOutputAng
      -- then cmul gW and sum over (kH, kW, nOutputAng)
      for i=1,self.weight:size(4) do -- kH
	 for j=1,self.weight:size(5) do -- kW
	    local pM = self.probMat[{{},i,j,{},{}}]:permute(2,3,1)
	    pM = pM:reshape(1,1,1,pM:size(1),pM:size(2),pM:size(3)) -- cp memory
	    local pMExpand = pM:expand(gW:size(1),gW:size(2),gW:size(3),pM:size(4),pM:size(5),pM:size(6))
	    if self.circular then
	       local tmp = torch.cmul(pMExpand,gW):sum(4):sum(5)
	       -- tmp: nOutputFilter x nInputFilter x nInputAng x 1 x 1 x nOutputAng
	       for r=2,self.nOutputAng do
		  tmp[{{},{},{},1,1,r}] = mcpr.circshift(tmp[{{},{},{},1,1,r}],-r+1,3)
	       end
	       self.gradW[{{},{},{},i,j}]:add(scale,tmp:sum(6):view(gW:size(1),gW:size(2),gW:size(3))) ------------------------------------check scale ccould be a bug!!!
	    else
	       self.gradW[{{},{},{},i,j}]:add(scale,torch.cmul(pMExpand,gW):sum(4):sum(5):sum(6):view(gW:size(1),gW:size(2),gW:size(3)))
	    end
	 end
      end
   end
   -- gradOutput: [batch_size] x nOutputPlane x otime(nOutputAng) x owidth x oheight
   self.gradB:resize(self.bias:size())
   for i=1,self.nOutputFilter do
      self.gradB[i] = torch.sum(gradOutput:select(rotDim-1,i))
   end

   if self.circular and not opt.debug then -- only update for self.W
      -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
      -- output: [batch_size] x nOutputFilter x nOutputAng x owidth x oheight 
      local W = self.W; local rp = self.rp
      local nInputAng = W:size(3)
      if self.loss == 'variance' then
      	    local dr_dw = (W-(W:sum(3)/nInputAng):expand(W:size()))
	    	    * (rp*2/nInputAng)
      	    print(string.format('update ratio for %s and SGD for Roconv is %f',self.loss,dr_dw:norm()/self.gradW:norm()))
      	    self.gradW:add(dr_dw)
      elseif self.loss == 'variance/sum(w^2)' then
      	    local sumW = W:sum(3):expand(W:size())
	    local sqSumW = torch.cmul(sumW,sumW)
	    local sumSqW = torch.cmul(W,W):sum(3):expand(W:size())
      	    local dr_dw = (sumSqW==0) and 0 or
	    	  (torch.cmul(W,sqSumW) - torch.cmul(sumW,sumSqW)):cdiv(torch.cmul(sumSqW,sumSqW))
	    	    * (rp*2/(nInputAng^2))
      	    print(string.format('update ratio for %s and SGD for Roconv is %f',self.loss,dr_dw:norm()/self.gradW:norm()))
      	    self.gradW:add(dr_dw)
      else error(string.format('%s loss is not defined',self.loss)) end
   end

end

