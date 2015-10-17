--[[
   The thin layer or the type A module
--]]
require 'nn'
require 'math'

local Rorigid, Parent = torch.class('nn.Rorigid', 'nn.Module')

function Rorigid:__init(nInputAng,nInputFilter,nOutputFilter,kH,kW,isBatchmode,height,width,dH,dW,padW,padH)
   Parent.__init(self)
   -- nInputAng: filter_depth
   -- nOutputAng: num_angles
   -- nInputFilter: num_channels
   -- nOutputFilter: num_filters
   -- kH: filter_height
   -- kW: filter_width
   -- dH: stepH
   -- dW: stepW
   -- padH: pad in height direction
   -- padW: pad in width direction
   -- batchmode: batchmode or not
   -- width: input width
   -- height: input height
   isBatchmode = isBatchmode and true or false
   nInputAng = nInputAng or 8
   assert(nInputAng >= 1, "nInputAng should be positive")
   nOutputAng = nInputAng; -- for obvious reason
   dW = dW or 1
   dH = dH or 1
   padW = padW or 0
   padH = padH or 0
   -- hyperparams setting
   self.isBatchmode = isBatchmode
   -- for compatibility wit 2d conv
   self.nInputAng = nInputAng
   self.nOutputAng = nOutputAng
   self.nInputFilter = nInputFilter; self.nInputPlane = nInputFilter
   self.nOutputFilter = nOutputFilter; self.nOutputPlane = nOutputFilter
   self.kH = kH
   self.kW = kW
   self.dH = dH
   self.dW = dW
   self.padH = padH
   self.padW = padW
   self.width = width
   self.height = height
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
   -- parameters: inherited self.output and self.gradInput
   self.W = torch.Tensor(nOutputFilter,nInputFilter,kH,kW)
   self.bias = torch.Tensor(nOutputFilter)
   -- for compatibility wit 2d conv
   self.gradW = torch.zeros(self.W:size())
   self.gradB = torch.Tensor(self.bias:size())
   -- the helping modules
   -- input: [batch_size] x nInputFilter x nInputAng x height x width
   -- output: [batch_size] x nOutputFilter x nOutputAng x owidth x oheight
   local rotDim = isBatchmode and 3 or 2
   self.module = nn.Parallel(rotDim,rotDim)
   local owidth = math.floor((width  + 2*padW - kW) / dW + 1)
   local oheight = math.floor((height + 2*padH - kH) / dH + 1)
   for i=1,nOutputAng do
      -- add individual workers with weight and bias sharing
      local subContainer = nn.Sequential()
      subContainer:add(nn.SpatialConvolution(self.nInputPlane,self.nOutputPlane,
					     self.kW,self.kH,self.dW,self.dH,self.padW,self.padH))
      subContainer:add(nn.Reshape(nOutputFilter,1,owidth,oheight,isBatchmode))
      self.module:add(subContainer)
   end
   self:reset()

end

function Rorigid:__tostring__()
   -- weight: nOutputFilter x nInputFilter x kH x kW
   return torch.type(self) ..
      string.format('(%d->%d, %dx%d rf, %d ang)', self.nInputFilter, self.nOutputFilter,
		    self.kH, self.kW, self.nOutputAng)
end

function Rorigid:debug()
   -- empty function to comply with tests
end

function Rorigid:zeroGradParameters()
   Parent:zeroGradParameters()
   self.module:zeroGradParameters()
end

-- not using share bias because they should change simultaneously
function Rorigid:alignParams()
   -- here the params are weight and bias
   self.module:get(1):get(1).weight = self.W:clone()
   self.module:get(1):get(1).bias = self.bias:clone()
   for i=2,self.nOutputAng do
      -- rotate W:
      self.module:get(i):get(1).weight = mcpr.circular_rotate(self.W,(i-1)/self.nOutputAng*2*math.pi,i-1,self.probMat[i],false):clone()
      self.module:get(i):get(1).bias = self.bias:clone()
   end

end

function Rorigid:reset()
   self.module:reset()
   self.bias = self.module:get(1):get(1).bias:clone()
   self.W = self.module:get(1):get(1).weight:clone()
   self:alignParams()
end

function Rorigid:updateOutput(input)
   assert(input:dim()==4 or input:dim()==5,
	  '4D or 5D (batch-mode) tensor expected for input')
   -- input: [batch_size] x nInputFilter x nInputAng x height x width
   -- output: [batch_size] x nOutputFilter x nOutputAng x oheight x owidth
   -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
   local rotDim = input:dim()==4 and 2 or 3 -- dimesion of rotation
   assert(self.isBatchmode and rotDim==3 or rotDim==2,
	  'isBatchmode error, check definition of ro_rigid module')
   assert(self.width==input:size(rotDim+2) and self.height==input:size(rotDim+1),"check your width and height")
   assert(self.nInputAng==input:size(rotDim),'input angle not aligned, bad input')
   self:alignParams()
   self.module:updateOutput(input)
   self.output = self.module.output
   return self.output
end

function Rorigid:updateGradInput(input, gradOutput)
   assert(input:dim()==4 or input:dim()==5,
	  '4D or 5D (batch-mode) tensor expected for input')   
   local rotDim = input:dim()==4 and 2 or 3 -- dimesion of rotation
   assert(self.isBatchmode and rotDim==3 or rotDim==2,
	  'isBatchmode error, check definition of ro_rigid module')
   assert(self.width==input:size(rotDim+2) and self.height==input:size(rotDim+1),"check your width and height")
   assert(self.nInputAng==input:size(rotDim),'input angle not aligned, bad input')
   self:alignParams()
   self.gradInput = self.module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Rorigid:accGradParameters(input, gradOutput, scale)
   assert(input:dim()==4 or input:dim()==5,
	  '4D or 5D (batch-mode) tensor expected for input')   
   local rotDim = input:dim()==4 and 2 or 3 -- dimesion of rotation
   assert(self.isBatchmode and rotDim==3 or rotDim==2,
	  'isBatchmode error, check definition of ro_rigid module')
   assert(self.width==input:size(rotDim+2) and self.height==input:size(rotDim+1),"check your width and height")
   assert(self.nInputAng==input:size(rotDim),'input angle not aligned, bad input')
   self:alignParams()
   scale = scale or 1
   -- actual stuff begin
   self.module:accGradParameters(input, gradOutput, scale)
   -- gradB is the sum of all module:get(i):get(1).gradBias
   -- gradB is the transformed sum of all module:get(i):get(1).gradWeight
   self.gradB = self.module:get(1):get(1).gradBias

   self.gradW = self.module:get(1):get(1).gradWeight
   for k=2,self.nOutputAng do
      self.gradB:add(self.module:get(k):get(1).gradBias)
      -- self.probMat: nOutputAng x oriBinH x oriBinW x rotBinH x rotBinW
      -- weight: nOutputFilter x nInputFilter x kH x kW
      local gradWeight = self.module:get(k):get(1).gradWeight
      for i=1,self.kW do
	 for j=1,self.kH do
	    -- set self.graW
	    local pM = self.probMat[{k,i,j,{},{}}]:view(1,1,self.kH,self.kW):expand(self.nOutputFilter,self.nInputFilter,self.kH,self.kW)
	    self.gradW[{{},{},i,j}]:add(scale,torch.cmul(pM,gradWeight):sum(3):sum(4))
	 end
      end
      
   end


   
end

function Rorigid:parameters()
   -- return parameters and gradient 
   if self.W and self.bias and self.gradW and self.gradB then
      return {self.W, self.bias}, {self.gradW, self.gradB}
   else error('parameters not setup correctly') end
end
