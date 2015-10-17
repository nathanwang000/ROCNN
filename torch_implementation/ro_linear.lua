--[[
   The purpose of this module is to add rp on the module following Roconv layer
--]]
require 'nn'

local Rolinear, Parent = torch.class('nn.Rolinear', 'nn.Linear')

function Rolinear:__init(inputSize,outputSize,nInputFilter,nInputAng,kH,kW,rp,loss)
   Parent.__init(self,inputSize,outputSize)
   -- weight:   outputSize x inputSize
   -- changeTO: nOutputFilter x nInputFilter x nInputAng x kH x kW
   self.nInputFilter = nInputFilter
   self.nInputAng = nInputAng
   self.kH = kH
   self.kW = kW
   self.rp = rp or 0.005
   self.loss = loss or 'variance/sum(w^2)'
   self.isDebug = false
   assert(nInputFilter*nInputAng*kH*kW == inputSize)
end

function Rolinear:reset()
   Parent.reset(self)
   self.isDebug = false
end

function Rolinear:debug()
   print('=>enter debug mode: rotation prior disgarded for Rolinear')
   self.isDebug = true
end

function Rolinear:accGradParameters(input, gradOutput, scale)
   Parent.accGradParameters(self,input,gradOutput,scale)
   if not self.isDebug then -- only update for self.W
      -- weight: nOutputFilter x nInputFilter x nInputAng x kH x kW
      -- output: [batch_size] x nOutputFilter x nOutputAng x owidth x oheight 
      local W = self.weight:view(self.weight:size(1),self.nInputFilter,self.nInputAng,
				 self.kH,self.kW); local rp = self.rp
      local nInputAng = W:size(3)
      if self.loss == 'variance' then
            local dr_dw = (W-(W:sum(3)/nInputAng):expand(W:size())) * (rp*2/nInputAng)
      	    print(string.format('update ratio for %s and SGD for Rolinear is %f',self.loss,dr_dw:norm()/self.gradWeight:norm()))
      	    self.gradWeight = (self.gradWeight:view(self.weight:size(1),self.nInputFilter,self.nInputAng,
					      self.kH,self.kW):add(dr_dw)):view(self.weight:size())
      elseif self.loss == 'variance/sum(w^2)' then
      	    local sumW = W:sum(3):expand(W:size())
	    local sqSumW = torch.cmul(sumW,sumW)
	    local sumSqW = torch.cmul(W,W):sum(3):expand(W:size())
      	    local dr_dw = (sumSqW==0) and 0 or
	    	  (torch.cmul(W,sqSumW) - torch.cmul(sumW,sumSqW)):cdiv(torch.cmul(sumSqW,sumSqW))
	    	    * (rp*2/(nInputAng^2))
      	    print(string.format('update ratio for %s and SGD for Rolinear is %f',self.loss,dr_dw:norm()/self.gradWeight:norm()))
      	    self.gradWeight = (self.gradWeight:view(self.weight:size(1),self.nInputFilter,self.nInputAng,
					      self.kH,self.kW):add(dr_dw)):view(self.weight:size())
      else error(string.format('%s loss is not defined',self.loss)) end
   end
end

function Rolinear:__tostring__()
   return torch.type(self) ..
      string.format('(%d -> %d, %f rp)', self.weight:size(2), self.weight:size(1), self.rp)
end
