-- This is the function that uses DSN_roconv
require 'package'
package.path = package.path .. ";../?.lua" -- add to path
require 'nn'
require 'ro_conv'
require 'ro_linear'
require 'ro_rigid'

local Name = 'DSN_roconv'

function createModel(opt)

   local model = nn.Sequential()
   opt = opt or {}
   opt.rpLoss = opt.rpLoss or "variance/sum(w^2)"
   opt.rp = opt.rp or 0
   -- stage 1: reshape input and rotate once and average pool
   model:add(nn.Reshape(1,1,28,28,true))
   model:add(nn.Roconv(1,8,1,20,11,11,1,1,false,opt.rpLastLayer and 0 or opt.rp,opt.rpLoss)) -- expect 4d input
   model:add(nn.ReLU())
   model:add(nn.VolumetricAveragePooling(1,2,2))
   -- stage 2: circular rotate filter
   model:add(nn.Roconv(8,8,20,10,9,9,1,1,true,opt.rpLastLayer and 0 or opt.rp,opt.rpLoss)) -- bsx10x8x1x1    
   model:add(nn.ReLU())
   -- stage 3: ro linear to determine angle
   local pl = nn.Parallel(2,2)
   for i=1,10 do
      local seq = nn.Sequential()
      seq:add(nn.Reshape(8,true))
      seq:add(nn.Rolinear(8,1,1,8,1,1,opt.rp,opt.rpLoss))
      pl:add(seq)
   end
   model:add(pl)

   return model
end

print(string.format('%s used %f gb memory', Name, collectgarbage("count")/1024))
return createModel()
