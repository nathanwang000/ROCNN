-- This is the function that uses lenet 5
require 'package'
package.path = package.path .. ";../?.lua" -- add to path
require 'nn'
require 'ro_conv'
require 'ro_linear'
require 'ro_rigid'

local Name = 'lenet5_rot'

function createModel(opt)

   local model = nn.Sequential()
   opt = opt or {}
   opt.rpLoss = opt.rpLoss or "variance/sum(w^2)"
   opt.rp = opt.rp or 0
   -- stage 1: reshape input and rotate once and average pool
   model:add(nn.Reshape(1,1,28,28,true))
   model:add(nn.Roconv(1,8,1,6,5,5,1,1,false,0)) -- 6*8*24*24, params: 1*6*5*5
   model:add(nn.ReLU())
   model:add(nn.VolumetricAveragePooling(1,2,2)) -- 6*8*12*12, params: 0
   -- stage 2: rigid rotate filter
   model:add(nn.Rorigid(8,6,16,5,5,true,12,12)) -- 16*8*8*8, params: 6*16*5*5
   model:add(nn.ReLU())
   model:add(nn.Rorigid(8,16,16,5,5,true,8,8)) -- 16*8*4*4, params: 16*16*5*5
   model:add(nn.ReLU())
   -- fully connected layers
   model:add(nn.Rorigid(8,16,120,4,4,true,4,4)) -- 120*8*1*1
   model:add(nn.ReLU())
   model:add(nn.Rorigid(8,120,84,1,1,true,1,1)) -- 84*8*1*1
   model:add(nn.ReLU())
   model:add(nn.Rorigid(8,84,10,1,1,true,1,1)) -- 10*8*1*1
   model:add(nn.ReLU())
   -- stage 3: ro linear to determine angle
   local pl = nn.Parallel(2,2) -- first dim is batch dim
   for i=1,10 do
      local seq = nn.Sequential()
      seq:add(nn.Reshape(8,1,true))
      seq:add(nn.TemporalMaxPooling(8)) -- 1x1
      seq:add(nn.Reshape(1,true))
      pl:add(seq)
   end
   model:add(pl)
   
   return model
end

print(string.format('%s used %f gb memory', Name, collectgarbage("count")/1024))

return createModel()



