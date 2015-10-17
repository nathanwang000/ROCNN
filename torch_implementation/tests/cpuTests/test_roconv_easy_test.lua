require 'package'
package.path = package.path .. ";../../?.lua" -- add to path
require 'nn'
require 'ro_conv'
require 'ro_linear'
require 'image'

------------ debug call -------------------
local calls, total, this = {}, {}, {}
debug.sethook(
   function(event)
      local i = debug.getinfo(2, "Sln")
      if i.what ~= 'Lua' then return end
      local func = i.name or (i.source..':'..i.linedefined)
      if event == 'call' then
	 this[func] = os.clock()
      else
	 local time = os.clock() - this[func]
	 total[func] = (total[func] or 0) + time
	 calls[func] = (calls[func] or 0) + 1
      end
   end, "cr")

--- code to debug ---
criterion = nn.MSECriterion()

function ro_test0() -- non batch mode test circular rotate
   -- make up some input: size 1x1x8x10x10
   local input = torch.range(1,8*100):view(1,8,10,10)
   local target = torch.range(1,8):view(1,8,1,1) -- let's do a regression problem
   -- Roconv:__init(nInputAng,nOutputAng,nInputFilter,nOutputFilter,kH,kW,dH,dW,circular)
   local model = nn.Roconv(8,8,1,1,10,10,1,1,true) --output: 1x8x1x1
   --model:verbose()
   -- forward
   local out = model:forward(input)
   local loss = criterion:forward(out,target)
   print(string.format('loss for rotest0 is %f',loss))
   local df_do = criterion:backward(out,target)
   local gradInput = model:backward(input,df_do)
   --print (gradInput:size())
   --print(input:size())
end   

function ro_test1() -- non batch mode test
   -- make up some input: size 1x1x1x10x10
   local input = torch.range(1,100):view(1,1,10,10)
   local target = torch.range(1,8):view(1,8,1,1) -- let's do a regression problem
   -- Roconv:__init(nInputAng,nOutputAng,nInputFilter,nOutputFilter,kH,kW,dH,dW,circular)
   local model = nn.Roconv(1,8,1,1,10,10,1,1,false) --output: 1x8x1x1
   -- forward
   local out = model:forward(input)
   local loss = criterion:forward(out,target)
   print(string.format('loss for rotest1 is %f',loss))
   local df_do = criterion:backward(out,target)
   model:backward(input,df_do)
end   

function ro_test2() -- batch mode test
   -- make up some input: size 1x1x1x10x10
   local input = torch.range(1,100):view(1,1,1,10,10)
   local target = torch.range(1,8):view(1,1,8,1,1) -- let's do a regression problem
   -- Roconv:__init(nInputAng,nOutputAng,nInputFilter,nOutputFilter,kH,kW,dH,dW,circular)
   local model = nn.Roconv(1,8,1,1,10,10,1,1,false) --output: 1x1x8x1x1
   -- forward
   local out = model:forward(input)
   --print(out:size())
   local loss = criterion:forward(out,target)
   print(string.format('loss for rotest2 is %f',loss))
   local df_do = criterion:backward(out,target)
   model:backward(input,df_do)
end

function ro_test3() -- combined model test for non circular rotation layer
   -- make up some input: size 1x1x1x10x10
   local input = torch.range(1,90):view(1,1,1,10,9)
   local target = torch.Tensor({2}):view(1,1) -- let's do a regression problem
   -- create a easy network: without circular rotate
   local model = nn.Sequential() -- Roconv:__init(nInputAng,nOutputAng,nInputFilter,nOutputFilter,kH,kW,dH,dW,circular)
   local n1 = nn.Roconv(1,8,1,1,5,5,1,1,false)
   model:add(n1) -- output: 1x1x8x6x6
   local n2 = nn.Reshape(8*6*6,true)
   model:add(n2)
   local n3 = nn.Linear(8*6*6,1)
   model:add(n3) -- output: 1x1
   -- forward
   local output = model:forward(input)
   local loss = criterion:forward(output,target)
   print(string.format('loss for rotest3 is %f',loss))
   -- backward
   local df_do = criterion:backward(output,target)
   model:backward(input,df_do)
end

function ro_test4() -- combined model test for circular rotation, batch mode
   local bs = 2
   local input = torch.range(1,bs*28*28*1):view(bs,28*28)--:view(bs,1,1,28,28)
   local target = torch.Tensor({1,2})
   local model = nn.Sequential()
   model:add(nn.Reshape(1,1,28,28,true))
   model:add(nn.Roconv(1,8,1,20,11,11,1,1,false))
   model:add(nn.ReLU())
   model:add(nn.VolumetricAveragePooling(1,2,2))
   model:add(nn.Roconv(8,8,20,10,9,9,1,1,true)) -- bsx10x8x1x1
   model:add(nn.ReLU())
   local pl = nn.Parallel(2,2)
   for i=1,10 do
      local seq = nn.Sequential()
      seq:add(nn.Reshape(8,true))
      seq:add(nn.Rolinear(8,1,1,8,1,1))
      pl:add(seq)
   end
   model:add(pl)
   model:add(nn.LogSoftMax())
   local criterion = nn.ClassNLLCriterion()
   -- forward
   local output = model:forward(input)
   local loss = criterion:forward(output,target)
   print(string.format('loss for rotest4 is %f',loss))
   -- backward
   local df_do = criterion:backward(output,target)
   model:backward(input,df_do)
end
--ro_test0()
--ro_test1()
--ro_test2()
ro_test3()
--ro_test4()
--- debug end ----
debug.sethook()

-- print the results
for f,time in pairs(total) do
   print(("Function %s took %.3f seconds after %d calls"):format(f, time, calls[f]))
end
