require 'package'
package.path = package.path .. ";../../?.lua" -- add to path

require 'nn'
require 'ro_conv'
require 'ro_linear'
require 'image'

iOri = image.lena()
bs = 1
i = image.scale(iOri,28,28)[1]:view(1,1,1,28,28):expand(bs,1,1,28,28)

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
RoconvNet = nn.Sequential()
-- Roconv:__init(nInputAng,nOutputAng,nInputFilter,nOutputFilter,kH,kW,dH,dW,circular)
n1 = nn.Roconv(1,8,1,20,11,11,1,1,false)
re1 = nn.ReLU()
-- nn.VolumetricMaxPooling(kT,kW,kH) -- see https://github.com/torch/nn/blob/master/doc/convolution.md#nn.VolumetricAveragePooling
p1 = nn.VolumetricAveragePooling(1,2,2)
n2 = nn.Roconv(8,8,20,10,9,9,1,1,true) -- bsx10x8x1x1
re2 = nn.ReLU()
-- ro_linear layer is just a parrallel linear layer
-- module = Parallel(inputDimension,outputDimension)
pl1 = nn.Parallel(2,2) -- output should be 10
for i=1,10 do 
   local seq = nn.Sequential() -- input is bsx1x8x1x1
   seq:add(nn.Reshape(8,true)) -- use View if applicable
   seq:add(nn.Rolinear(8,1,1,8,1,1))
   pl1:add(seq)
end
-- softmax layer
sm1 = nn.SoftMax()

n1out = n1:forward(i)
re1out = re1:forward(n1out)
p1out = p1:forward(re1out) -- bsx20x8x9x9
n2out = n2:forward(p1out)
re2out = re2:forward(n2out)
pl1out = pl1:forward(n2out)

-- gradient check http://code.madbits.com/wiki/doku.php?id=tutorial_morestuff
function gradCheck(m,input) -- module input
   local precision = 1e-5
   local jac = nn.Jacobian
   local err = jac.testJacobian(m,input)
   print('==> error: ' .. err)
   if err<precision then
      print('==> module OK')
   else
      print('==> error too large, incorrect implementation')
   end
end
print('testing for n1')
gradCheck(n1,i)
print('testing for n2')
gradCheck(n2,p1out)
print('testing for pl1')
gradCheck(pl1,n2out)
--- debug end ----
debug.sethook()

-- print the results
-- for f,time in pairs(total) do
--    print(("Function %s took %.3f seconds after %d calls"):format(f, time, calls[f]))
-- end
