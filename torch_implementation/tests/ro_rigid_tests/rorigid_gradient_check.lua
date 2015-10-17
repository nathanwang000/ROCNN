require 'package'
package.path = package.path .. ";../../?.lua" -- add to path

require 'ro_conv'
require 'ro_linear'
require 'ro_rigid'
require 'nn'
require 'math'
local jac = nn.Jacobian

local RorigidTests = {}

local function generateTest(testName,mod,input) 
   RorigidTests[testName] = function()
      mod:debug() -- use debug mode so that rp is never used
      print(mod)
      local f = jac.forward(mod,input)
      local b = jac.backward(mod,input)
      local fW = jac.forward(mod,input,mod.W)
      local bW = jac.backward(mod,input,mod.W,mod.gradW)
      local fB = jac.forward(mod,input,mod.bias)
      local bB = jac.backward(mod,input,mod.bias,mod.gradB)

      local errIn = (b-f):abs():max()
      local errW = (bW-fW):abs():max()
      local errB = (bB-fB):abs():max()
      print(string.format('error input is %f',errIn))
      print(string.format('error weight is %f',errW))
      print(string.format('error bias is %f',errB))
      
      local errCode = errPrint('dv_input',errIn,f,b) +
	 errPrint('dv_W',errW,fW,bW) +
	 errPrint('dv_b',errB,fB,bB)
      
      return errCode,mod
   end
end

function errPrint(errName,err,f,b,tolerence)
   tolerence = tolerence or 1e-6
   if err > tolerence then
      print(errName .. ' is wrong')
      print('forward output:')
      print(f)
      print('backward output:')
      print(b)
      return 1 -- report error
   end
   return 0 -- no error
end

function runTests()
   local passed = 0
   local count = 0
   for k,v in pairs(RorigidTests) do
      print('test ' .. k)
      local err = v()
      passed = passed + (err==0 and 1 or 0)
      print('')
      count = count + 1
   end
   print(string.format('passed %d/%d tests',passed,count))
end
------------ test generation
generateTest('[Rorigid]BatchNoRotationTest',
	     nn.Rorigid(1,2,2,2,2,true,2,2), -- nInputAng,nInputFilter,nOutputFilter,kH,kW,isBatchmode,height,width
	     torch.range(1,10*2*1*2*2):view(10,2,1,2,2)-- input: [batch_size] x nInputFilter x nInputAng x height x width 
)
generateTest('[Rorigid]nonBatchNoRotationTest',
	     nn.Rorigid(1,2,2,2,2,false,2,2), -- nInputAng,nInputFilter,nOutputFilter,kH,kW,isBatchmode,height,width
	     torch.range(1,2*1*2*2):view(2,1,2,2)-- input: [batch_size] x nInputFilter x nInputAng x height x width 
)
generateTest('[Rorigid]BatchRotationTest',
 	     nn.Rorigid(3,2,1,2,2,true,2,2), -- nInputAng,nInputFilter,nOutputFilter,kH,kW,isBatchmode,height,width
 	     torch.range(1,10*2*3*2*2):view(10,2,3,2,2) -- input: [batch_size] x nInputFilter x nInputAng x height x width 
)
generateTest('[Rorigid]nonBatchRotationTest',
 	     nn.Rorigid(3,2,1,2,2,false,2,2), -- nInputAng,nInputFilter,nOutputFilter,kH,kW,isBatchmode,height,width
 	     torch.range(1,2*3*2*2):view(2,3,2,2) -- input: [batch_size] x nInputFilter x nInputAng x height x width 
)

------------ run tests
runTests()


