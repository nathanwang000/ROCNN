require 'torch'
require 'math'
require 'nn'
--require 'nnx'
require 'optim'
--require 'image'
--require 'pl'
require 'modelCleaning'
require 'paths'
require 'dataset-mnistRot'
require 'ro_linear'
require 'ro_conv'

local opt = lapp[[
    -n,--network       (default "")          reload pretrained model
]]

if opt.network == '' then error("plz specify a model by setting the -n flag with the path to the model")
else 
   print('<tester> using trained network ' .. opt.network)
   model = torch.load(opt.network)
end

testData = mnistRot.loadTestSet(tesize, false) -- maxLoad and display
testData:normalizeGlobal(mean, std)
confusion = optim.ConfusionMatrix(classes)

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)
      local f = criterion:forward(preds, targets)
      print(string.format("te loss was %f",f))

      -- confusion:
      for i = 1,opt.batchSize do
         confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

print('testing starts:')
test(testData)