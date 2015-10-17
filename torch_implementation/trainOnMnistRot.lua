-- adapted from Clement Farabet https://github.com/torch/demos/blob/master/train-a-digit-classifier/train-on-mnist.lua

require 'package'
require 'torch'
require 'math'
require 'nn'
require 'optim'
require 'modelCleaning'
require 'paths'
require 'dataset-mnistRot'
require 'ro_linear'
require 'ro_conv'
package.path = package.path .. ";modelZoo/?.lua"

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--save          (default "logs")      subdirectory to save logs
   -n,--network       (default "")          reload pretrained network
   --model            (default "lenetF")       file path to model to train inside modelZoo: eg. DSN, lenetF
   -f,--full                                use the full dataset
   -p,--plot                                plot while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.01)        learning rate, for SGD only
   -b,--batchSize     (default 100)         batch size
   -m,--momentum      (default 0.0)         momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0.0005)      L2 penalty on the weights, other words weight decay
   --seed                                   seed for randomization
   -t,--threads       (default 1)           number of threads
   --gpu                                    use gpu training
   -v,--val                                 use validation for testing
   --rp               (default 0.005)       rotational prior (only for roconv)
   --rpLoss           (default "variance/sum(w^2)")  loss used for rotation prior: variance | variance/sum(w^2)
   --rpLastLayer                            rp only apply on last layer
   --lrd              (default 5e-5)        learning rate decay
   --trsize	      (default 100)	    takes effect only when full is false
   --tesize	      (default 100)	    takes effect only when full is false
]]

--[[ fix seed --]]
if opt.seed then
   torch.manualSeed(opt.seed)
end

-- print state to file
print(string.format('Model saved in: %s', opt.save))
print(string.format('LearningRate: %f', opt.learningRate))
print(string.format('Momentum: %f', opt.momentum))
print(string.format('LearningRateDecay: %f', opt.lrd))
print(string.format('WeightDecay: %f', opt.coefL2))
print(string.format('BatchSize: %d', opt.batchSize))
print(string.format('RotationPrior: %f', opt.rp))

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

-- batch size
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 10-class classification problem
--
classes = {'0','1','2','3','4','5','6','7','8','9'}

-- geometry: width and height of input images
geometry = {28,28}

if opt.network == '' then
   -- define model to train
   local modelPath = 'modelZoo/' .. opt.model .. '.lua'
   if not paths.filep(modelPath) then
      error(string.format("unkonwn model %s", modelPath))
   end
   model = dofile(modelPath)

else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
   sgdState = torch.load(paths.concat(paths.dirname(opt.network), 'sgd.state'))
   -- model = nn.Sequential()
   -- model:read(torch.DiskFile(opt.network))
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<mnistRot> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

--[[ set default type --]]
if opt.gpu then
   print('<trainer> training with GPU')
   require 'cunn'
   model:cuda()
   -- we put the mlp in a new container:
   local model_gpu = nn.Sequential()
   model_gpu:add(nn.Copy('torch.DoubleTensor', 'torch.CudaTensor'))
   model_gpu:add(model)
   model_gpu:add(nn.Copy('torch.CudaTensor', 'torch.DoubleTensor'))
   model = model_gpu
end

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   trsize = opt.val and 10000 or 12000
   tesize = opt.val and 2000 or 50000
else
   trsize = opt.trsize
   tesize = opt.tesize
   print(string.format('<warning> only using %d samples to train quickly (use flag --full to use 12000 samples)',trsize))
end

assert(math.floor(trsize/opt.batchSize)*opt.batchSize==trsize and
	  math.floor(tesize/opt.batchSize)*opt.batchSize==tesize,
       "for convenience in training, try a batchsize that's multiple of training and testing size")

-- create training set
print 'read in training data'
trainData = mnistRot.loadTrainSet(trsize, false, 1) -- maxLoad and display

-- create test set and normalize
if opt.val then
   print 'read in validation data'
   testData = mnistRot.loadTrainSet(tesize, false, 10001) -- maxLoad and display
else
   print 'read int test data'
   testData = mnistRot.loadTestSet(tesize, false) -- maxLoad and display
end

-- preprocessing
trainData:normalizeGlobal(mean, std)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- randomize
   perm = torch.randperm(dataset:size())

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,1,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
      -- 	 -- load new sample
      --    -- local sample = dataset[i]
      --    -- local input = sample[1]:clone()
      --    -- local _,target = sample[2]:clone():max(1)
      --    -- target = target:squeeze()
      --    -- inputs[k] = input
      --    -- targets[k] = target
       	 inputs[k] = dataset.data[perm[i]]
       	 targets[k] = dataset.labels[perm[i]]
         k = k + 1
      end

      inputs = dataset.data[{{t,t+opt.batchSize-1}}]
      targets = dataset.labels[{{t,t+opt.batchSize-1}}]

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)
	 print(string.format("tr loss was %f",f))

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
				    }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = opt.lrd
				}
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- save/log current net
   local filename
   local sgdStateFN
   if opt.network=="" then
      filename = paths.concat(opt.save, string.format('%s%s',opt.model,'.net'))
      sgdStateFN = paths.concat(opt.save, 'sgd.state')
   else  
      filename = opt.network 
      dir = paths.dirname(opt.network)
      sgdStateFN = paths.concat(dir, 'sgd.state')
   end
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   cleanupModel(model)
   torch.save(filename, model)
   torch.save(sgdStateFN, sgdState)

   -- next epoch
   epoch = epoch + 1
end

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

----------------------------------------------------------------------
-- and train!
--
while true do
   -- train/test
   train(trainData)
   test(testData)

   -- plot errors
   if opt.plot then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
      trainLogger:plot()
      testLogger:plot()
   end
end

