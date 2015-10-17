require 'torch'
require 'paths'

mnistRot = {}

mnistRot.path_remote = 'http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip'
mnistRot.path_dataset = 'data/'
mnistRot.path_trainset = paths.concat(mnistRot.path_dataset, 'mnist_all_rotation_normalized_float_train_valid.amat')
mnistRot.path_testset = paths.concat(mnistRot.path_dataset, 'mnist_all_rotation_normalized_float_test.amat')

function mnistRot.download()
   if not paths.filep(mnistRot.path_trainset) or not paths.filep(mnistRot.path_testset) then
      local remote = mnistRot.path_remote
      local zip = paths.basename(remote)
      os.execute('mkdir -p ' .. mnistRot.path_dataset .. '; cd ' .. mnistRot.path_dataset .. 
      		 ';pwd; wget ' .. remote .. '; ' .. 'unzip ' .. zip .. '; cd -')
   end
end

function mnistRot.loadTrainSet(maxLoad, display, from)
   return mnistRot.loadDataset(mnistRot.path_trainset, maxLoad, display, from)
end

function mnistRot.loadTestSet(maxLoad, display, from)
   return mnistRot.loadDataset(mnistRot.path_testset, maxLoad, display, from)
end

function mnistRot.loadDataset(fileName, maxLoad, display, from)
   from = from or 1

   print "downloading mnistRot"
   mnistRot.download()

   print "reading mnistRot"
   local f = assert(io.open(fileName,'r'))
   local data = {}
   local labels = {}
   local index = 1
   for l in f:lines() do
      data[index] = {}
      for tok in string.gmatch(l,'%S+') do
	 table.insert(data[index],tonumber(tok))
      end
      table.insert(labels,table.remove(data[index])+1) -- comply with torch convention by starting from 1 so 0->1, 1->2
      index = index + 1
   end
   f:close()
   local tmp = torch.Tensor(#data,28,28)
   for i=1,#data do
      tmp[i] = torch.Tensor(data[i]):view(28,28):t()
   end
   data = tmp:contiguous()
   labels = torch.Tensor(labels)

   local nExample = data:size(1)
   if maxLoad and maxLoad > 0 and from+maxLoad-1 <= nExample and from > 0 and from <= nExample then
      nExample = maxLoad
      print('<mnistRot> loading only ' .. nExample .. ' examples')
   else from = 1 end
   data = data[{{from,from+nExample-1}}]
   labels = labels[{{from,from+nExample-1}}]
   print('<mnistRot> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   function dataset:display(num)
      num = num or 5
      num = math.min(num,nExample)
      assert(num>0,"error num in display")
      torch.randperm(nExample)[{{1,num}}]:apply(function(i)
	 image.display{image=data[i],legend=labels[i]} 
	 print('label ' .. i .. ' is ' .. labels[i])
	 end)
   end

   function dataset:normalize(mean_, std_)
      local mean = mean or data:view(data:size(1), -1):mean(1)
      local std = std_ or data:view(data:size(1), -1):std(1, true)
      for i=1,data:size(1) do
         data[i]:add(-mean[1][i])
         if std[1][i] > 0 then
            tensor:select(2, i):mul(1/std[1][i])
         end
      end
      return mean, std
   end

   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end

   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(10)

   setmetatable(dataset, {__index = function(self, index)
			     local input = self.data[index]
			     local class = self.labels[index]
			     local label = labelvector:zero()
			     label[class] = 1
			     local example = {input, label}
			     return example
   end})
   
   if display then dataset:display() end

   return dataset
end
