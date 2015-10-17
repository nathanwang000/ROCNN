-- In this file I would like to parse the output
-- specifically, I want to make a summary of all the files in the output folder (excluding output of non models)

-- I care about
-- 1) the model name (done)
-- 2) model specification: lr, wd, rp (done)
-- 3) losses: a vector of losses (done)
-- 4) accuracy: on validation set (done)
-- 5) confusion matrices (done)

require 'paths'
require 'image'
require 'gnuplot'

parseOutput = {}

local function find(s1,s2)
   return s1:find(s2,nil,true)
end

local function rfind(s1,s2)
   -- reverse find s2 in s1
   pos1, pos2 = s1:reverse():find(s2:reverse(),nil,true)
   return pos1 and #s1 - pos2 + 1 or nil
end

function parseOutput.parseOutput(filename,isOld)
    -- return (bool, dic) where bool indicate file is outputfile
    -- and dic is of the shape {LearningRate:0.1,...}
   res = {}
   isOld = isOld or false

   local f = assert(io.open(filename, "r"))
   switch = {'Model','LearningRate','Momentum','LearningRateDecay','WeightDecay','BatchSize','RotationPrior'}
   convert = {tostring,tonumber,tonumber,tonumber,tonumber,tonumber,tonumber}

   local lines = {}
   -- read the lines in table 'lines'
   for line in f:lines() do
      table.insert(lines, line)
   end

   -- read in first 7 line parameters
   for i=1,#lines do
      local l = lines[i]
      if not l:find(switch[i]) then return false,res end
      res[switch[i]] = convert[i](l:sub(find(l,':')+2,rfind(l,'\x1b[0m')-1))
      if i==7 then break end
   end
   
   res['loss'] = {tr={}, te={}}
   res['acc'] = {tr={}, te={}}
   res['confusion'] = {tr={}, te={}}
   local i=1
   while i<=#lines do
      local l = lines[i]

      if isOld then -- deprecated: use see NOTE.txt for detail
      if find(l,'loss') then table.insert(res['loss']['tr'],tonumber(l:sub(find(l,'was')+4,rfind(l,'\x1b[0m')-1))) end
      else
      if find(l,'tr loss') then table.insert(res['loss']['tr'],tonumber(l:sub(find(l,'was')+4,rfind(l,'\x1b[0m')-1))) end
      if find(l,'te loss') then table.insert(res['loss']['te'],tonumber(l:sub(find(l,'was')+4,rfind(l,'\x1b[0m')-1))) end
      end
      
      -- read in confusion matrices for both train and test
      if  find(l,'time to learn') then
	 l = ""
	 i = i+2
	 while i<=#lines and not find(lines[i],'global correct') do
	    l = l..lines[i]..'\n'
	    i = i+1
	 end
	 if i<=#lines then 
	    table.insert(res['confusion']['tr'],l..lines[i]..'\n') 
	    table.insert(res['acc']['tr'],tonumber(lines[i]:sub(find(lines[i],':')+2,rfind(lines[i],'%')-1)) / 100)
	 end
      elseif find(l,'time to test') then
	 l = ""
	 i = i+2
	 while i<=#lines and not find(lines[i],'global correct') do
	    l = l..lines[i]..'\n'
	    i = i+1
	 end
	 if i<=#lines then 
	    table.insert(res['confusion']['te'],l..lines[i]..'\n') 
	    table.insert(res['acc']['te'],tonumber(lines[i]:sub(find(lines[i],':')+2,rfind(lines[i],'%')-1)) / 100)	 
	 end
      end
      
      i = i+1
   end

   return true,res
end

function parseOutput.parseAll(dir,isOld)
   -- parseOutput for every file in directory
   outputs = {}
   for f in paths.files(dir) do
      if paths.filep(paths.concat(dir,f)) then 
	 valid, output = parseOutput.parseOutput(paths.concat(dir,f),isOld)
	 if valid then table.insert(outputs,output)end
      end
   end
   return outputs
end

function parseOutput.plotOutput(output, savename)
   -- output should follow the result from parseOutput
   -- I should plot
   -- graph 1
   -- 1) tr loss
   -- 2) te loss
   -- graph 2
   -- 3) tr acc
   -- 4) te acc
   local lossPlot = {}
   local accPlot = {}

   if savename then os.execute('mkdir -p ' .. paths.dirname(savename)) end

   if #output.loss.tr~=0 then table.insert(lossPlot,{'tr loss', torch.Tensor(output.loss.tr),'-'}) end
   if #output.loss.te~=0 then table.insert(lossPlot,{'te loss', torch.Tensor(output.loss.te),'-'}) end
   if #output.acc.tr~=0 then table.insert(accPlot,{'tr acc', torch.Tensor(output.acc.tr),'-'}) end
   if #output.acc.te~=0 then table.insert(accPlot,{'te acc', torch.Tensor(output.acc.te),'-'}) end

   if not savename then gnuplot.figure(1) else gnuplot.pngfigure(savename .. '_loss.png') end
   gnuplot.plot(lossPlot)
   gnuplot.plotflush()

   if not savename then gnuplot.figure(2) else gnuplot.pngfigure(savename .. '_acc.png') end
   gnuplot.plot(accPlot)
   gnuplot.movelegend('left','top')
   gnuplot.plotflush()
end

function parseOutput.visualizeModel(model)
   print('<warning> implementation not checked!')
   -- model should be an nn.Roconv model, this function plots
   -- 1) critical layers of a Roconv model
   local W1 = model:get(2).W
   local W2 = model:get(5).W
   local W3 = model:get(7):get(1):get(2).weight
   for i=2,10 do
      W3 = torch.cat(W3,model:get(7):get(i):get(2).weight,1) -- 1x8
   end
   image.display{
      image=W1:view(W1:size(1)*W1:size(2)*W1:size(3),W1:size(4),W1:size(5)), zoom=4, nrow=10,
      min=-1, max=1,
      legend='stage 1: weights', padding=1
   }
   image.display{
      image=W2:view(W2:size(1)*W2:size(2)*W2:size(3),W2:size(4),W2:size(5)), zoom=4, nrow=30,
      min=-1, max=1,
      legend='stage 2: weights', padding=1
   }
   image.display{
      image=W3, zoom=10,
      min=-1, max=1,
      legend='stage 3: weights', padding=1
   }
end

function parseOutput.summary(output)
   -- make a summary of the output
   -- the following info should be displayed
   -- 1) {'Model','LearningRate','Momentum','LearningRateDecay','WeightDecay','BatchSize','RotationPrior'}
   -- 2) final confusion matrices
   local items = {'Model','LearningRate','Momentum','LearningRateDecay','WeightDecay','BatchSize','RotationPrior'}
   for i=1,#items do
      print(items[i] .. ': ' .. output[items[i]])
   end
   if #output.confusion.tr~=0 then print('train confusion matrix:\n' .. output.confusion.tr[#output.confusion.tr]) end
   if #output.confusion.te~=0 then print('test confusion matrix:\n' .. output.confusion.te[#output.confusion.te]) end
end

return parseOutput
