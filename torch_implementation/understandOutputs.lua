require 'parseOutput'

-- oNew = parseOutput.parseAll('outputs/new_api',false)
-- oOld = parseOutput.parseAll('outputs/old_api',true)

utils = {}

function utils.getMaxTestAcc(outputs)
    -- find the output with the highest testAcc
    -- return testAcc and index of the output with highest Acc
    local res = 0
    local index = -1
    for k,v in pairs(outputs) do
    	local te = v['acc']['te']    	
    	if (te[#te]>res) then
	   res = te[#te]
	   index = k
	end
    end
    return res, index
end

function utils.getMaxTrAcc(outputs)
    -- find the output with the highest trainAcc
    -- return testAcc and index of the output with highest Acc
    local res = 0
    local index = -1
    for k,v in pairs(outputs) do
    	local tr = v['acc']['tr']    	
    	if (tr[#tr]>res) then
	   res = tr[#tr]
	   index = k
	end
    end
    return res, index
end

function utils.combineOutputs(o1,o2)
    -- concate items in o2 to o1
    local it = #o1
    for i,v in ipairs(o2) do
    	o1[it+i] = v
    end
end

-- sort the concatenated table
-- sort by te:
-- table.sort(o1,function(a,b) return a['acc']['te'][#a['acc']['te']] > b['acc']['te'][#b['acc']['te']] end)
-- sort by tr:
-- table.sort(o1,function(a,b) return a['acc']['tr'][#a['acc']['tr']] > b['acc']['tr'][#b['acc']['tr']] end)
function utils.sortOutput(o,mode)
	 mode = mode or "tr" -- mode are tr or te
	 if mode=='tr' then
	    	table.sort(o,function(a,b) return a['acc']['tr'][#a['acc']['tr']] > b['acc']['tr'][#b['acc']['tr']] end)
	 else
		table.sort(o,function(a,b) return a['acc']['te'][#a['acc']['te']] > b['acc']['te'][#b['acc']['te']] end)
	 end
	 
end

return utils