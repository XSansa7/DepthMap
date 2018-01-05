require 'torch'
require 'optim'
require 'os'
require 'optim'
require 'xlua'
-- require 'cudnn' -- faster convolutions

--[[
--  Hint:  Plot as much as you can.  
--  Look into torch wiki for packages that can help you plot.
--]]

local tnt = require 'torchnet'
local image = require 'image'
local optParser = require 'opts'

local opt = optParser.parse(arg)

local WIDTH, HEIGHT = 60, 60
local DATA_PATH = (opt.data ~= '' and opt.data or './data/')

if opt.cuda == true then
    require 'cunn'
    require 'cudnn'
    cutorch = require 'cutorch'
    cutorch.manualSeedAll(opt.manualSeed)
end

torch.setdefaulttensortype('torch.DoubleTensor')

torch.setnumthreads(2)
-- torch.manualSeed(opt.manualSeed)


function resize(img)
    return image.scale(img, WIDTH,HEIGHT)
end

--[[
-- Hint:  Should we add some more transforms? shifting, scaling?
-- Should all images be of size 32x32?  Are we losing 
-- information by resizing bigger images to a smaller size?
--  normalize : done and work
--  more transform: 
--]]
function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    g = tnt.transform.normalize{

    }
    return g(f(inp))
end

function getTrainSample(dataset, idx)
    r = dataset[idx]
    classId, track, file = r[9], r[1], r[2]
    file = string.format("%05d/%05d_%05d.ppm", classId, track, file)
    return transformInput(image.load(DATA_PATH .. '/train_images/'..file))
end

function getTrainLabel(dataset, idx)
    return torch.LongTensor{dataset[idx][9] + 1}
end

function getTestSample(dataset, idx)
    r = dataset[idx]
    file = DATA_PATH .. "/test_images/" .. string.format("%05d.ppm", r[1])
    return transformInput(image.load(file))
end

function getIterator(dataset)
    --[[
    -- Done
    -- Hint:  Use ParallelIterator for using multiple CPU cores
    --]]
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = opt.batchsize,
            dataset = dataset
        }
    }
    -- return tnt.ParallelDatasetIterator{
    --     nthread = 1,
    --     init = function() 
    --         local tnt = require 'torchnet' 
    --     end,
    --     closure = function()
    --         -- return batches of data:
    --         return dataset
    --     end
    -- }
end

function regularization_penalty(network, l1_weight, l2_weight)
  local parameters, _ = network:parameters()
  local penalty = 0
  for i=1, table.getn(parameters) do
    penalty = penalty + l1_weight * parameters[i]:norm(1) + l2_weight * parameters[i]:norm(2) ^ 2
  end
  return penalty
end

local trainData = torch.load(DATA_PATH..'train.t7')
local testData = torch.load(DATA_PATH..'test.t7')

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    --[[
    --  Hint:  Use a resampling strategy that keeps the 
    --  class distribution even during initial training epochs 
    --  and then slowly converges to the actual distribution 
    --  in later stages of training.
    --]]
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, trainData:size(1)):long(),
            load = function(idx)
                return {
                    input =  getTrainSample(trainData, idx),
                    target = getTrainLabel(trainData, idx)
                }
            end
        }
        ,
        -- size = trainData:size(1),
        replacement = true
    }
}


testDataset = tnt.ListDataset{
    list = torch.range(1, testData:size(1)):long(),
    load = function(idx)
        return {
            input = getTestSample(testData, idx),
            target = torch.LongTensor{testData[idx][1]}
        }
    end
}


--[[
-- Done
-- Hint:  Use :cuda to convert your model to use GPUs
--]]
local model = require("models/".. opt.model)
local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local criterion = nn.CrossEntropyCriterion()
local clerr = tnt.ClassErrorMeter{topk = {1}}
local timer = tnt.TimeMeter()
local batch = 1

if opt.cuda == true then
    model:cuda()
    criterion:cuda()
end
-- print(model)

engine.hooks.onStart = function(state)
    meter:reset()
    clerr:reset()
    timer:reset()
    batch = 1
    if state.training then
        mode = 'Train'
    else
        mode = 'Val'
    end
end

--[[
-- DONE
-- Hint:  Use onSample function to convert to 
--        cuda tensor for using GPU
--]]
if opt.cuda == true then
    local meter_gpu, clerr_gpu = torch.CudaTensor(), torch.CudaTensor()
    engine.hooks.onSample = function(state)
        meter_gpu:resize(state.sample.input:size() ):copy(state.sample.input)
        clerr_gpu:resize(state.sample.target:size()):copy(state.sample.target)
        state.sample.input  = meter_gpu
        state.sample.target = clerr_gpu
    end
end

engine.hooks.onForwardCriterion = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    if opt.verbose == true then
        print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
                mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
    else
        xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{k = 1}, timer:value()))
end

local epoch = 1

while epoch <= opt.nEpochs do
    model:clearState()
    trainDataset:select('train')
    engine:train{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = opt.LR,
            momentum = opt.momentum
        }
    }

    trainDataset:select('val')
    engine:test{
        network = model,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    print('Done with Epoch '..tostring(epoch))
    epoch = epoch + 1

    regularization_penalty(model, 0.3, 0.3)
end

local submission = assert(io.open(opt.logDir .. "/submission.csv", "w"))
submission:write("Filename,ClassId\n")
batch = 1

--[[
--  This piece of code creates the submission
--  file that has to be uploaded in kaggle.
--]]
engine.hooks.onForward = function(state)
    local fileNames  = state.sample.target
    local _, pred = state.network.output:max(2)
    pred = pred - 1
    for i = 1, pred:size(1) do
        submission:write(string.format("%05d,%d\n", fileNames[i][1], pred[i][1]))
    end
    xlua.progress(batch, state.iterator.dataset:size())
    batch = batch + 1
end

engine.hooks.onEnd = function(state)
    submission:close()
end

engine:test{
    network = model,
    iterator = getIterator(testDataset)
}

print("The End!")
