require 'nn'
require 'image'
require 'torch'
require 'optim'
require 'os'
require 'xlua'

local cmd = torch.CmdLine()
cmd:option('-lr', 0.001, 'learning rate')
cmd:option('-momentum',         0.9,            'momentum')
cmd:option('-batchsize', 100, 'batch size')
cmd:option('-epochs', 100 , 'epochs')
cmd:option('-model', '', 'Model to use for training')
cmd:option('-verbose', 'false', 'Print stats for every batch')
cmd:option('-seed', 5 , 'random seed')
cmd:option('-localt', false, 'local testing')
local config = cmd:parse(arg)
if config.model == '' or not paths.filep('models/'..config.model..'.lua') then
        cmd:error('Invalid model ' .. config.model)
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(10)

local tnt  = require 'torchnet'

if config.localt == true then
    train_image_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainImage/'
    train_label_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainLabel/'
    train_depth_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainDepth/'
    N =  10
end

if config.localt == false then

    train_image_path = '/home/qc449/project/trainImage/'
    train_label_path = '/home/qc449/project/trainLabel/'
    train_depth_path = '/home/qc449/project/trainDepth/'
    N = 1449
end
----------------------------- UTILITIES --------------------------------
function resize(img)
    return image.scale(img, 120 ,160)
    -- return img
end

function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    g = tnt.transform.normalize{

    }
    return g(f(inp))
end

function ReadCSV(filePath, ROWS, COLS)
    local csvFile = io.open(filePath, 'r')  
    local header = csvFile:read()

    local data = torch.Tensor(ROWS, COLS)

    local i = 0  
    for line in csvFile:lines('*l') do  
      i = i + 1
      local l = line:split(',')
      for key, val in ipairs(l) do
        data[i][key] = val
      end
    end
    csvFile:close() 
end

function getImage(idx)
    print(idx)
    local file = string.format("%04d.ppm", idx)
    local img = transformInput(image.load(train_image_path .. file))
    local nan_mask = img:ne(img)
    img[nan_mask] = 0
    return img
end

function getLabel(idx)
    local file = string.format("%04d.t7", idx)
    return torch.load(train_label_path .. file):reshape(640*480,1):float()
end

function getDepth(idx)
    print(idx)
    local file = string.format("%04d.t7", idx)
    local depth = torch.load(train_depth_path .. file):reshape(1,640,480):float()
    local nan_mask = depth:ne(depth)
    depth[nan_mask] = 0
    return depth
end

function getIterator(dataset)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = dataset
        }
    }
end

function groupError(output, target, threshold)
    local c1 = threshold
    local c2 = c1*threshold
    local c3 = c2*threshold
    local correct1 = 0
    local correct2 = 0
    local correct3 = 0
    local new_output = output:reshape(output:size(1)*output:size(2)*output:size(3)*output:size(4))
    local new_target = target:reshape(output:size(1)*output:size(2)*output:size(3)*output:size(4))
    
    -- for i = 1, output:size(1) do
    --     for j = 1, output:size(2) do
    --         for k = 1, output:size(3) do
    --             for l = 1, output:size(4) do
    --                 local error = torch.abs(output[{i,j,k,l}] - target[{i,j,k,l}])
    --                 if error < c1 then
    --                     correct1 = correct1 + 1
    --                 end
    --                 if error < c2 then
    --                     correct2 = correct2 + 1
    --                 end
    --                 if error < c3 then
    --                     correct3 = correct3 + 1
    --                 end
    --             end
    --         end
    --     end
    -- end

    for i = 1, new_output:size(1) do
        local error = torch.abs(new_output[{i}] - new_target[{i}])
        if error < c1 then
            correct1 = correct1 + 1
        end
        if error < c2 then
            correct2 = correct2 + 1
        end
        if error < c3 then
            correct3 = correct3 + 1
        end
    end

    return correct1/output:size(1), correct2/output:size(1), correct3/output:size(1)
end



-- DEBUG AREA ----------------------------------------

TrainSize = torch.floor(N * 0.6) -- 1449 * 0.6 * 0.9 = 782
ValidSize = torch.floor(N * 0.6) 
TestSize = N - TrainSize - ValidSize
seed = config.seed

function BuildPreP1()
    preprocessing1 = nn.Sequential()
    -- preprocessing1:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304
    preprocessing1:add(nn.SpatialSubSampling(3, 3, 3, 1, 1)) -- (120-3)/1 + 1 = 118, (160-3)/1 + 1 = 158
    preprocessing1:add(nn.PReLU())
    return preprocessing1
end

function BuildPreP2()
    preprocessing2 = nn.Sequential()
    -- preprocessing2:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304
    preprocessing2:add(nn.SpatialSubSampling(3, 3, 3, 1, 1)) -- (120-3)/1 + 1 = 118, (160-3)/1 + 1 = 158
    preprocessing2:add(nn.SpatialConvolution(3, 96, 7, 9, 1, 1, 0, 0)) -- 150 110
    preprocessing2:add(nn.PReLU())
    preprocessing2:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- 96 74 55
    -- preprocessing2:add(nn.Reshape(96,74*55))
    return preprocessing2
end

function BuildPreP3()
    preprocessing3 = nn.Sequential()
    -- preprocessing3:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304
    preprocessing3:add(nn.SpatialSubSampling(3, 7, 7, 1, 1)) -- (120-6)/1 + 1 = 115, (160-6)/1 + 1 = 155
    preprocessing3:add(nn.SpatialConvolution(3, 96, 6, 8, 1, 1, 0, 0))
    preprocessing3:add(nn.PReLU())
    -- preprocessing3:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- 96 147 109
    return preprocessing3
end

function BuildNetwork1()
    network1 = nn.Sequential()
    -- network:add(nn.SpatialSubSampling(3, 2, 2, 2, 2)) -- (480-2)/2 + 1 = 240, (640-2)/2 + 1 = 320
    -- network:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304

    -- layer 1.1
    -- network1:add(nn.SpatialConvolution(3, 64, 3, 3))
    -- -- network1:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(64, 64, 3, 3))
    -- -- network1:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network1:add(nn.PReLU())
    -- network1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- 64 150 112

    -- layer 1.2
    -- network1:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(128,1e-3))
    -- network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(128,1e-3))
    -- network1:add(nn.PReLU())
    -- network1:add(nn.SpatialMaxPooling(2,2,2,2))
    network1:add(nn.SpatialConvolution(3, 64, 5, 3, 1, 1)) -- 154 116
    -- network1:add(nn.SpatialBatchNormalization(128,1e-3))
    network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    network1:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1)) -- 150 112
    -- network1:add(nn.SpatialBatchNormalization(128,1e-3))
    network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- 128 75 56

    -- -- layer 1.3
    network1:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    -- network1:add(nn.SpatialBatchNormalization(256,1e-3))
    network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(256,1e-3))
    -- network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(256,1e-3))
    -- network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- 256 37 28

    -- layer 1.4
    network1:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    -- network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(512,1e-3))
    -- network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(512,1e-3))
    -- network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- 512 18 14

    -- layer 1.5
    network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(512,1e-3))
    -- network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(512,1e-3))
    -- network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- 512 9 7

    -- layer 1.6
    network1:add(nn.Reshape(512*9*7))
    network1:add(nn.Linear(512*9*7, 4096))
    -- network1:add(nn.BatchNormalization(4096))
    network1:add(nn.PReLU())
    network1:add(nn.Dropout(0.3))
    -- 1 4096

    -- layer 1.7
    network1:add(nn.Linear(4096, 64*19*14))
    network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    network1:add(nn.Reshape(64,19,14))

    -- layer 1.8
    network1:add(nn.SpatialUpSamplingNearest(4)) -- 64 76 56
    network1:add(nn.PReLU())
    network1:add(nn.SpatialSubSampling(64, 2, 3, 1, 1)) -- 64 74 55
    network1:add(nn.PReLU())

    -- network1:add(nn.Reshape(64,74*55))
    return network1
end

function BuildNetwork2()
    network2 = nn.Sequential()

    --layer 2.1
    -- network2:add(nn.SpatialConvolution(160, 160, 9, 9, 1, 1, 4, 4)) --160 74 55
    -- -- network2:add(nn.SpatialBatchNormalization(160,1e-3))
    -- network2:add(nn.PReLU())
    -- network2:add(nn.Dropout(0.3))
    --layer 2.2
    network2:add(nn.SpatialConvolution(160, 64, 5, 5, 1, 1, 2, 2)) --64 74 55
    -- network2:add(nn.SpatialBatchNormalization(64,1e-3))
    network2:add(nn.PReLU())
    -- network2:add(nn.Dropout(0.3))
    --layer 2.3
    -- network2:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 74 55
    -- -- network2:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network2:add(nn.PReLU())
    -- network2:add(nn.Dropout(0.3))
    -- --layer 2.4
    -- network2:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 74 55
    -- -- network2:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network2:add(nn.PReLU())
    -- network2:add(nn.Dropout(0.3))
    --layer 2.5
    network2:add(nn.SpatialConvolution(64, 40, 5, 5, 1, 1, 2, 2)) --64 74 55
    network2:add(nn.PReLU())
    -- network2:add(nn.Dropout(0.3))
    --layer 2.6
    network2:add(nn.SpatialUpSamplingNearest(2)) --  40 147 109
    network2:add(nn.PReLU())
    network2:add(nn.SpatialSubSampling(40, 2, 2, 1, 1))
    network2:add(nn.PReLU())
    return network2
end

function BuildNetwork3()
    network3 = nn.Sequential()
    --layer 3.1
    network3:add(nn.SpatialConvolution(136, 64, 9, 9, 1, 1, 4, 4)) --64 147 109
    -- network3:add(nn.SpatialBatchNormalization(64,1e-3))
    network3:add(nn.PReLU())
    -- network3:add(nn.Dropout(0.3))
    --layer 3.2
    -- network3:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 147 109
    -- -- network3:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network3:add(nn.PReLU())
    -- network3:add(nn.Dropout(0.3))
    -- --layer 3.3
    -- network3:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 147 109
    -- -- network3:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network3:add(nn.PReLU())
    -- network3:add(nn.Dropout(0.3))
    --layer 3.4
    network3:add(nn.SpatialConvolution(64, 1, 5, 5, 1, 1, 2, 2)) --1 147 109
    network3:add(nn.PReLU())
    -- network2:add(nn.Dropout(0.3))
    --layer 3.5
    network3:add(nn.SpatialSubSampling(1, 14, 20, 1, 1)) -- (147-13)/1 + 1 = 128, (109-14)/1 + 1 = 96
    network3:add(nn.SpatialUpSamplingNearest(5)) -- 640 -1 / 5 + 1 = 128 96
    network3:add(nn.PReLU())

    return network3
end


-----------                 ------------
-----------  LOSS FUNCTION  ------------

-- function depthLoss(estimation, output)
--     local item1 = ( torch.csub(network.output, target):reshape(640*480):sum() )/ (640*480)
--     local item2 = ( torch.csub(network.output, target):reshape(640*480):sum() )/ (2*640*640*480*480)

-- end

-- local DepthCriterion, parent = torch.class('nn.DepthCriterion', 'nn.Criterion') -- MSE type
-- function DepthCriterion:__init(self)
--    parent.__init(self)
--    self.Li = torch.Tensor()
--    self.gradInput = {}
-- end

-- function DepthCriterion:updateOutput(input, target)
--    self.output_tensor = self.output_tensor or input.new(1)
   
--    local N = a:size(1) 
--    self.Li = torch.max(torch.cat(torch.Tensor(N):zero():type(torch.type(a)) , (a - p):norm(2,2):pow(2) -  (a - n):norm(2,2):pow(2) + self.alpha, 2), 2)
--    self.output = self.Li:sum() / N


--    self.output = self.output_tensor[1]
--    return self.output
-- end

-- function DepthCriterion:updateGradInput(input, target)
--     local a = input[1] -- ancor
--    local p = input[2] -- positive
--    local n = input[3] -- negative
--    local N = a:size(1)

--    self.gradInput[1] = (n - p):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
--    self.gradInput[2] = (p - a):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)
--    self.gradInput[3] = (a - n):cmul(self.Li:gt(0):repeatTensor(a:size(2),1):t():type(a:type()) * 2/N)

--    return self.gradInput
--    return self.gradInput
-- end

-----------      END        ------------
network1 = nn.Sequential()
network1:add(BuildPreP1())
network1:add(BuildNetwork1()) --64 74 55
network1:add(nn.PReLU())


network12 = nn.Concat(2) -- 1 is batch index
network12:add(BuildPreP2()) --96 74 55
network12:add(network1) -- 64 74 55 -- 160 74 55
-- network12:add(nn.PReLU())

mlp2 = nn.Sequential()
mlp2:add(network12) -- 160 74 55
mlp2:add(BuildNetwork2()) -- 40 147 109
mlp2:add(nn.PReLU())

network23 = nn.Concat(2)
network23:add(BuildPreP3())
network23:add(mlp2) -- 136 147 109
-- network23:add(nn.PReLU())

network = nn.Sequential()
network:add(network23)
network:add(BuildNetwork3())
network:add(nn.PReLU())


-- network = nn.Sequential()
-- network:add(nn.SpatialConvolution(3,1,3,3,1,1,1,1))
-- network:add(nn.PReLU())
-- network:add(nn.SpatialUpSamplingNearest(2))


criterion = nn.MSECriterion()

local lr = config.lr
local epochs = config.epochs
torch.manualSeed(0)
local generator = torch.Tensor(1) 

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            list = torch.range(1, TrainSize):long(),
            -- list = torch.randperm(TrainSize)[{{1,5}}]:long(),
            -- print(list)
            load = function(idx)
                return {
                    input =  getImage(idx),
                    target = getDepth(idx)
                }
            end
        }
    }
}

testDataset = tnt.ListDataset{
    list = torch.range(TrainSize+1, N):long(),
    load = function(idx)
        return {
            input =  getImage(idx),
            target = getDepth(idx)
        }
    end
}

local engine = tnt.OptimEngine()
local meter = tnt.AverageValueMeter()
local clerr = tnt.MSEMeter(true)
local timer = tnt.TimeMeter()
local batch = 1

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

engine.hooks.onForwardCriterion = function(state)
    print(state.network.output:size())
    meter:add(state.criterion.output)
    -- print(state.network.output:nDimension())
    -- print(state.sample.target:size())
    clerr:add(state.network.output, state.sample.target)

    if config.verbose == true then
        -- print(string.format("%s Batch: %d/%d; avg. loss: %2.4f; avg. error: %2.4f",
        --         mode, batch, state.iterator.dataset:size(), meter:value(), clerr:value{k = 1}))
        
    else
        -- xlua.progress(batch, state.iterator.dataset:size())
    end
    batch = batch + 1 -- batch increment has to happen here to work for train, val and test.
    timer:incUnit()
    collectgarbage();
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    mode, meter:value(), clerr:value{}, timer:value()))
end

for epoch = 1, epochs do  
    print(epoch)
    network:clearState()

    trainDataset:select('train')
    engine:train{
        network = network,
        criterion = criterion,
        iterator = getIterator(trainDataset),
        optimMethod = optim.sgd,
        maxepoch = 1,
        config = {
            learningRate = config.lr,
            momentum = 0.1
        }
    }
    trainDataset:select('val')
    engine:test{
        network = network,
        criterion = criterion,
        iterator = getIterator(trainDataset)
    }
    -- print('Done with Epoch '..tostring(epoch))
end

correct1 = 0;
correct2 = 0;
correct3 = 0;
engine.hooks.onForward = function(state)
    meter:add(state.criterion.output)
    clerr:add(state.network.output, state.sample.target)
    sub_1,sub_2,sub_3 = groupError(state.network.output, state.sample.target, 1.25)
    correct1 = correct1 + sub_1 / (640*480)
    correct2 = correct2 + sub_2 / (640*480)
    correct3 = correct3 + sub_3 / (640*480)
end

engine.hooks.onEnd = function(state)
    print(string.format("%s: avg. loss: %2.4f; avg. error: %2.4f, time: %2.4f",
    "test", meter:value(), clerr:value{}, timer:value()))
end

meter:reset()
clerr:reset()
engine:test{
    network = network,
    iterator = getIterator(testDataset),
    criterion = criterion
}
print(correct1 / TestSize)
print(correct2 / TestSize)
print(correct3 / TestSize)

