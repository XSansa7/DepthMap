require 'nn'
require 'image'
require 'torch'
require 'optim'
require 'os'
require 'xlua'
require 'nnlr'

version = "newdata_depth15_origin_small"
owner = "qc449"

local cmd = torch.CmdLine()
cmd:option('-lr', 0.001, 'learning rate')
cmd:option('-momentum',         0.9,            'momentum')
cmd:option('-batchsize', 100, 'batch size')
cmd:option('-epochs', 100 , 'epochs')
-- cmd:option('-model', '', 'Model to use for training')
cmd:option('-verbose', 'false', 'Print stats for every batch')
cmd:option('-seed', 5 , 'random seed')
cmd:option('-duplicate', 4, 'number of duplicate')
cmd:option('-localt', false, 'local testing')
cmd:option('-cuda', false, 'cuda')

local config = cmd:parse(arg)
-- if config.model == '' or not paths.filep('models/'..config.model..'.lua') then
--         cmd:error('Invalid model ' .. config.model)
-- end

duplicate = config.duplicate

if config.cuda == true then
    require 'cunn'
    require 'cudnn' -- faster convolutions
    cudnn.benchmark = false
    cudnn.fastest = false
    cutorch = require 'cutorch'
    cutorch.manualSeedAll(config.seed)
end

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(10)

local tnt  = require 'torchnet'

if config.localt == true then
    train_image_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainImage/'
    train_label_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainLabel/'
    train_depth_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainDepth/'
    train_split = '/Users/QimingChen/Desktop/Computer Vision/project/trainsplit.t7'
    test_split = '/Users/QimingChen/Desktop/Computer Vision/project/testsplit.t7'
    N =  10
    TrainSize = 6
    TestSize = 4
    trainList = torch.range(1, TrainSize):long()
    testList = torch.range(TrainSize+1, N):long()
end

if config.localt == false then
    train_image_path = '/scratch/' .. owner .. '/project/NewTrainImage/'
    train_depth_path = '/scratch/' .. owner .. '/project/NewTrainDepth/'
    train_split = '/home/' .. owner .. '/project/trainsplit.t7'
    test_split = '/home/' .. owner .. '/project/testsplit.t7'
    N = 1449
    TrainSize = 795 -- 1449 * 0.6 * 0.9 = 782
    TestSize = 654
    trainList = torch.load(train_split):reshape(1*795):long()
    testList = torch.load(test_split):reshape(1*654):long()


    newtrainlist = torch.Tensor(TrainSize*duplicate)
    for i = 1, TrainSize do
        for j = 1, duplicate do
            newtrainlist[{i*duplicate-duplicate+j}] = trainList[{i}]*duplicate-duplicate + j
        end
    end
    
    -- newtestlist = torch.Tensor(TestSize*duplicate)
    -- for i = 1, TestSize do
    --     for j = 1, duplicate do
    --         newtestlist[{i*duplicate-duplicate+j}] = testList[{i}]*duplicate-duplicate + j
    --     end
    -- end

    
    -- TrainSize = N*0.8 -- 1449 * 0.8 * 0.9 = 782
    -- TestSize = N-TrainSize
    -- trainList = torch.range(1, TrainSize):long()
    -- testList = torch.range(TrainSize+1, N):long()
end
----------------------------- UTILITIES --------------------------------
local gen = torch.Generator()
function rand_scale(input, target)
    local r = torch.uniform(gen, 1, 1.5)
    input = input * r;
    target = target / r;
    return input, target
end

-- Rotation: Input and target are rotated by r ∈ [−5, 5] degrees.
function rand_rotate(input, target)
    local angle = torch.uniform(gen, -5, 5) * 3.14 / 180
    input = image.rotate(input, angle)
    target = image.rotate(target, angle)
    return input, target
end

-- Color: Input values are multiplied globally by a random RGB value c ∈ [0.8, 1.2]3.
function rand_color(input, target)
    -- print(input:size())
    for i = 1,3 do
        local color = torch.uniform(gen, 0.8, 1.2)
        input[{i,{},{}}] = input[{i,{},{}}] * color
    end
    return input, target
end

-- Flips: Input and target are horizontally flipped with 0.5 probability.
function rand_flip(input, target)
    local prob = torch.uniform(gen, 0, 1)
    if prob > 0.5 then
        input = image.hflip(input)
        target = image.hflip(target)
    end
    return input,target
end

function rand_crop(input, target)
    local kind = torch.random(gen, 1, 5)
    --print(input:size())
    if kind == 1 then
        input = image.crop(input, "c", 114, 152)
    elseif kind == 2 then
        input = image.crop(input, "tl", 114, 152)
    elseif kind == 3 then
        input = image.crop(input, "tr", 114, 152)
    elseif kind == 4 then
        input = image.crop(input, "bl", 114, 152)
    else 
        input = image.crop(input, "br", 114, 152)
    end
    input = image.scale(input, 120, 160)
    return input, target
end

function random_transformation(input, target)
    for batch = 1, input:size(1) do
      -- input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_scale(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
      -- input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_rotate(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
      input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_crop(input[{batch,{},{},{}}], target[{batch,1,{},{}}]);
      -- input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_color(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
      -- input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_flip(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
    end
    collectgarbage();
    return input, target
end 

function resize(img)
    return image.scale(img, 120 , 160)
    -- return img
end

function transformInput(inp)
    f = tnt.transform.compose{
        [1] = resize
    }
    -- g = tnt.transform.normalize{

    -- }
    -- return g(f(inp))
    return f(inp)
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
    -- print(idx)
    local i1 = 1+torch.floor(idx / duplicate)
    local i2 = idx - (i1-1) * duplicate
    if i2 == 0  then
        i1 = i1 - 1
        i2 = duplicate
    end
    local file = string.format("%04d", i1) .. string.format("-%04d.ppm", i2)
    -- local img = transformInput(image.load(train_image_path .. file))
    local img = transformInput(torch.load(train_image_path .. file))
    local nan_mask = img:ne(img)
    img[nan_mask] = 0
    return img
end

function getLabel(idx)
    local file = string.format("%04d.t7", idx)
    return torch.load(train_label_path .. file):reshape(640*480,1):float()
end

function getDepth(idx)
    -- print(idx)
    local i1 = 1+torch.floor(idx / duplicate)
    local i2 = idx - (i1-1) * duplicate
    if i2 == 0  then
        i1 = i1 - 1
        i2 = duplicate
    end
    local file = string.format("%04d", i1) .. string.format("-%04d.t7", i2)
    local depth = torch.load(train_depth_path .. file):reshape(1,640,480):float()
    local nan_mask = depth:ne(depth)
    depth[nan_mask] = 0
    return depth
end

-- function getImage(idx)
--     -- print(idx)
--     local file = string.format("%04d.ppm", idx)
--     -- local img = transformInput(torch.load(train_image_path .. file))
--     local img = transformInput(image.load(train_image_path .. file))
--     local nan_mask = img:ne(img)
--     img[nan_mask] = 0
--     return img
-- end

-- function getLabel(idx)
--     local file = string.format("%04d.t7", idx)
--     return torch.load(train_label_path .. file):reshape(640*480,1):float()
-- end

-- function getDepth(idx)
--     -- print(idx)
--     local file = string.format("%04d.t7", idx)
--     local depth = torch.load(train_depth_path .. file):reshape(1,640,480):float()
--     local nan_mask = depth:ne(depth)
--     depth[nan_mask] = 0
--     return depth
-- end

function getIterator(dataset)
    return tnt.DatasetIterator{
        dataset = tnt.BatchDataset{
            batchsize = config.batchsize,
            dataset = dataset
        }
    }
end

-- function groupError(output, target, threshold)
--     local c1 = threshold
--     local c2 = c1*threshold
--     local c3 = c2*threshold
--     local correct1 = 0
--     local correct2 = 0
--     local correct3 = 0
--     local new_output = output:reshape(output:size(1)*output:size(2)*output:size(3)*output:size(4))
--     local new_target = target:reshape(output:size(1)*output:size(2)*output:size(3)*output:size(4))
    
--     -- for i = 1, output:size(1) do
--     --     for j = 1, output:size(2) do
--     --         for k = 1, output:size(3) do
--     --             for l = 1, output:size(4) do
--     --                 local error = torch.abs(output[{i,j,k,l}] - target[{i,j,k,l}])
--     --                 if error < c1 then
--     --                     correct1 = correct1 + 1
--     --                 end
--     --                 if error < c2 then
--     --                     correct2 = correct2 + 1
--     --                 end
--     --                 if error < c3 then
--     --                     correct3 = correct3 + 1
--     --                 end
--     --             end
--     --         end
--     --     end
--     -- end

--     for i = 1, new_output:size(1) do
--         local error = torch.abs(new_output[{i}] - new_target[{i}])
--         if error < c1 then
--             correct1 = correct1 + 1
--         end
--         if error < c2 then
--             correct2 = correct2 + 1
--         end
--         if error < c3 then
--             correct3 = correct3 + 1
--         end
--     end

--     return correct1, correct2, correct3
-- end

function groupError(output, target, threshold)
    local correct1 = 0
    local correct2 = 0
    local correct3 = 0
    local new_output = output:reshape(output:size(1)*output:size(2)*output:size(3)*output:size(4))
    local new_target = target:reshape(output:size(1)*output:size(2)*output:size(3)*output:size(4))
    
    local threshold1 = torch.cdiv(new_output, new_target)
    local threshold2 = torch.cdiv(new_target, new_output)
    local threshold = torch.cat(threshold1, threshold2, 2)
    local mask = torch.max(threshold,2)
    local c1 = mask.lt(mask,1.25)
    correct1 = c1:sum()
    local c2 = mask.lt(mask,1.25*1.25)
    correct2 = c2:sum()
    local c3 = mask.lt(mask,1.25*1.25*1.25)
    correct3 = c3:sum()
    
    return correct1, correct2, correct3
end

-- DEBUG AREA ----------------------------------------

seed = config.seed

function BuildPreP1()
    local preprocessing1 = nn.Sequential()
    -- preprocessing1:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304
    preprocessing1:add(nn.SpatialSubSampling(3, 3, 3, 1, 1)) -- (120-3)/1 + 1 = 118, (160-3)/1 + 1 = 158
    --preprocessing1:add(nn.ReLU())
    return preprocessing1
end

function BuildPreP2()
    preprocessing2 = nn.Sequential()
    -- preprocessing2:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304
    preprocessing2:add(nn.SpatialSubSampling(3, 3, 3, 1, 1)) -- (120-3)/1 + 1 = 118, (160-3)/1 + 1 = 158
    preprocessing2:add(nn.SpatialConvolution(3, 96, 7, 9, 1, 1, 0, 0)) -- 150 110
    preprocessing2:add(nn.SpatialBatchNormalization(96))
    preprocessing2:add(nn.PReLU())
    preprocessing2:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- 96 74 55
    preprocessing2:add(nn.SpatialDropout(0.3))
    -- preprocessing2:add(nn.Reshape(96,74*55))
    return preprocessing2
end

function BuildPreP3()
    preprocessing3 = nn.Sequential()
    -- preprocessing3:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304
    preprocessing3:add(nn.SpatialSubSampling(3, 7, 7, 1, 1)) -- (120-6)/1 + 1 = 115, (160-6)/1 + 1 = 155
    preprocessing3:add(nn.SpatialConvolution(3, 96, 6, 8, 1, 1, 0, 0))
    preprocessing3:add(nn.SpatialBatchNormalization(96))
    preprocessing3:add(nn.PReLU())
    --preprocessing3:add(nn.SpatialDropout(0.3))
    -- preprocessing3:add(nn.SpatialMaxPooling(3, 3, 2, 2)) -- 96 147 109
    return preprocessing3
end

function BuildNetwork1()
    local network1 = nn.Sequential()
    -- network:add(nn.SpatialSubSampling(3, 2, 2, 2, 2)) -- (480-2)/2 + 1 = 240, (640-2)/2 + 1 = 320
    -- network:add(nn.SpatialSubSampling(3, 13, 17, 1, 1)) -- (240-13)/1 + 1 = 228, (320-17)/1 + 1 = 304

    -- layer 1.1
    -- network1:add(nn.SpatialConvolution(3, 64, 3, 3))
    -- -- network1:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network1:add(nn.ReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(64, 64, 3, 3))
    -- -- network1:add(nn.SpatialBatchNormalization(64,1e-3))
    -- network1:add(nn.ReLU())
    -- network1:add(nn.SpatialMaxPooling(2,2,2,2))
    -- 64 150 112

    -- layer 1.2
    -- network1:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(128,1e-3))
    -- network1:add(nn.ReLU())
    -- network1:add(nn.Dropout(0.3))
    -- network1:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    -- -- network1:add(nn.SpatialBatchNormalization(128,1e-3))
    -- network1:add(nn.ReLU())
    -- network1:add(nn.SpatialMaxPooling(2,2,2,2))


    network1:add(nn.SpatialConvolution(3, 64, 5, 3, 1, 1)) -- 154 116
    network1:add(nn.SpatialBatchNormalization(64,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    network1:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(64,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    network1:add(nn.SpatialConvolution(64, 128, 5, 5, 1, 1)) -- 150 112
    network1:add(nn.SpatialBatchNormalization(128,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    network1:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(128,1e-3))
    network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    network1:add(nn.SpatialDropout(0.3))
    -- 128 75 56

    -- -- layer 1.3
    network1:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(256,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    -- network1:add(nn.Dropout(0.3))
    network1:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(256,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    -- network1:add(nn.Dropout(0.3))
    network1:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(256,1e-3))
    network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    network1:add(nn.SpatialDropout(0.3))
    -- 256 37 28

    -- layer 1.4
    network1:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    -- network1:add(nn.Dropout(0.3))
    network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    -- network1:add(nn.Dropout(0.3))
    --network1:add(nn.SpatialDropout(0.3))
    network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    network1:add(nn.SpatialDropout(0.3))
    -- 512 18 14

    -- layer 1.5
    network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    -- network1:add(nn.Dropout(0.3))
    network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    --network1:add(nn.SpatialDropout(0.3))
    -- network1:add(nn.Dropout(0.3))
    network1:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1))
    network1:add(nn.SpatialBatchNormalization(512,1e-3))
    network1:add(nn.PReLU())
    network1:add(nn.SpatialMaxPooling(2,2,2,2))
    network1:add(nn.SpatialDropout(0.3))
    -- 512 9 7

    -- layer 1.6
    network1:add(nn.View(512*9*6))
    network1:add(nn.Linear(512*9*6, 4096))
    network1:add(nn.BatchNormalization(4096))
    network1:add(nn.PReLU())
    network1:add(nn.Dropout(0.5))
    -- network1:add(nn.Dropout(0.3))
    -- 1 4096

    -- network1:add(nn.L1Penalty(0.2))

    -- layer 1.7
    network1:add(nn.Linear(4096, 64*19*14))
    network1:add(nn.PReLU())
    network1:add(nn.Dropout(0.5))
    network1:add(nn.Reshape(64,19,14))

    -- layer 1.8
    network1:add(nn.SpatialUpSamplingNearest(4)) -- 64 76 56
    --network1:add(nn.ReLU())
    network1:add(nn.SpatialSubSampling(64, 2, 3, 1, 1)) -- 64 74 55
    network1:add(nn.SpatialBatchNormalization(64,1e-3))
    --network1:add(nn.ReLU())

    -- network1:add(nn.Reshape(64,74*55))
    return network1
end

function BuildNetwork2()
    local network2 = nn.Sequential()

    --layer 2.1
    network2:add(nn.SpatialConvolution(160, 160, 9, 9, 1, 1, 4, 4)) --160 74 55
    network2:add(nn.SpatialBatchNormalization(160,1e-3))
    network2:add(nn.PReLU())
    --network2:add(nn.SpatialDropout(0.3))
    -- network2:add(nn.Dropout(0.3))
    --layer 2.2
    network2:add(nn.SpatialConvolution(160, 64, 5, 5, 1, 1, 2, 2)) --64 74 55
    network2:add(nn.SpatialBatchNormalization(64,1e-3))
    network2:add(nn.PReLU())
    --network2:add(nn.SpatialDropout(0.3))
    -- network2:add(nn.Dropout(0.3))
    --layer 2.3
    network2:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 74 55
    network2:add(nn.SpatialBatchNormalization(64,1e-3))
    network2:add(nn.PReLU())
    --network2:add(nn.SpatialDropout(0.3))
    -- network2:add(nn.Dropout(0.3))
    -- --layer 2.4
    network2:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 74 55
    network2:add(nn.SpatialBatchNormalization(64,1e-3))
    network2:add(nn.PReLU())
    --network2:add(nn.SpatialDropout(0.3))
    -- network2:add(nn.Dropout(0.3))
    --layer 2.5
    network2:add(nn.SpatialConvolution(64, 40, 5, 5, 1, 1, 2, 2)) --64 74 55
    network2:add(nn.SpatialBatchNormalization(40,1e-3))
    network2:add(nn.PReLU())
    --network2:add(nn.SpatialDropout(0.3))
    -- network2:add(nn.Dropout(0.3))
    --layer 2.6
    network2:add(nn.SpatialUpSamplingNearest(2)) --  40 147 109
    --network2:add(nn.ReLU())
    network2:add(nn.SpatialSubSampling(40, 2, 2, 1, 1))
    network2:add(nn.SpatialBatchNormalization(40,1e-3))
    --network2:add(nn.ReLU())
    return network2
end

function BuildNetwork3()
    local network3 = nn.Sequential()
    --layer 3.1
    network3:add(nn.SpatialConvolution(136, 64, 9, 9, 1, 1, 4, 4)) --64 147 109
    network3:add(nn.SpatialBatchNormalization(64,1e-3))
    network3:add(nn.PReLU())
    --network3:add(nn.SpatialDropout(0.3))
    -- network3:add(nn.Dropout(0.3))
    --layer 3.2
    network3:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 147 109
    network3:add(nn.SpatialBatchNormalization(64,1e-3))
    network3:add(nn.PReLU())
    --network3:add(nn.SpatialDropout(0.3))
    -- network3:add(nn.Dropout(0.3))
    -- --layer 3.3
    network3:add(nn.SpatialConvolution(64, 64, 5, 5, 1, 1, 2, 2)) --64 147 109
    network3:add(nn.SpatialBatchNormalization(64,1e-3))
    network3:add(nn.PReLU())
    --network3:add(nn.SpatialDropout(0.3))
    -- network3:add(nn.Dropout(0.3))
    --layer 3.4
    network3:add(nn.SpatialConvolution(64, 1, 5, 5, 1, 1, 2, 2)) --1 147 109
    network3:add(nn.SpatialBatchNormalization(1,1e-3))
    network3:add(nn.PReLU())
    --network3:add(nn.SpatialDropout(0.3))
    -- network2:add(nn.Dropout(0.3))
    --layer 3.5
    network3:add(nn.SpatialSubSampling(1, 14, 20, 1, 1)) -- (147-13)/1 + 1 = 128, (109-14)/1 + 1 = 96
    network3:add(nn.SpatialUpSamplingNearest(5)) -- 640 -1 / 5 + 1 = 128 96
    network3:add(nn.SpatialBatchNormalization(1,1e-3))
    --network3:add(nn.ReLU())

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
--network1:add(nn.ReLU())

-- network1:add(nn.SpatialConvolution(64, 1, 5, 5, 1, 1, 2, 2)) --1 74 55
-- network1:add(nn.ReLU())
-- network1:add(nn.SpatialUpSamplingNearest(9)) -- 640 -1 / 5 + 1 = 128 96
-- network1:add(nn.ReLU())
-- network1:add(nn.SpatialSubSampling(1, 16, 27, 1, 1)) -- (147-13)/1 + 1 = 128, (109-14)/1 + 1 = 96
-- network1:add(nn.ReLU())


network12 = nn.Concat(2) -- 1 is batch index
network12:add(BuildPreP2()) --96 74 55
network12:add(network1) -- 64 74 55 -- 160 74 55
-- network12:add(nn.ReLU())

mlp2 = nn.Sequential()
mlp2:add(network12) -- 160 74 55
mlp2:add(BuildNetwork2()) -- 40 147 109
--mlp2:add(nn.ReLU())

network23 = nn.Concat(2)
network23:add(BuildPreP3())
network23:add(mlp2) -- 136 147 109
-- network23:add(nn.ReLU())

network = nn.Sequential()
network:add(network23)
network:add(BuildNetwork3())


criterion = nn.MSECriterion()

if config.cuda == true then
    network:cuda()
    criterion:cuda()
end

local lr = config.lr
local epochs = config.epochs
torch.manualSeed(0)
local generator = torch.Tensor(1) 

trainDataset = tnt.SplitDataset{
    partitions = {train=0.9, val=0.1},
    initialpartition = 'train',
    dataset = tnt.ShuffleDataset{
        dataset = tnt.ListDataset{
            -- list = torch.range(1, TrainSize):long(),
            -- list = torch.randperm(TrainSize)[{{1,5}}]:long(),
            list = newtrainlist:long(),
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

testDataset = tnt.ShuffleDataset{
    dataset = tnt.ListDataset{
        -- list = torch.range(1, TrainSize):long(),
        -- list = torch.randperm(TrainSize)[{{1,5}}]:long(),
        list = testList,
        -- print(list)
        load = function(idx)
            return {
                input =  getImage(idx),
                target = getDepth(idx)
            }
        end
    }
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

if config.cuda == true then
    local meter_gpu, clerr_gpu = torch.CudaTensor(), torch.CudaTensor()
    engine.hooks.onSample = function(state)
        -- state.sample.input:cuda()
        -- state.sample.target:cuda()
        state.sample.input, state.sample.target = random_transformation(state.sample.input, state.sample.target);
        meter_gpu:resize(state.sample.input:size()):copy(state.sample.input)
        state.sample.input  = meter_gpu
        if state.sample.target ~= nil then
            clerr_gpu:resize(state.sample.target:size()):copy(state.sample.target)
            state.sample.target = clerr_gpu
        end
    end
else
    engine.hooks.onSample = function(state)
        -- state.sample.input, state.sample.target = random_transformation(state.sample.input, state.sample.target);
        -- print(state.sample.input:size())
        -- print(state.sample.target:size())
    end
end

-- engine.hooks.onForward = function(state)
--     print(state.network.output:size())
-- end


engine.hooks.onForwardCriterion = function(state)
    -- print(state.network.output:size())
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

--local baseLearningRate = 0.01
--local baseWeightDecay = 0.0001
--local learningRates, weightDecays = network:getOptimConfig(baseLearningRate, baseWeightDecay)

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
            learningRate = lr,
            --learningRates = learningRates,
            --weightDecays = weightDecays,
            --learningRate = baseLearningRate,
            momentum = config.momentum
        }
    }

    network:clearState()
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
    -- correct1 = correct1 + sub_1 / (218*294)
    -- correct2 = correct2 + sub_2 / (218*294)
    -- correct3 = correct3 + sub_3 / (218*294)
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

-- save model
network:clearState()
network:double()
torch.save("/scratch/" .. owner .. "/project/" ..  version .. ".t7",network)

