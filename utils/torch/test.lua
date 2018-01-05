require 'image'
require 'nn'
require 'torch'
require 'cudnn'
require 'cunn'
require 'cutorch'


-- model_path = '/Users/QimingChen/Desktop/Computer Vision/project/simplest10.94'
-- train_image_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainImage/'
train_image_path = '/home/qc449/project/trainImage/'
-- train_depth_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainDepth/'

id = 1449
file = string.format("%04d.ppm", id)
input = image.load(train_image_path .. file)
-- image.display(input)

-- file = string.format("%04d.t7", id)
-- target = torch.load(train_depth_path .. file)
-- image.display(target)

-- color_target = torch.Tensor(3,target:size(1),target:size(2))
-- color_target[{1,{}}]:copy(target)
-- color_target[{2,{}}]:copy(target)
-- color_target[{3,{}}]:copy(target)
-- -- print(color_target[{{},1,1}])
-- -- image.display(color_target)
-- image.display{image=image.y2jet(torch.linspace(1,10,10)), zoom=50}


network = torch.load('/home/qc449/simplest10.94')
network:float()
network.feedford(input)
output = network.output

-- image.display(output)

