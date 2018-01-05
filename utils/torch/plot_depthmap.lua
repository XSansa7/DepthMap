require 'image'
require 'nn'
require 'torch'

model_path1 = '/Users/QimingChen/Desktop/simplest10.94'
model_path2 = '/Users/QimingChen/Desktop/model.t7'
train_image_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainImage/'
train_depth_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainDepth/'

-- model_path = '/scratch/qc449/project/'
-- train_image_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainImage/'
-- train_depth_path = '/Users/QimingChen/Desktop/Computer Vision/project/trainDepth/'

function groupError(output, target, threshold)
    local c1 = threshold
    local c2 = c1*threshold
    local c3 = c2*threshold
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

id = 649
file = string.format("%04d.ppm", id)
input = image.load(train_image_path .. file)
image.display(input)

file = string.format("%04d.t7", id)
target = torch.load(train_depth_path .. file)
image.display(target)

-- target:div(10)
-- target:mul(255)

-- color_target = torch.Tensor(3,target:size(1),target:size(2))
-- color_target[{1,{}}]:fill(0)
-- color_target[{2,{}}]:copy(target)
-- color_target[{3,{}}]:fill(0)
-- print(color_target[{{},1,1}])
-- image.display(color_target)


input = image.scale(input, 120 ,160)
network = torch.load(model_path1)
network:double()
network:forward(input)
output = network.output

sub_1,sub_2,sub_3 = groupError(output, target, 1.25)
print(sub_1)
print(sub_2)
print(sub_3)
image.display(output)

network = torch.load(model_path2)
network:double()
network:forward(input)
output = network.output

sub_1,sub_2,sub_3 = groupError(output, target, 1.25)
print(sub_1)
print(sub_2)
print(sub_3)
image.display(output)