-- data augmentation

require 'image'
require 'torch'
require 'os'

Base_Path = '/home/qc449/' 
-- Base_Path = '/Users/QimingChen/Desktop/Computer Vision/'

train_image_path = Base_Path .. 'project/trainImage/'
train_depth_path = Base_Path .. 'project/trainDepth/'
DATA_PATH_Image = Base_Path .. 'project/NewTrainImage/'
DATA_PATH_Depth = Base_Path .. 'project/NewTrainDepth/'

N = 1449

function getImage(idx)
    -- print(idx)
    local file = string.format("%04d.ppm", idx)
    local img = image.load(train_image_path .. file)
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
    local file = string.format("%04d.t7", idx)
    local depth = torch.load(train_depth_path .. file):reshape(1,640,480):float()
    local nan_mask = depth:ne(depth)
    depth[nan_mask] = 0
    return depth
end

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
        input = image.crop(input, "c", 228, 304)
    elseif kind == 2 then
        input = image.crop(input, "tl", 228, 304)
    elseif kind == 3 then
        input = image.crop(input, "tr", 228, 304)
    elseif kind == 4 then
        input = image.crop(input, "bl", 228, 304)
    else 
        input = image.crop(input, "br", 228, 304)
    end
    -- input = image.scale(input, 120, 160)
    target = image.scale(target, 228,304)
    return input, target
end

function random_transformation(input, target)
    for batch = 1, input:size(1) do
      input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_scale(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
      input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_rotate(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
      input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_crop(input[{batch,{},{},{}}], target[{batch,1,{},{}}]);
      input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_color(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
      input[{batch,{},{},{}}], target[{batch,1,{},{}}] = rand_flip(input[{batch,{},{},{}}],  target[{batch,1,{},{}}]);
    end
    collectgarbage();
    return input, target
end 


for i = 1,1449 do
	print(i)

	local input = getImage(i)
	local target = getDepth(i)

	input = image.scale(input, 240, 320)
	target = image.scale(target, 240, 320)

	for j = 1,3 do
		  k = 1 + j * 4 - 4

		  input, target = rand_scale(input,  target);
		  file = DATA_PATH_Image .. string.format("%04d", i) .. string.format("-%04d.ppm", k) 
		  torch.save(file, input)
		  file = DATA_PATH_Depth .. string.format("%04d", i) .. string.format("-%04d.t7", k) 
		  torch.save(file, target)

		  k = 2 + j * 4 - 4

		  input, target = rand_rotate(input,  target);
		  file = DATA_PATH_Image .. string.format("%04d", i) .. string.format("-%04d.ppm", k) 
		  torch.save(file, input)
		  file = DATA_PATH_Depth .. string.format("%04d", i) .. string.format("-%04d.t7", k) 
		  torch.save(file, target)

		  k = 3 + j * 4 - 4

		  input, target = rand_color(input,  target);
		  file = DATA_PATH_Image .. string.format("%04d", i) .. string.format("-%04d.ppm", k) 
		  torch.save(file, input)
		  file = DATA_PATH_Depth .. string.format("%04d", i) .. string.format("-%04d.t7", k) 
		  torch.save(file, target)

		  k = 4 + j * 4 - 4

		  input, target = rand_flip(input,  target);
		  file = DATA_PATH_Image .. string.format("%04d", i) .. string.format("-%04d.ppm", k) 
		  torch.save(file, input)
		  file = DATA_PATH_Depth .. string.format("%04d", i) .. string.format("-%04d.t7", k) 
		  torch.save(file, target)
	end



end

