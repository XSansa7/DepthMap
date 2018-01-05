local torch = require 'torch'

function string:split(sep)
  local sep, fields = sep, {}
  local pattern = string.format("([^%s]+)", sep)
  self:gsub(pattern, function(substr) fields[#fields + 1] = substr end)
  return fields
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

    return data
end

-- DATA_PATH = '/Users/QimingChen/Desktop/Computer Vision/project/trainsplit.csv'
-- data = ReadCSV(DATA_PATH, 795, 1)
-- data = ReadCSV(DATA_PATH, 654, 1)
-- torch.save("/Users/QimingChen/Desktop/Computer Vision/project/trainsplit.t7", data)

DATA_PATH = '/Users/QimingChen/Desktop/Computer Vision/project/trainDepth2/'
for i = 1450, 3733 do
  file = DATA_PATH .. string.format("%04d.csv", i)
  data = ReadCSV(file, 640, 480)
  torch.save(DATA_PATH .. "t7/" .. string.format("%04d.t7", i), data)
end

-- data = torch.load("/Users/QimingChen/Desktop/Computer Vision/project/trainsplit.t7")
-- print(data)