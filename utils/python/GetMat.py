'''
Created on Nov 27, 2016

@author: QimingChen
'''
import numpy as np  
import h5py

  
file_path = "/Users/QimingChen/Desktop/Computer Vision/project/nyu_depth_v2_labeled.mat"
data = h5py.File(file_path)  
  
images = data['images']
print np.shape(images) # (1449, 3, 640, 480)
#instances = data['instances']
#print np.shape(instances) # (1449, 640, 480)