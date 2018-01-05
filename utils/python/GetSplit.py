'''
Created on Nov 27, 2016

@author: QimingChen
'''
import numpy as np  
import h5py
import csv
import scipy.io as sio
  

file_path = "/Users/QimingChen/Desktop/Computer Vision/project/nyu_depth_data_labeled.mat"
#file_path = "/Users/QimingChen/Desktop/Computer Vision/project/splits.mat"
#data = sio.loadmat(file_path)
#print data
data = h5py.File(file_path)

#images = data['images']
#print np.shape(images) # (2284, 3, 640, 480)
#
#def write_ppm(fname, im, xval):
#    nc, height, width = im.shape
#    assert nc == 3
#    
#    f = open(fname, 'w')
#    
#    f.write('P3\n')
#    f.write(str(width)+' '+str(height)+'\n')
#    f.write(str(xval)+'\n')
#    
#    # interleave image channels before streaming
#    c1 = np.reshape(im[0, :, :], (width*height, 1))
#    c2 = np.reshape(im[1, :, :], (width*height, 1))
#    c3 = np.reshape(im[2, :, :], (width*height, 1))
#    
#    im1 = np.hstack([c1, c2, c3])
#    im2 = im1.reshape(3*width*height)
#    
#    f.write('\n'.join(im2.astype('str')))
#    f.write('\n')
#    
#    f.close()
#
#ct = 1
#folder = "/Users/QimingChen/Desktop/Computer Vision/project/trainImage2/"
#for image in images:
#    name = str(ct+1449).zfill(5)[-4:]
#    print(name)
#    write_ppm(folder+name+".ppm", image, 255)
#    ct += 1

depths = data['depths']
folder = "/Users/QimingChen/Desktop/Computer Vision/project/trainDepth2/"
ct = 1
for depth in depths:
    name = str(ct+1449).zfill(5)[-4:]
    print(name)
    a = np.asarray(depth)
    np.savetxt(folder+name+".csv", a, delimiter=",")
    ct += 1


#a = []
#for ele in data['trainNdxs']:
#    a.append(ele[0])
# ### sudo
#for ele in range (1450, 3734):
#    a.append(ele)
#### sudo
#a = np.asarray(a)
#print a
#np.savetxt("/Users/QimingChen/Desktop/Computer Vision/project/trainsplit.csv", a, fmt='%4.0f', delimiter=",")
# a = np.asarray(instance)
#     np.savetxt(folder+name+".csv", a, delimiter=",")
# print data['testNdxs'][1][0]

# testNdxs
# trainNdxs
# data = h5py.File(file_path)  
# print(data["trainNdxs"])
# print(data["testNdxs"])

#########  IMAGES ############
# images = data['images']  
# print np.shape(images) # (1449, 3, 640, 480)
# def write_ppm(fname, im, xval):
#     nc, height, width = im.shape
#     assert nc == 3
#        
#     f = open(fname, 'w')
#        
#     f.write('P3\n')
#     f.write(str(width)+' '+str(height)+'\n')
#     f.write(str(xval)+'\n')
#        
#     # interleave image channels before streaming    
#     c1 = np.reshape(im[0, :, :], (width*height, 1))
#     c2 = np.reshape(im[1, :, :], (width*height, 1))
#     c3 = np.reshape(im[2, :, :], (width*height, 1))
#        
#     im1 = np.hstack([c1, c2, c3])
#     im2 = im1.reshape(3*width*height)
#    
#     f.write('\n'.join(im2.astype('str')))
#     f.write('\n')
#    
#     f.close()
# ct = 1
# folder = "/Users/QimingChen/Desktop/Computer Vision/project/trainImage/"
# for image in images:
#     name = str(ct).zfill(5)[-4:]
#     write_ppm(folder+name+".ppm", image, 255)
#     ct += 1

#########  LABELS ############
# 
# labels = data['labels']
# print np.shape(labels) # (1449, 640, 480)
# max_val = 0
# for i in range(1449):
#     labels_matrix = np.reshape(labels[i],(1,640*480))
# #     max_val = max( max_val, np.amax(labels_matrix) )
#     max_val = np.amax(labels_matrix)
#     print i
#     print max_val
# print max_val

# 894 labels

# ct = 1
# folder = "/Users/QimingChen/Desktop/Computer Vision/project/trainLabel/"
# for label in labels:
#     name = str(ct).zfill(5)[-4:]
#     a = np.asarray(label)
#     np.savetxt(folder+name+".csv", a, delimiter=",")
#     ct += 1
    
##########  DEPTHS ############
# depths = data['depths']
# print np.shape(depths) # (1449, 640, 480)
# 
# 
# max_val = 0
# for i in range(1449):
#     depths_matrix = np.reshape(depths[i],(1,640*480))
#     max_val = max( max_val, np.amax(depths_matrix) )
#     print i
#     print max_val
# print max_val # 9.99547
# 
# folder = "/Users/QimingChen/Desktop/Computer Vision/project/trainDepth/"
# ct = 1
# for depth in depths:
#     name = str(ct).zfill(5)[-4:]
#     a = np.asarray(depth)
#     np.savetxt(folder+name+".csv", a, delimiter=",")
#     ct += 1

###########  NAMES #############
# names = data['names'].value
# # print names
# for i in range(894):
#     st = names[0][i]
#     obj = data[st]
#     str1 = ''.join(chr(i) for i in obj[:])
#     print( str1 )

##########  Instances #########
# instances = data['instances']
# print instances

# max_val = 0
# for i in range(1449):
#     instances_matrix = np.reshape(instances[i],(1,640*480))
#     max_val = max( max_val, np.amax(instances_matrix) )
#     print i
#     print max_val
# print max_val # 37

# folder = "/Users/QimingChen/Desktop/Computer Vision/project/trainInstance/"
# ct = 1
# for instance in instances:
#     name = str(ct).zfill(5)[-4:]
#     a = np.asarray(instance)
#     np.savetxt(folder+name+".csv", a, delimiter=",")
#     ct += 1