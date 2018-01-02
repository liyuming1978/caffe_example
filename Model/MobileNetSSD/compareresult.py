#!/usr/bin/python
# coding: utf-8
import cv2
import time
import os
import numpy as np
import sys

print(cv2.__version__)

caffe_root = '/home/lym/code/caffe/'            #you must change it here!!!!!!!!!!!!
sys.path.insert(0, caffe_root + 'python')
import caffe.proto.caffe_pb2 as caffe_pb2
import caffe
caffe.set_device(0)
caffe.set_mode_gpu()
net_file= os.path.split(os.path.realpath(__file__))[0]+'/MobileNetSSD_deploy_fuse.prototxt'  
caffe_model= os.path.split(os.path.realpath(__file__))[0]+'/MobileNetSSD_deploy_fuse.caffemodel'  
net = caffe.Net(net_file,caffe_model,caffe.TEST)  

#person is 15
CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

def preprocess(src):
    img = cv2.resize(src, (300,300))
    img = img - 127.5
    img = img * 0.007843
    return img

def postprocess(img, out):   
    h = img.shape[0]
    w = img.shape[1]
    box = out['detection_out'][0,0,:,3:7] * np.array([w, h, w, h])

    cls = out['detection_out'][0,0,:,1]
    conf = out['detection_out'][0,0,:,2]
    return (box.astype(np.int32), conf, cls)

def detect(camimg):   
    img = preprocess(camimg)
    
    img = img.astype(np.float32)
    img = img.transpose((2, 0, 1))

    net.blobs['data'].data[...] = img
    out = net.forward()  
    box, conf, cls = postprocess(camimg, out)
    
    for i in range(len(box)):
        p1 = (box[i][0], box[i][1])
        p2 = (box[i][2], box[i][3])
        cv2.rectangle(camimg, p1, p2, (0,255,0))             
        p3 = (max(p1[0], 15), max(p1[1], 15))
        title = "%s:%.2f" % (CLASSES[int(cls[i])], conf[i])
        cv2.putText(camimg, title, p3, cv2.FONT_ITALIC, 0.6, (255, 0, 0), 1)
        
    cv2.imwrite('testresult.jpg',camimg)

    ##start write each layers data
    for layer_name, blob in net.blobs.iteritems():
        layername = layer_name.replace('/','_')
        fd=open('data/'+layername+'.bin','rb')
        rdata=np.frombuffer(fd.read(),dtype=np.float32)
        cdata= blob.data[0].reshape(-1)
        print layername,'--result=',np.sqrt(np.sum(np.square(rdata - cdata)))
        if(layername=='data'):
            print 'img','--result=',np.sqrt(np.sum(np.square(rdata - img.reshape(-1))))
        fd.close()
        
##################################################################################             
detect(cv2.imread('test.bmp'))