# -*- coding: UTF-8 -*-
import sys
import logging
import cv2
import time

#caffe_root:caffe安装路径
try:
    caffe_root = '/home/zby/work/software/ssd/'
    sys.path.insert(0,caffe_root+'python')
    import caffe
except ImportError:
    logging.fatal("Cannot find caffe!")

import numpy as np

#指定caffe prototxt和caffe_model路径
deploy      = "./ssd_mobilenet/deploy.prototxt"
caffe_model = "./ssd_mobilenet/deploy.caffemodel"

#指定图片路径,protoxtx文件输出层的name
img = "./ssd_mobilenet/test.jpg"
outputName = "detection_out"

net = caffe.Net(deploy,caffe_model,caffe.TEST)

#图片预处理 resize,减均值，scale
im = cv2.imread(img)
im = cv2.resize(im,(300,300))
im = im.astype(np.float32)
im = im - 127.5
im = im * 0.007843
#BGR->RGB
im = im.transpose((2, 0, 1))
net.blobs['data'].data[...]=im
out = net.forward()
out = out['detection_out']
#输出结果
print(out)

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
		   'sheep', 'sofa', 'train', 'tvmonitor')
im_out = cv2.imread(img)
im_h = im_out.shape[0]
im_w = im_out.shape[1]
box = out[0,0,:,3:7] * np.array([im_w, im_h, im_w, im_h])
cls = out[0,0,:,1]
conf= out[0,0,:,2]
offset = 0
for i in range(len(box)):
    cv2.rectangle(im_out,(int(box[i][0]+0.5),int(box[i][1]+0.5)),(int(box[i][2]+0.5),int(box[i][3]+0.5)),(0,255,255))
    text = "%s:%.3f" % (CLASSES[int(cls[i])], conf[i])
    text_p = (int(max(box[i][0],15)+offset),int(max(box[i][1],15))+offset)
    offset+=30
    cv2.putText(im_out,text,text_p,cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
cv2.imshow("ssd",im_out)
cv2.waitKey(0)
print(box)



