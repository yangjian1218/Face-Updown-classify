# -*- coding: utf-8 -*-
# @Author: Jerry
# @Date:   2020-10-15 15:08:34
# @Last Modified by:   Jerry
# @Last Modified time: 2020-10-18 17:26:50
# 注: 该代码只能检测一张脸,原因在升维那个代码有错,在另检测脸关于维度测试.py中已经改过来了
# 用法
# python detect_mask_imageyj.py --image examples/0pic_1.jpg

# 导包
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import cv2
import os
import time
from tensorflow.keras.preprocessing.image import img_to_array  # 视频采集的帧图像采用兼容的数据形式
from tools import *

# from numba import jit
# 构建命令行
ap = argparse.ArgumentParser()

args = vars(ap.parse_args())


class FaceUpdown():
    def __init__(self,modelpath):
        self.model= load_model(modelpath)
    def prepare(self):
        img= np.zeros(shape=(1,224,224,3),dtype=np.int8)
        img = preprocess_input(img)  # 图像标准化到[-1,1]
        self.model.predict(img)
    def predict(self,img):
        ret = self.model.predict(img)
        return ret

modelpath ='model/face_updown_cls3_0.979.model'
model = FaceUpdown(modelpath)
model.prepare()



def infer_img(imgpath):
    imgname = os.path.split(imgpath)[-1]
    image = cv2.imread(imgpath)[:, :, ::-1]

    #查看清晰度
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray_blur = cv2.GaussianBlur(gray.copy(), (3, 3), 1, 0)
    # blur_score, blur_status = blur_detector_score(
    #     gray, gray_blur, blur_threshold=6)  # 清晰度值
    # print("blur_score={},blur_status={}".format(blur_score,blur_status))

    image= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = resize2netsize(image)
    image = preprocess_input(image)  # 归一化到[-1,1]
    face = np.expand_dims(image, axis=0)  # 扩展到4维 ,因为深度学习的图片格式为(n,H,W,c)
    time1= time.time()
    ret = model.predict(face)[0]
    (up, down,uncertain)=ret
    print("up={},down={},uncertain={}".format(up,down,uncertain))
    max_index = np.argmax(ret)
    label_list=['Up','Down','Uncertain']
    label = label_list[max_index]
    print("{} {}: {:.2f}%".format(imgname, label, max(down, up, uncertain) * 100))
    usedtime2=time.time()-time1
    # print("人像正反识别用时:{}ms".format(usedtime2*1000))
    return label
def infer_imgs(dir,updown='Up'):
    TrueNum=0
    imglist= os.listdir(dir)
    for img in imglist:
        imgpath =  os.path.join(dir,img)
        label = infer_img(imgpath)
        if label==updown:
            TrueNum+=1

    print("总共:{}张图片,识别正确{}张,准确率为{}".format(len(imglist),TrueNum,TrueNum/len(imglist)))

imgpath = r"D:\AI\Face\Face-Updown-classify\images\up\00098.jpg"
# infer_img(imgpath)

dir ='./images/down'
infer_imgs(dir,updown='Down')