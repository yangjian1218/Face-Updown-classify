# -*- encoding: utf-8 -*-
'''
Filename      : tools.py
Description   : 工具包
Author  	  : Yang Jian
Contact 	  : lian01110@outlook.com
Time          : 2021/11/03 11:15:00
IDE           : PYTHON
REFERENCE 	  : https://github.com/yangjian1218
'''
import cv2
import numpy as np


def laplacian_score(gray_img):
    """
    计算灰度图的拉普拉斯值
    :param gray_img:灰度图
    :return:
    """
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def blur_detector_score(gray, gray_blur, blur_threshold=6):
    """
    description: (1) for laplacian, if value >= 7, get 10 score, if 5<= value < 7, score = 10 * value / 7,
                     if value < 5, score = 10 * value * 0.8 / 7 (value = lap(img_gray) / lap(img_gray_blur));
    :param gray: original image of gray
    :param gray_blur: gray image after 1 cv2.GaussianBlur(gray_img, (3, 3), 1, 0)
    :param blur_threshold: api blur threshold
    :return score: blur assessment score for the image
    """
    try:
        lap_change_intense = laplacian_score(gray) / laplacian_score(gray_blur)
        if lap_change_intense >= 7:
            score = 10 - np.random.uniform() * .01
        elif 5 <= lap_change_intense < 7:
            score = 10 * lap_change_intense / 7
        else:
            score = 0.8 * 10 * lap_change_intense / 7

        if score >= blur_threshold:
            blur_status = "照片清晰度正常！"
        else:
            blur_status = "照片过于模糊！"
    except:
        score = 0.0
        blur_status = "照片过于模糊！"
    return min(score, 10), blur_status

def resize2netsize(img, input_size=(224, 224)):
    im_ratio = float(img.shape[0]) / img.shape[1]  # 图片的高/宽
    model_ratio = float(input_size[1]) / input_size[0]  # 图片与网络输入的 高/宽
    if im_ratio > model_ratio:
        new_height = input_size[1]  # 原图改高=设置高
        new_width = int(new_height / im_ratio)  # 宽进行缩小
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / img.shape[0]  # 缩放比例
    resized_img = cv2.resize(img, (new_width, new_height))
    img_ret=np.zeros(shape=(input_size[1],input_size[0],3),dtype=np.uint8)
    img_ret[:new_height,:new_width,:]=resized_img
    return img_ret