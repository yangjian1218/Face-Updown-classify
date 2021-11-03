# -*- encoding: utf-8 -*-
'''
Filename      : train_tf.py
Description   : 人脸图片正反判断训练
Author  	  : Yang Jian
Contact 	  : lian01110@outlook.com
Time          : 2021/11/01 09:51:59
IDE           : PYTHON
REFERENCE 	  : https://github.com/yangjian1218
'''


# 导包
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array  # 视频采集的帧图像采用兼容的数据形式
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
# 第一步:建立命令行


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
    img_ret = np.zeros(shape=(input_size[1], input_size[0], 3), dtype=np.uint8)
    img_ret[:new_height, :new_width, :] = resized_img
    return img_ret


# 初始化学习率,epoch,batch_size
INIT_LR = 1e-4
EPOCHS = 30
BS = 128

print('[INFO] loading image...')
imagePaths = list(paths.list_images("datasets/data"))
# print(imagePaths)
data = []
labels = []


# 循环遍历每张图,包括带口罩和不带口罩
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]  # 按/或\分割
    image = cv2.imread(imagePath)[:, :, ::-1]
    image = resize2netsize(image)
    image = preprocess_input(image)  # 归一化到[-1,1]
    # 更新data和labels
    data.append(image)
    labels.append(label)
# data和label转到np array
data = np.array(data, dtype='float32')
print(data.shape)
labels = np.array(labels)  # (1376,) 1维
# print(labels.shape)

lb = LabelEncoder()   # 多分类的标签热编码
labels = lb.fit_transform(labels)
# print(labels)
labels = to_categorical(labels)
print(labels.shape)
# 分割数据集
(trainX, testX, trainY, testY) = train_test_split(
    data, labels, test_size=0.2, stratify=labels, random_state=42)
# stratify 按labels分布进行分层切分训练集跟测试集

# 图像增强器
aug = ImageDataGenerator(
    rotation_range=15,  # 循转范围
    zoom_range=0.15,  # 缩放比例范围
    width_shift_range=0.2,  # 水平移动范围
    height_shift_range=0.2,   # 垂直移动范围
    shear_range=0.15,   # 浮点数,剪切强度(逆时针方向的剪切变换角度)
    horizontal_flip=True,  # 水平翻转
    fill_mode='nearest')  # 像素填充方式

# 迁移学习
# 构建模型方式一:此方式会把预训练模型展开
baseModel = MobileNetV2(weights='imagenet', alpha=0.35,
                        include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# baseModel.summary()
# 冻结参数
for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(3, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)  # 模型组合
model.summary()
# 编译模型
print('[IFRO] compiling model....')
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='categorical_crossentropy',
              optimizer=opt, metrics=['accuracy'])  # 多分类


# 训练模型
print('[INFO] training head...')
start_time = time.time()
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

end_time = time.time()
use_time = end_time-start_time
print('耗时:{}min{}s'.format(use_time//60, use_time % 60))
# 模型训练后进行预测
print('[INFO] evaluating network...')
predIdxs = model.predict(testX, batch_size=BS)   # 得到戴口罩和不戴口罩的概率
predIdxs = np.argmax(predIdxs, axis=1)  # 取出概率最大值得索引

# 显示分类结果(模块自带report)
print(classification_report(testY.argmax(axis=1),
                            predIdxs, target_names=lb.classes_))

# 保存模型
print('[INFO] saving mask detector model...')
# model.save(args['model'],save_format='h5')
model.save('mask_detector.model', save_format='h5')

# 绘制模型指标
N = EPOCHS
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, N), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, N), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, N), H.history['accuracy'], label='train_acc')
plt.plot(np.arange(0, N), H.history['val_accuracy'], label='val_acc')
plt.title('training Loss and accuracy')
plt.xlabel('Epoch#')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower left')
# plt.savefig(args['plot'])
plt.savefig('plotyj.png')
