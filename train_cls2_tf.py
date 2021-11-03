# -*- encoding: utf-8 -*-
'''
Filename      : train_tf.py
Description   : 
Author  	  : Yang Jian
Contact 	  : lian01110@outlook.com
Time          : 2021/11/01 09:51:59
IDE           : PYTHON
REFERENCE 	  : https://github.com/yangjian1218
'''



# 导包
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# 第一步:建立命令行
# ap = argparse.ArgumentParser()
# ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
# ap.add_argument('-p', '--plot', type=str, default='plotyj.png', help='path to output loss/accuracy plot')
# ap.add_argument('-m', '--model', type=str, default='mask_detector.model',
#                 help='path to output face mask detector model')
# args = vars(ap.parse_args())

# 初始化学习率,epoch,batch_size
INIT_LR = 1e-4
EPOCHS = 10
BS = 128

print('[INFO] loading image...')
# imagePaths = list(paths.list_images(args['dataset']))   # dataset文件夹一直追溯下去,直到文件,并存到生成器中,通过list列出
imagePaths = list(paths.list_images("datasets/data"))
# print(imagePaths)
data = []
labels = []

#todo 测试一张图片的预处理
# label1 = imagePaths[0].split(os.path.sep)[-2]
# print('label1', label1)
# image = load_img(imagePaths[0], target_size=(224, 224))  # 原本是180x270
# # image1 = load_img(imagePaths[0])

# image = img_to_array(image)
# # print(image)
# # print('=====')
# image = preprocess_input(image)  # 归一化到[-1,1]
# print(image)
# print('====')
# data.append(image)
# # print(data.type)
# labels.append(label1)
# data = np.array(data, dtype='float32')
# labels = np.array(labels)
# print(data)
# print(labels)


# 循环遍历每张图,包括带口罩和不带口罩
for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]  # 按/或\分割,取倒数第二个即为图片所在的文件夹即带口罩/不戴口罩
    image = load_img(imagePath, target_size=(224, 224))  # 读取每张图片,且进行resize
    image = img_to_array(image)
    image = preprocess_input(image)  # 归一化到[-1,1]

    # 更新data和labels
    data.append(image)
    labels.append(label)
# data和label转到np array
data = np.array(data, dtype='float32')  # (1376, 224, 224, 3)  变4维了
# print(data.shape)
labels = np.array(labels)  # (1376,) 1维
# print(labels.shape)

lb = LabelBinarizer()  # 标签二值化
labels = lb.fit_transform(labels)  # 
# print(labels)
labels = to_categorical(labels)   # 戴口罩为[1,0],不戴口罩为[0,1]
# print(labels)
# 分割数据集
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=42)
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
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
# baseModel.summary()
# 冻结参数
for layer in baseModel.layers:
    layer.trainable = False

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name='flatten')(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)  # 模型组合

# ====================================================================================================
# 构建模型方式二:   此方式会把预训练模型封装
# net = Sequential([
#     MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3))),
#     AveragePooling2D(pool_size=(7, 7)),
#     Flatten(name='flatten'),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(2, activation='softmax')
# ])
# net.summary()

# output = model0.output
# index_noadd = [18, 36, 63, 98, 125]
# index_add = [26, 44, 53, 71, 80, 89, 106, 115, 133, 142]
# i = 1
# N_index = 150  # 要选择前150层
# while i < N_index:
#     if i in index_noadd:
#         output = baseModel.layers[i](output)
#         output_stay = output
#         i += 1
#     elif i in index_add:
#         output_pre = baseModel.layers[i](output)
#         # output_stay=output_pre
#         output = output_pre + output_stay
#         output_stay = output_pre
#         i += 2
#     else:
#         output = baseModel.layers[i](output)
#         i += 1

# model = Model(inputs=model0.input, outputs=output)
# model.summary()
# print('baseModel的层数为', len(baseModel.layers), '注释:第0层为空,故实际为154层')
# print('model新网络层数:', len(model.layers), '注释:第0层为空,故实际为{}层'.format(N_index - 1))
# ====================================================================================================

# 编译模型
print('[IFRO] compiling model....')
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

# 训练模型
print('[INFO] training head...')
import time
start_time =time.time()
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

end_time =time.time()
use_time = end_time-start_time
print('耗时:{}min{}s'.format(use_time//60,use_time%60))
# 模型训练后进行预测
print('[INFO] evaluating network...')
predIdxs = model.predict(testX, batch_size=BS)   # 得到戴口罩和不戴口罩的概率
predIdxs = np.argmax(predIdxs, axis=1)  # 取出概率最大值得索引

# 显示分类结果(模块自带report)
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

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
