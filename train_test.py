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
from sklearn.preprocessing import LabelBinarizer,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os


# 初始化学习率,epoch,batch_size
INIT_LR = 1e-4
EPOCHS = 10
BS = 128

print('[INFO] loading image...')
# imagePaths = list(paths.list_images(args['dataset']))   # dataset文件夹一直追溯下去,直到文件,并存到生成器中,通过list列出
imagePaths = list(paths.list_images("datasets/test"))
# print(imagePaths)
data = []
labels = []

#todo 测试一张图片的预处理
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
print(data.shape)
labels = np.array(labels)  # (1376,) 1维
print(labels.shape)

# lb = LabelBinarizer()  # 标签二值化
lb = LabelEncoder()  
labels = lb.fit_transform(labels)  # 带口罩为1,不戴口罩为0
print(labels)
labels = to_categorical(labels)   # 戴口罩为[1,0],不戴口罩为[0,1]
# print("labels:",labels)
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