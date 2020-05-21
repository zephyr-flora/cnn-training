import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
#from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

'''
python 3.7
tensorflow 2.0.0b0
'''


class CNN(object):
    def __init__(self):
        model = models.Sequential()
        # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第2层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        # 第3层卷积，卷积核大小为3*3，64个
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(10, activation='softmax'))

        model.summary()

        self.model = model
        
class AlexNet(object):
    def __init__(self):
        # 创建模型序列
        model = models.Sequential()
        #第一层卷积网络，使用96个卷积核，大小为11x11步长为4， 
        #要求输入的图片为227x227， 3个通道，不加边，激活函数使用relu
        model.add(layers.Conv2D(96, 
                                (11, 11), 
                                strides=(1, 1), 
                                input_shape=(28, 28, 1), 
                                padding='same', 
                                activation='relu',
                                kernel_initializer='uniform'))
        # 池化层
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 第二层加边使用256个5x5的卷积核，加边，激活函数为relu
        model.add(layers.Conv2D(256, 
                                (5, 5), 
                                strides=(1, 1), 
                                padding='same', 
                                activation='relu', 
                                kernel_initializer='uniform'))
        #使用池化层，步长为2
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        # 第三层卷积，大小为3x3的卷积核使用384个
        model.add(layers.Conv2D(384, 
                                (3, 3), 
                                strides=(1, 1), 
                                padding='same', 
                                activation='relu', 
                                kernel_initializer='uniform'))
        # 第四层卷积,同第三层
        model.add(layers.Conv2D(384, 
                                (3, 3), 
                                strides=(1, 1), 
                                padding='same', 
                                activation='relu', 
                                kernel_initializer='uniform'))
        # 第五层卷积使用的卷积核为256个，其他同上
        model.add(layers.Conv2D(256, 
                                (3, 3), 
                                strides=(1, 1), 
                                padding='same', 
                                activation='relu', 
                                kernel_initializer='uniform'))
        model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        
        model.add(layers.Flatten())
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(4096, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        model.summary()
        self.model = model

class DataSource(object):
    def __init__(self):
        # mnist数据集存储的位置，如何不存在将自动下载
        data_path = os.path.abspath(os.path.dirname(__file__)) + '/../data_set_tf2/mnist.npz'
        (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data(path=data_path)
        
        #print(len(train_images),"\n",train_images,"===",len(test_images),"\n",test_images)
        
        # 6万张训练图片，1万张测试图片
        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels

    
class DataSet:
    def __init__(self):
        data_path_x = os.path.abspath('../data_set_tf2/X.npy') 
        data_path_y = os.path.abspath('../data_set_tf2/Y.npy') 
        x, y = np.load(data_path_x), np.load(data_path_y)
        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(x, y, test_size=0.1, random_state=0)

        self.train_size = len(self.train_x)
        # 160张训练图片，160张测试图片
        self.train_x = self.train_x.reshape((180, 28, 28, 1))
        self.test_x = self.test_x.reshape((20, 28, 28, 1))


class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()
        print(len(self.data.train_labels), "====", self.data.train_labels)

    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images, self.data.train_labels,
                           epochs=5, callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(
            self.data.test_images, self.data.test_labels)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))

class TrainOne(object):
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSet()
        print(len(self.data.train_y), "====", self.data.train_y)
        
    def train(self):
        check_path = './ckpt/cp-{epoch:04d}.ckpt'
        # period 每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(
            check_path, save_weights_only=True, verbose=1, period=5)

        self.cnn.model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_x, self.data.train_y,
                           epochs=5, callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(
            self.data.test_x, self.data.test_y)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_y)))

if __name__ == "__main__":
    #app = Train()
    #app.train()
    app1 = TrainOne()
    app1.train()
    