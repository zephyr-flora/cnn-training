import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import h5py

"""
PIL 4.3.0 (pip install pillow)
numpy 1.13.1
sklearn: 0.19.1
"""
if not os.path.exists('./data_set'):
    os.mkdir('./data_set')


def create_h5():
    x, y = [], []
    
    for i, image_path in enumerate(os.listdir('./images')):
        #label
        label = int(image_path.split('_')[0])
        y.append(label)
        
        img = Image.open('./images/{}'.format(image_path)).convert('L');
        imgData = 1 - np.reshape(img, 784) / 255.0
        x.append(imgData)
    
    if not os.path.exists('./data_set/1.h5'):
        with h5py.File('./data_set/1.h5') as f:
            f['data'] = x
            f['labels'] = y
            
def read_h5():
    with h5py.File('./data_set/1.h5','r') as f:
        print(f['labels'][:])


def make_npy_data_set():
    x, y = [], []

    for i, image_path in enumerate(os.listdir('./images')):
        # label转为独热编码后再保存
        label = int(image_path.split('_')[0])
        #print(label,"====")
        #label_one_hot = [0 if i != label else 1 for i in range(10)]
        #y.append(label_one_hot)
        y.append(label)

        # 图片像素值映射到 0 - 1之间
        image = Image.open('./images/{}'.format(image_path)).convert('L')
        # 28x28 = 784;   rgb < 256
        image_arr = 1 - np.reshape(image, 784) / 255.0
        x.append(image_arr)

    #print(len(x),"\n","===","\n",len(y),y)

    np.save('data_set/X.npy', np.array(x))
    np.save('data_set/Y.npy', np.array(y))


class DataSet:
    def __init__(self):
        x, y = np.load('data_set/X.npy'), np.load('data_set/Y.npy')

        self.train_x, self.test_x, self.train_y, self.test_y = \
            train_test_split(x, y, test_size=0.1, random_state=0)

        self.train_size = len(self.train_x)

    def get_train_batch(self, batch_size=64):
        # 随机获取batch_size个训练数据
        choice = np.random.randint(self.train_size, size=batch_size)
        batch_x = self.train_x[choice, :]
        batch_y = self.train_y[choice, :]

        return batch_x, batch_y

    def get_test_set(self):
        return self.test_x, self.test_y


if __name__ == '__main__':
    #make_npy_data_set()

    #data_set = DataSet()
    #print(len(data_set.train_x),"\n","===","\n",len(data_set.train_y))
    #data_set.train_x = data_set.train_x.reshape((180, 28, 28, 1))
    #data_set.test_x = data_set.test_x.reshape((20, 28, 28, 1))
    create_h5()
    read_h5()

