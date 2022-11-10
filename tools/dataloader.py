import os
import random
import cv2
import numpy as np
# 实现自己的dataloader

# 实现dataset类  必须最少实现__init__  __getitem__ __len__三个方法
class dataset():
    def __init__(self,image_path,label_path):
        self.image_paths = os.path.listdir(image_path)
        self.label_paths = os.path.listdir(image_path)


    def __getitem__(self,index):
        image = cv2.imread(self.image_paths[index])
        label = cv2.imread(self.label_paths[index])
        return image,label

    def __len__(self):
        return len(self.image_paths)

# 实现dataloaderiter类
class DataloaderIter():
    
    def __init__(self,dataloader):
        self.dataloader = dataloader
        self.indexs = list(range(self.dataloader.count_data))
        self.cursor = 0
        if self.dataloader.shuffle:
            random.shuffle(self.indexs)

    def merge_to(self,container,b):
        if len(container) == 0:
            for index,data in enumerate(b):
                if isinstance(data,np.ndarray):
                    container.append(data)
                else:
                    container.append(np.array([data],dtype = type(data)))
        else:
            for index,data in enumerate(b):
                container[index] = np.vstack((container[index],data))
        return container

    def __next__(self):
        if self.cursor >= self.dataloader.count_data:
            raise StopIteration()

        batch_data = []
        remain = min(self.dataloader.batch_size,self.dataloader.count_data - self.cursor)
        for n in range(remain):
            index = self.indexs[self.cursor]
            data = self.dataloader.dataset[index]
            batch_data = self.merge_to(batch_data,data)
            self.cursor += 1
        return batch_data

# 实现dataloader类

class Dataloader():
    def __init__(self,dataset,batch_size,shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.count_data = len(self.dataset)

    def __iter__(self):
        return DataloaderIter(self)
