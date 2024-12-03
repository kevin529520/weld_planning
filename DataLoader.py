import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
import cv2
# import skimage
import glob
import numpy as np
import random


def getInitPose(filename='init_pose.txt'):
    temp = np.loadtxt(filename)
    init_pose = np.mean(temp, axis=0)
    return init_pose


def loadData(file_list:type.List = ["./data/1.txt"]):
    train_dataset = []
    var_dataset = []
    for i in range(len(file_list)):
        temp = np.loadtxt(file_list[i])
        train_num = int(len(temp)*0.8)
        random.seed(42)
        train_sample = random.sample(list(temp),train_num)
        var_sample = list(set(temp)-set(list(train_sample)))

        train_dataset.append(train_sample)
        var_dataset.append(var_sample)

    return np.array(train_dataset), np.array(var_dataset)


class ForceDataset(Dataset):
    def __init__(self, file_list=["./data/1.txt"], data_model='train'):
        super().__init__()
        # [time,fx,fy,fz,tx,ty,tz,x,y,z,rx,ry,rz]
        self.train_data, self.var_data = loadData(file_list)
        # [x,y,z,rx,ry,rz]
        self.init_pose = getInitPose(filename='init_pose.txt')
        self.dataset_model = data_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.dataset_model == "train":
            temp_force = self.train_data[item][1:7]
            temp_pose = self.train_data[item][7:]-self.init_pose
        elif self.dataset_model == "var":
            temp_force = self.train_data[item][1:7]
            temp_pose = self.train_data[item][7:]-self.init_pose
        else:
            raise ValueError("Wrong Dataset Model Set!")
 
        return torch.Tensor(temp_pose), torch.Tensor(temp_force)



if __name__ == "__main__":
    ww = ForceDataset(file_list=["./data/1.txt"], data_model='train')
    m = DataLoader(dataset=ww, batch_size=8, shuffle=True, num_workers=2, drop_last=False, pin_memory=False, collate_fn=None)
    print(len(ww))