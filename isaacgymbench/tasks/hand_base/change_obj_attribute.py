import os
import re
import numpy as np
from scipy.io import loadmat

class Obj_attribute:
    def __init__(self, num_envs, num_task_with_random=0):

        # self.file = '../assets/' + file

        self.num_task_with_random = num_task_with_random

        self.masses = np.random.standard_normal(num_envs) # np.random.rand(num_envs) * (0.1-0.01) + 0.01  ////  np.random.randint(low=1, high=100, size=num_envs) * 1e-3 
        self.masses = np.around((self.masses/np.max(np.abs(self.masses)) + 1)/2 * (0.1-0.01) + 0.01 ,3)
        
        self.ixx = np.random.standard_normal(num_envs)  # np.random.rand(num_envs) * (2e-4 - 5e-5) + 5e-5  ////  np.random.randint(low=5, high=20, size=num_envs) * 1e-5
        self.ixx = np.around((self.ixx/np.max(np.abs(self.ixx)) + 1)/2 * (2e-4 - 5e-5) + 5e-5 ,5)
        
        self.ixy = np.zeros(num_envs)
        self.ixz = np.zeros(num_envs)

        self.iyy = np.random.standard_normal(num_envs)  # np.random.rand(num_envs) * (2e-4 - 5e-5) + 5e-5  ////  np.random.randint(low=5, high=20, size=num_envs) * 1e-5
        self.iyy = np.around((self.iyy/np.max(np.abs(self.iyy)) + 1)/2 * (2e-4 - 5e-5) + 5e-5 ,5)
        
        self.iyz = np.zeros(num_envs)

        self.izz = np.random.standard_normal(num_envs)  # np.random.rand(num_envs) * (2e-4 - 5e-5) + 5e-5  ////  np.random.randint(low=5, high=20, size=num_envs) * 1e-5
        self.izz = np.around((self.izz/np.max(np.abs(self.izz)) + 1)/2 * (2e-4 - 5e-5) + 5e-5 ,5)

        self.point_info = loadmat('../assets/2dim_feature.mat')
        self.point_feature = self.point_info['feature_ycb_2dim']
        self.point_label = self.point_info['label_ycb']

        self.obj_label = {
            # "block", 
            # "egg", 
            # "pen", 
            "mug":  43,
            "poker": 60,
            "banana": 51,
            "clamp": 41,
            "stapler": 85,
            "suger": 90,
            "bowl": 17,
            "pie": 45,
            # "pen_container", 
            "apple": 50,
            "peach": 54,
            "pear": 55,
            "strawberry": 57,
            "washer": 4,
            "pen": 3,
            "bottle": 16,
            "scissors": 74,
            "bluecup": 2,
            "plate": 5,
            "teabox": 6,
            "clenser": 18,
            "conditioner": 22,
            "correctionfluid": 23,
            "crackerbox": 24, 
            "doraemonbowl": 25,
            "largeclamp": 27,
            "flatscrewdrive": 28,
            "fork": 29,
            "glue": 31,
            "liption": 39,
            "lemon": 52,
            "orange": 53,
            "remotecontroller": 67,
            "sugerbox": 87,
            "repellent": 69,
            "shampoo": 75,
            }
        

    def alter(self, file, old_str, new_str):
        """
        替换文件中的字符串
        :param file:文件路径
        :param old_str:旧字符串
        :param new_str:新字符串
        :return:
        """
        file = '../assets/' + file
        f=open(file,'r')
        alllines=f.readlines()
        f.close()
        f=open(file,'w+')
        for eachline in alllines:
            a=re.sub(old_str,new_str,eachline)
            f.writelines(a)
        f.close()
    

