### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
import cv2
import torch
import numpy as np
import pandas as pd
import scipy.io as scio

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot    
        
        # input A (input images)
        self.dir_A = os.path.join(opt.dataroot,opt.phase,'images/')
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (ground truth)
        self.dir_B = os.path.join(opt.dataroot,opt.phase,'ground_truth/')  
        self.B_paths = sorted(make_dataset(self.dir_B))

        if opt.roi:
            self.dir_roi = os.path.join(opt.dataroot,'mask.mat')

        self.dataset_size = len(self.A_paths) 
      
    def __getitem__(self, index):        
        A_path = self.A_paths[index]              
        A = cv2.imread(A_path,0)
        A = A.astype(np.float32,copy=False)
        ht = A.shape[0]
        wd = A.shape[1]
        ht_1 = (ht/64)*64
        wd_1 = (wd/64)*64
        A = cv2.resize(A,(wd_1,ht_1))
        A = A.reshape((1,A.shape[0],A.shape[1]))
        A = torch.from_numpy(A).type(torch.FloatTensor)
        # print A.size()

        B = roi_tensor = feat_tensor = 0

        B_path = self.B_paths[index]   
        B = pd.read_csv(B_path, sep=',',header=None).as_matrix()
        B = B.astype(np.float32,copy=False)
        B = cv2.resize(B,(wd_1,ht_1))
        B = B * ((wd*ht)/(wd_1*ht_1))
        B = B.reshape((1,B.shape[0],B.shape[1]))
        B = torch.from_numpy(B).type(torch.FloatTensor)

        if self.opt.roi:
            roi = scio.loadmat(self.dir_roi)
            mask = np.zeros((158,238,1),dtype=np.uint8)
            mask = cv2.fillPoly(mask,roi['density'].astype(int),(1,1,1))
            mask = mask[:,:,0]
            mask = cv2.resize(mask,(wd_1,ht_1))
            mask = mask.reshape(1,mask.shape[0],mask.shape[1])
            roi_tensor = torch.from_numpy(mask).type(torch.FloatTensor)
        input_dict = {'label': A, 'roi': roi_tensor, 'image': B,'feat':feat_tensor,'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'AlignedDataset'