# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-24
# ******************************************************

import os
try:
    import simplejson as json
except ImportError:
    import json
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class ThyroidDataset(Dataset):
    def __init__(self, anno_path, transforms, input_size):
        """
        :param data_dir: dataset_dir
        :param anno_path: json labels
        :param transforms: dataset transforms
        """
        self.transforms = transforms
        self.imgs = list()
        self.annos = list()
        self.inputsize = input_size
        # self.data_dir = data_dir

        with open(anno_path, 'r') as j:
            json_datas = json.load(j)
        samples = json_datas['samples']
        self.classes = json_datas['labels']

        for sample in samples:
            self.imgs.append(sample['image_name'])
            self.annos.append(sample['image_labels'])

        for item_id in range(len(self.annos)):
            item = self.annos[item_id]
            one_hot_vec = [cls in item for cls in self.classes]
            # one_hot_vec = [one_hot_vec[0]]
            self.annos[item_id] = np.array(one_hot_vec, dtype=float)
        
    def __getitem__(self, item):
        anno = self.annos[item]
        # img_path = os.path.join(self.data_dir+'/JPEGImages_ori', self.imgs[item])
        img_path = self.imgs[item]
        input_size = self.inputsize
        # img = Image.open(img_path)
        img = Image.open(img_path).convert("RGB")
        
        # uniform scale
        ratio = min(input_size/img.height, input_size/img.width)
        scale_img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        scale_img = np.asarray(scale_img)
        pad_img = np.zeros((input_size, input_size, 3))
        dw = int((input_size - int(img.width * ratio)) / 2)
        dh = int((input_size - int(img.height * ratio)) / 2)
        pad_img[dh:int(img.height * ratio)+dh, dw:int(img.width * ratio) + dw] = scale_img
        pad_img = Image.fromarray(np.uint8(pad_img))
        
        if self.transforms is not None:
            pad_img = self.transforms(pad_img)

        # if self.transforms is not None:
        #     img = self.transforms(img)
            
        return pad_img, anno
        # return img, anno

    def __len__(self):
        return len(self.imgs)
