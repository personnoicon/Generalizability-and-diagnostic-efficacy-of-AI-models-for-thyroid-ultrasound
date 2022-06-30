# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-25
# ******************************************************

import os

dataset_dir = 'datasets'
exists_serid = os.listdir(dataset_dir)
with open('val.txt', 'a+') as t:
    with open('labels/ori_labels.txt', 'r') as f:
        infos = f.readlines()
        for info in infos:
            info = info.strip('\n')
            serid = info.split(' ')[0]
            label = info.split(' ')[1].split('_')[0]
            data_cls = info.split(' ')[1].split('_')[1]
            if data_cls == 'valid':
                try:
                    if serid in exists_serid:
                        img_dir = os.path.join(dataset_dir, serid)
                        img_files = os.listdir(img_dir)
                        for file in img_files:
                            img_full_path = os.path.join(img_dir, file)
                            t.writelines(img_full_path + '\n')
                except FileNotFoundError as e:
                    pass 
    f.close()
t.close()