# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-25
# ******************************************************
"""
json contents：
[
  'samples':
     [
        {'image_name':***.jpg,'image_labels':['','',...]},
        ...
     ]
  'labels':['', '']
]
"""

import os
try:
    import simplejson as json
except ImportError:
    import json

labels_dict = {'neg': 'benign', 'pos': 'malignant'}
labels = ['malignant']
total_tag = list()
total_label = dict()

with open('labels/ori_labels.txt', 'r') as f:
    infos = f.readlines()
    for info in infos:
        info = info.strip('\n')
        serid = info.split(' ')[0]
        label = info.split(' ')[1].split('_')[0]
        total_label[serid] = label
f.close()


def generate_json_file(txt_file):
    json_file_name = txt_file.split('/')[1].split('.')[0]
    with open(txt_file, 'r') as f:
        img_paths = f.readlines()
        for path in img_paths:
            path = path.strip('\n')
            tag = dict()
            serid = path.split('/')[1]
            tag['image_name'] = path
            tag['image_labels'] = [labels_dict[total_label[serid]]]
            total_tag.append(tag)
    f.close()
    json_content = {'samples': total_tag, 'labels': labels}

    with open(json_file_name + '.json', 'w') as j:
        json.dump(json_content, j)
    j.close()


if __name__ == '__main__':
    generate_json_file(txt_file='')
