# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-25
# ******************************************************


import os
import shutil
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def create_dst_dir(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        print('---Created folder：{}---'.format(dst_dir))
    return


def copy_file(src_file, dst_dir):
    try:
        shutil.copy(src_file, dst_dir)
    except Exception as e:
        print('---{}---'.format(e))


def create_txt_file(txt_name, msg, dst_dir):
    try:
        with open(os.path.join(dst_dir, txt_name), 'a+') as f:
            f.writelines(msg + '\n')
        f.close()
        print('---{} created----'.format(txt_name))
    except Exception as e:
        print(e)


def calculate_metrics_binary_classification(pred, target, threshold=0.6):
    pred = np.array(pred > threshold, dtype=float)
    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'micro/f1': f1_score(y_true=target, y_pred=pred, average='micro'),

            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'macro/f1': f1_score(y_true=target, y_pred=pred, average='macro')}
