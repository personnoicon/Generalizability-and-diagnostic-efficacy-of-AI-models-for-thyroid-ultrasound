# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-09-02
# ******************************************************

from sklearn import metrics
from numpy.core.fromnumeric import sort


pros = list()
img_info = dict()
with open('infer_res.txt', 'r') as f:
    for data in f.readlines():
        data = data.strip('\n')
        info = data.split(',')
        serid, pro, gt = info[0], info[1], info[2]
        img_info[serid] = [float(pro), int(gt)]
        pros.append(float(pro))
f.close()


def calculate_confusion_matrix(y_pred, y_true):
    tn, fp, fn, tp = metrics.confusion_matrix(y_pred=y_pred, y_true=y_true).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    youden_index = (sensitivity + specificity) - 1.0

    return youden_index, sensitivity, specificity


def calculate_youden_index():
    thres = sort(pros)
    # youd_index = dict()
    youd_index = list()
    sens = list()
    specs = list()
    for thre in thres:
        pre_cls = list()
        img_gt = list()
        for data in img_info:
            value = img_info[data]
            pro = value[0]
            img_gt.append(value[1])
            if pro > thre:
                pre_cls.append(1)
            else:
                pre_cls.append(0)

        youden, sen, spec = calculate_confusion_matrix(y_pred=pre_cls, y_true=img_gt)
        # youd_index[thre] = youden
        youd_index.append(youden)
        sens.append(sen)
        specs.append(spec)

    max_yd_index = youd_index.index(max(youd_index))
    se = sens[max_yd_index]
    sp = specs[max_yd_index]
    max_yd_value = max(youd_index)
    th = thres[max_yd_index]

    print("sensitivity:{:.4f}, specificity:{:.4f}, max youden_index:{:.4f}, thres:{:.4f}".format(se, sp, max_yd_value,
                                                                                                 th))

    return None


if __name__ == '__main__':
    calculate_youden_index()
