# coding: utf-8
# *****************************************************
# Aitrox Information Technology
# http://www.proxima-ai.com
# Copyright 2021 Aitrox. All rights reserved.
# CreateDate: 2021-08-27
# ******************************************************


import os
import json
from PIL import Image
from tqdm import tqdm
from sklearn import metrics
from matplotlib import pyplot as plt

import torch
from src.models import DensNet121
from src.uniform_scale import UniformScale

torch.cuda.set_device(6)
device = torch.device('cuda')
total_labels = ['benign', 'malignant']

try:
    pth_file = 'backup/2021-08-30-densnet121/model_best.pth'
    input_classes = 2
    state = torch.load(pth_file, map_location='cpu')
    model = DensNet121(input_classes).cuda()
    model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})
    model.eval()
    print('--model load sucessed--')
except Exception as e:
    print('--initilize model error:{}--'.format(e))


gt_dict = dict()
with open('labels/test_label.txt', 'r') as f:
    gt_infos = f.readlines()
    for info in gt_infos:
        info = info.strip('\n')
        serid = info.split('  ')[0]
        gt = info.split('  ')[1]
        gt_dict[serid] = gt
f.close()


# draw ROC figure
def draw_roc(y_true, y_pred_scores, class_name):
    roc_figure_output_dir = 'ROC_Figure'
    if not os.path.exists(roc_figure_output_dir):
        os.mkdir(roc_figure_output_dir)
    fpr, tpr, threshold = metrics.roc_curve(y_true=y_true, y_score=y_pred_scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(10, 10))
    plt.title(class_name + ' Test ROC')
    plt.plot(fpr, tpr, 'b', label='Test AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    plt.savefig(os.path.join(roc_figure_output_dir, class_name + ".png"))


def benchmark_imgdir(test_dir):
    pred = list()
    gt = list()
    y_scores = list()
    with open('densnet121_infer.txt', 'a+') as f:
        for dir in tqdm(os.listdir(test_dir)):
            full_dir = os.path.join(test_dir, dir)
            for img_file in os.listdir(full_dir):
                img_full_path = os.path.join(full_dir, img_file)
                img = Image.open(img_full_path)
                img = UniformScale(size=512, img=img).scale()
                img_to_tensor = torch.from_numpy(img).permute(2, 1, 0).float() / 255.0
                tensor_batch = torch.unsqueeze(img_to_tensor, 0).cuda()
                output = torch.squeeze(model(tensor_batch)).tolist()
                gt.append(int(gt_dict[dir]))
                if output[1] > 0.65:
                    pred.append(1)
                else:
                    pred.append(0)
                y_scores.append(output[1])
                f.writelines(','.join((dir, '{:.14f}'.format(output[1]), gt_dict[dir])) + '\n')

        acc = metrics.accuracy_score(y_pred=pred, y_true=gt)
        auc = metrics.roc_auc_score(y_true=gt, y_score=y_scores)
        draw_roc(y_true=gt, y_pred_scores=y_scores, class_name='neg_sig')
        print(auc)
        print(acc)
    f.close()


def benchmark_person(img_dir):
    pred = list()
    gt = list()
    y_scores = list()
    for person_name in os.listdir(img_dir):
        temp_score = list()
        temp_gt = list()
        full_dir = os.path.join(img_dir, person_name)
        for img_folder in os.listdir(full_dir):
            img_full_dir = os.path.join(full_dir, img_folder)
            for img in os.listdir(img_full_dir):
                img_full_path = os.path.join(img_full_dir, img)
                img = Image.open(img_full_path)
                img = UniformScale(size=512, img=img).scale()
                img_to_tensor = torch.from_numpy(img).permute(2, 1, 0).float() / 255.0
                tensor_batch = torch.unsqueeze(img_to_tensor, 0).cuda()
                output = torch.squeeze(model(tensor_batch)).tolist()
                temp_score.append(output[1])
                temp_gt.append(int(gt_dict[img_folder]))
        one_folder_score = max(temp_score)
        y_scores.append(one_folder_score)
        gt.append(int(temp_gt[0]))
        info = person_name + ',' + '{:.14f}'.format(one_folder_score) + ',' + str(int(temp_gt[0]))
        print(info)

    print(len(y_scores), len(gt))
    auc = metrics.roc_auc_score(y_true=gt, y_score=y_scores)
    draw_roc(y_true=gt, y_pred_scores=y_scores, class_name='person_roc')
    print(auc)


def benchmark_json(json_file):
    pred = list()
    gt = list()
    y_scores = list()
    with open(json_file, 'r') as f:
        dataes = json.load(f)
    f.close()

    for data in tqdm(dataes['samples']):
        img_full_path = data['image_name']
        serid = img_full_path.split('/')[1]
        label = data['image_labels']
        img = Image.open(img_full_path)
        img = UniformScale(size=512, img=img).scale()
        img_to_tensor = torch.from_numpy(img).permute(2, 1, 0).float() / 255.0
        tensor_batch = torch.unsqueeze(img_to_tensor, 0).cuda()
        output = torch.squeeze(model(tensor_batch)).tolist()
        gt.append(total_labels.index(label[0]))

        y_scores.append(output[1])

    auc = metrics.roc_auc_score(y_true=gt, y_score=y_scores)
    draw_roc(y_true=gt, y_pred_scores=y_scores, class_name='test_ROC')
    print(auc)


if __name__ == '__main__':
    benchmark_imgdir('test_datasets')
