# evaluate the unet2d model for thyroid nodule segmentation
# import libs and initialization

from keras.models import Model, load_model
from keras import backend as K
from utils.losses import dice_metric, dice_loss, dice_bce_loss, dice_bce_focal_loss
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
gpu_id = 1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
print('Using GPU {}'.format(gpu_id))


# # define loss function

def dice_coef(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)


def main():
    # predict on test data

    root = '.'
    test_path_list = 'test_path_thyroid_seg.txt'

    image_rows = 512
    image_cols = 512

    with open(os.path.join(root, test_path_list)) as tp:
        paths = tp.readlines()

    image_list = list()
    mask_gt_list = list()
    mask_pred_list = list()
    dice_list = list()
    dice_binary_list = list()

    # loss_metrics = {'dice_coef_loss':dice_coef_loss, 'dice_coef':dice_coef}
    loss_metrics = {'dice_bce_loss':dice_bce_loss, 'dice_metric':dice_metric}
    # loss_metrics = {'dice_bce_focal_loss':dice_bce_focal_loss, 'dice_metric':dice_metric}
    model = load_model(os.path.join(root,'model_thyroid_seg','experiment_name','thyroid_seg_fpn_20200423.h5'), custom_objects=loss_metrics)


    for index, path in enumerate(paths):
        path_element = path.split()

        image_ori = cv2.imread(path_element[0], cv2.IMREAD_GRAYSCALE)
        mask_ori = cv2.imread(path_element[1], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image_ori, (image_cols, image_rows), interpolation = cv2.INTER_NEAREST)
        mask_gt = cv2.resize(mask_ori, (image_cols, image_rows), interpolation = cv2.INTER_NEAREST)

        image = image/255.0
        mask_gt = mask_gt/255.0

        image_test = image[np.newaxis, :, :, np.newaxis]

        # B = A[np.repeat(np.arange(A.shape[0]), 3)]
        image_test = np.concatenate((image_test,image_test,image_test),3)
        # print('image_test',image_test.shape)
        mask_test_pred = model.predict(image_test, batch_size=1, verbose=0)

        mask_pred = mask_test_pred[0, :, :, 0]
        mask_pred_binary = mask_pred.copy()
        mask_pred_binary[mask_pred_binary>0.5] = 1
        mask_pred_binary[mask_pred_binary<=0.5] = 0

        dice = 2*np.sum(mask_pred*mask_gt)/(np.sum(mask_pred)+np.sum(mask_gt))
        dice_binary = 2*np.sum(mask_pred_binary*mask_gt)/(np.sum(mask_pred_binary)+np.sum(mask_gt))

        image_list.append(image)
        mask_gt_list.append(mask_gt)
        mask_pred_list.append(mask_pred)
        dice_list.append(dice)
        dice_binary_list.append(dice_binary)

    dice_array = np.array(dice_list)
    dice_binary_array = np.array(dice_binary_list)

    print('average dice:%f'% np.mean(dice_array))
    print('median dice:%f'% np.median(dice_array))
    print('average binary dice:%f'% np.mean(dice_binary_array))
    print('median binary dice:%f'% np.median(dice_binary_array))

    plt.figure
    plt.hist(x=dice_array, bins='auto', color='b', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Dice')
    plt.ylabel('Frequency')
    plt.title('dice histogram')
    plt.show()

    plt.figure
    plt.hist(x=dice_binary_array, bins='auto', color='b', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Dice')
    plt.ylabel('Frequency')
    plt.title('binary dice histogram')
    plt.show()

if __name__=='__main__':
    main()