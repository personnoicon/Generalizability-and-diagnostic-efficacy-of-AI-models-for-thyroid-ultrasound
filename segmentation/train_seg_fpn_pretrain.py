# train model to segment thyroid nodule based on ultrasound image 
# with data augmentation
# import libs and initialization

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from keras.optimizers import Adam
from models.model import resnet50_fpn, densenet121_fpn, densenet121_fpn_thyroid_cls, densenet121_fpn_thyroid_cls_video
from utils.losses import dice_metric, dice_loss, dice_bce_loss, dice_bce_focal_loss
import numpy as np
import cv2
import os
# set specific GPU and limit GPU memory
gpu_id = 0,1
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
print('Using GPU {}'.format(gpu_id))


# create data generator

def data_generator(batch_size, paths, imgW, imgH):

    while True:
        image_batch = list()
        mask_batch = list()

        for index, path in enumerate(paths):
            # try:
            path_element = path.split()

            image_ori = cv2.imread(path_element[0], cv2.IMREAD_GRAYSCALE)
            mask_ori = cv2.imread(path_element[1], cv2.IMREAD_GRAYSCALE)

            image_resize = cv2.resize(image_ori, (imgW, imgH), interpolation = cv2.INTER_NEAREST)
            mask_resize = cv2.resize(mask_ori, (imgW, imgH), interpolation = cv2.INTER_NEAREST)

            image_resize = image_resize/255.0
            mask_resize = mask_resize/255.0

            image = image_resize[:, :, np.newaxis]
            image_3chn = np.concatenate([image, image, image], axis=-1)
            mask = mask_resize[:, :, np.newaxis]
            mask_3chn = np.concatenate([mask, mask, mask], axis=-1)

            image_batch.append(image_3chn)
            mask_batch.append(mask_3chn)

            if len(image_batch) == batch_size:

                yield np.array(image_batch), np.array(mask_batch)

                image_batch = list()
                mask_batch = list()
            # except:
            # continue


# train model

def main():

    study = 'experiment_name'
    imgW = 512
    imgH = 512
    lr = 1e-4
    epochs = 500
    batch_size = 1
    aug_factor = 2
    reduce_lr_patient = 10
    reduce_lr_factor = 0.8
    early_stop_patient = 21
    root = '/home/proxima-sx11/zhaokeyang/Thyroid_github/segmentation/'
    model_dir = os.path.join(root,'model_thyroid_seg/{}'.format(study))
    img_train_aug_dir = os.path.join(root,'data','image')
    mask_train_aug_dir = os.path.join(root,'data','mask')
    python_file = 'train_seg_fpn_pretrain'
    model_type = densenet121_fpn
    loss_type = dice_bce_loss #or  dice_bce_focal_loss
    augmentation = 'yes'
    project = 'thyroid_nodule_seg'


    config = {
        'study': study,
        'imgW': imgW,
        'imgH': imgH,
        'learning_rate': lr,
        'epochs': epochs,
        'batch_size': batch_size,
        'aug_factor': aug_factor,
        'reduce_lr_patient': reduce_lr_patient,
        'reduce_lr_factor': reduce_lr_factor,
        'early_stop_patient': early_stop_patient,
        'data_path_root': root,
        'model_dir': model_dir,
        'img_train_aug_dir': img_train_aug_dir,
        'mask_train_aug_dir': mask_train_aug_dir,
        'python_file': python_file,
        'model_type': model_type,
        'loss_type': loss_type,
        'augmentation': augmentation,
        'project': project,
    }

    model_type_str = '{}'.format(model_type).split()[1]
    loss_type_str = '{}'.format(loss_type).split()[1]

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    #     model_path = os.path.join(model_dir, 'seg_{}_lr-{}_bs-{}_'.format(model_type_str, lr, batch_size)+
    #                               'ep-{epoch:03d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.hdf5')
    model_path = os.path.join(model_dir, 'thyroid_seg_fpn_20200423.h5')

    with open(os.path.join(model_dir, 'config.txt'), 'w') as fout:
        fout.write(str(config))

    print('root:{}, aug_dir:{}, model_dir:{}'.format(root, img_train_aug_dir, model_dir))
    print('project:{}, py:{}, model:{}, loss:{}, study:{}, lr:{}, epochs:{}, batchsize:{}'.format
          (project, python_file, model_type_str, loss_type_str, study, lr, epochs, batch_size))

    input_shape = (imgH, imgW, 3)

    model = model_type(input_shape, channels=input_shape[-1], activation="sigmoid")
    model.compile(optimizer=Adam(lr=lr), loss=loss_type, metrics=[dice_metric])

    model_checkpoint = ModelCheckpoint(model_path, monitor='val_dice_metric',verbose=1, save_best_only=True, mode='max', period=1)
    csv_logger = CSVLogger(filename=os.path.join(model_dir, 'training_log.csv'), separator=',', append=False)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=reduce_lr_factor, patience=reduce_lr_patient, min_lr=1e-20, mode='min', verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=early_stop_patient, mode='min', verbose=1)
    callbacks_list = [model_checkpoint, csv_logger, reduce_lr, early_stopping]

    with open(os.path.join(root, 'train_path_thyroid_seg.txt')) as fin:
        train_paths = fin.readlines()
    with open(os.path.join(root, 'val_path_thyroid_seg.txt')) as fin:
        val_paths = fin.readlines()


    train_steps = len(train_paths)/batch_size
    val_steps = len(val_paths)/batch_size

    train_generator = data_generator(batch_size, train_paths, imgW, imgH)
    val_generator = data_generator(batch_size, val_paths, imgW, imgH)

    H = model.fit_generator(train_generator,
                            steps_per_epoch=train_steps,
                            epochs=epochs,
                            callbacks=callbacks_list,
                            validation_data=val_generator,
                            validation_steps=val_steps,
                            verbose=1)

    print('root:{}, aug_dir:{}, model_dir:{}'.format(root, img_train_aug_dir, model_dir))
    print('project:{}, py:{}, model:{}, loss:{}, study:{}, lr:{}, epochs:{}, batchsize:{}'.format
          (project, python_file, model_type_str, loss_type_str, study, lr, epochs, batch_size))


# execute main function

if __name__ == '__main__':
    main()
