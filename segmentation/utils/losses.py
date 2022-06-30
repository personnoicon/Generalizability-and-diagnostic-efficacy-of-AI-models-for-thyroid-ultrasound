# define loss function

from keras import backend as K
import tensorflow as tf


def dice_metric(y_true, y_pred, smooth=1e-3):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1.-dice_metric(y_true, y_pred)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred))

def dice_bce_loss(y_true, y_pred, w_dice=0.5, w_bce=0.5):
    return binary_crossentropy(y_true, y_pred) * w_bce + dice_loss(y_true, y_pred) * w_dice

def focal_loss(y_true, y_pred):
    gamma = 0.5
    alpha =0.8    
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999) 
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def focal_loss1(y_true, y_pred):
    gamma = 0.5
    alpha = 0.5
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    p_t = tf.where(K.equal(y_true, 1), y_pred, 1-y_pred)
    alpha_factor = K.ones_like(y_true)*alpha
    alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1-alpha_factor)
    cross_entropy = -K.log(p_t)
    weight = alpha_t * K.pow((1-p_t), gamma)
    return K.mean(weight * cross_entropy)

def focal_loss2(y_true, y_pred):
    gamma = 2.0
    alpha = 0.5
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    fl_1 = -K.mean(alpha*K.pow(1.- pt_1, gamma)*K.log(pt_1+K.epsilon()))
    fl_0 = -K.mean((1- alpha)*K.pow(pt_0, gamma)*K.log(1.- pt_0+K.epsilon()))
    return (fl_1+fl_0)

def dice_bce_focal_loss(y_true,y_pred):
    alpha=0.05
    return alpha * focal_loss(y_true,y_pred) - K.log(dice_bce_loss(y_true, y_pred, w_dice=0.7, w_bce=0.3))

def dice_bce_focal_loss1(y_true,y_pred):
    alpha=0.0005
    return alpha * focal_loss(y_true,y_pred) - K.log(dice_bce_loss(y_true, y_pred, w_dice=0.9, w_bce=0.1))
