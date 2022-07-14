# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np


def dice_loss(y_target, y_pred):

    A = K.sum(y_target[:, :, :, :, 0] * y_pred[:, :, :, :, 0])
    B = K.sum(y_target[:, :, :, :, 0])
    C = K.sum(y_pred[:, :, :, :, 0])

    score = (2*A) / (B + C)

    return 1-score


def dice_score(y_target, y_pred):

    temp1 = tf.cast(tf.math.greater_equal(y_target[:, :, :, :, 0], 0.5), dtype='float32')
    temp2 = tf.cast(tf.math.greater_equal(y_pred[:, :, :, :, 0], 0.5), dtype='float32')
    temp3 = tf.cast(tf.math.greater_equal(temp1*temp2, 0.5), dtype='float32')
    A = K.sum(temp3)
    B = K.sum(temp1)
    C = K.sum(temp2)


    score = (2*A + 27) / (B + C + 27)

    return score
