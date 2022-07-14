# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, BatchNormalization, Dense, Activation, \
    Conv3D, UpSampling3D, MaxPool3D, AveragePooling3D, GlobalMaxPool3D, GlobalAveragePooling3D, Add, Multiply, Lambda


# ref: https://github.com/kobiso/CBAM-tensorflow/blob/master/attention_module.py

def cbam_block(cbam_feature, ratio=8):
    cbam_feature = channel_attention(cbam_feature, ratio)
    cbam_feature = spatial_attention(cbam_feature)
    return cbam_feature


def channel_attention(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPool3D()(input_feature)
    # max_pool = tf.reshape((1, 1, channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])


def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = Lambda(lambda x: tf.math.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.math.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = tf.concat([avg_pool, max_pool], axis=-1)
    cbam_feature = Conv3D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    return Multiply()([input_feature, cbam_feature])


def BN_relu_Conv(src, dim, kernel_size):

    bn = BatchNormalization()(src)
    relu = tf.nn.relu(bn)
    conv = Conv3D(dim, kernel_size, activation=None, padding='same')(relu)

    return conv


def ResBlock(src, dim, kernel_size, attention):

    conv = Conv3D(dim, kernel_size, activation=None, padding='same')(src)
    conv1 = BN_relu_Conv(conv, dim, kernel_size)
    conv2 = BN_relu_Conv(conv1, dim, kernel_size)

    if attention:
        conv2 = cbam_block(conv2, ratio=8)

    return conv + conv2


def V_NET(data_shape, output, dim, kernel_size, attention):

    src = Input(data_shape)

    # block 1
    conv1 = ResBlock(src, dim, kernel_size, False)
    max1 = MaxPool3D((1, 2, 2), padding='same')(conv1)

    # block 2
    conv2 = ResBlock(max1, 2 * dim, kernel_size, False)
    max2 = MaxPool3D(2, padding='same')(conv2)

    # block 3
    conv3 = ResBlock(max2, 4*dim, kernel_size, False)
    max3 = MaxPool3D((1, 2, 2), padding='same')(conv3)

    # block 4
    conv4 = ResBlock(max3, 8*dim, kernel_size, attention)
    max4 = MaxPool3D(2, padding='same')(conv4)

    # block 5
    conv5 = ResBlock(max4, 16*dim, kernel_size, attention)

    # block 4
    up4 = UpSampling3D(2)(conv5)
    up4 = BN_relu_Conv(up4, dim, kernel_size)
    merge4 = tf.concat([up4, conv4], axis=-1)
    deconv4 = ResBlock(merge4, 8*dim, kernel_size, attention)

    # block 3
    up3 = UpSampling3D((1, 2, 2))(deconv4)
    up3 = BN_relu_Conv(up3, dim, kernel_size)
    merge3 = tf.concat([up3, conv3], axis=-1)
    deconv3 = ResBlock(merge3, 4*dim, kernel_size, False)

    # block 2
    up2 = UpSampling3D(2)(deconv3)
    up2 = BN_relu_Conv(up2, dim, kernel_size)
    merge2 = tf.concat([up2, conv2], axis=-1)
    deconv2 = ResBlock(merge2, 2 * dim, kernel_size, False)

    # block 1
    up1 = UpSampling3D((1, 2, 2))(deconv2)
    up1 = BN_relu_Conv(up1, dim, kernel_size)
    merge1 = tf.concat([up1, conv1], axis=-1)
    deconv1 = ResBlock(merge1, dim, kernel_size, False)

    pred = Conv3D(output, kernel_size, activation='sigmoid', padding='same')(deconv1)

    model = Model(src, pred)

    return model

