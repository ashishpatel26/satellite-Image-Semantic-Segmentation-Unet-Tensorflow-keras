import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *

def conv_block(input, filters):
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(input)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def up_conv(input, filters):
    out = UpSampling2D()(input)
    out = Conv2D(filters, kernel_size=(3,3), strides=1, padding='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    return out

def UNet(nClasses, input_height=304, input_width=304):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """
    inputs = Input(shape=(input_height, input_width, 3))
    
    n1 = 16
    filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
    conv1 = conv_block(inputs, n1)

    conv2 = MaxPooling2D(strides=2)(conv1)
    conv2 = conv_block(conv2, filters[1])

    conv3 = MaxPooling2D(strides=2)(conv2)
    conv3 = conv_block(conv3, filters[2])

    conv4 = MaxPooling2D(strides=2)(conv3)
    conv4 = conv_block(conv4, filters[3])

    conv5 = MaxPooling2D(strides=2)(conv4)
    conv5 = conv_block(conv5, filters[4])

    d5 = up_conv(conv5, filters[3])
    d5 = Concatenate()([conv4, d5])

    d4 = up_conv(d5, filters[2])
    d4 = Concatenate()([conv3, d4])
    d4 = conv_block(d4, filters[2])

    d3 = up_conv(d4, filters[1])
    d3 = Concatenate()([conv2, d3])
    d3 = conv_block(d3, filters[1])

    d2 = up_conv(d3, filters[0])
    d2 = Concatenate()([conv1, d2])
    d2 = conv_block(d2, filters[0])

    outputs = Conv2D(nClasses, (3, 3), padding='same')(d2)

    unet = Model(inputs=inputs, outputs=outputs)
    unet.compile(optimizer=Adam(lr=1e-4), 
              loss='binary_crossentropy', 
              metrics=['accuracy', 
                      tf.keras.metrics.MeanIoU(num_classes=2),
                      tf.keras.metrics.AUC(), 
                      tf.keras.metrics.Precision(top_k=5),
                      tf.keras.metrics.Recall(top_k=5),
                      ])

    return unet

if __name__ == '__main__':
  model = UNet(nClasses = 1)
  model.summary()
  