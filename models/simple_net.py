from models.pretrained import *


class SimpleCNN(tf.keras.layers.Layer):
    def get_config(self):
        pass

    def __init__(self, num_class=6, pretrained='resnet'):
        super(SimpleCNN, self).__init__()
        self.pretrained_type = pretrained
        self.pretrained_cnn = pretrained_cnn(pretrained, include_top=False)
        self.conv = tf.keras.layers.Conv2D(1024, 3, padding='same', activation='relu')
        self.fc1 = tf.keras.layers.Dense(1024, activation='relu')
        self.fc2 = tf.keras.layers.Dense(256, activation='relu')
        self.clf = tf.keras.layers.Dense(num_class, activation='softmax')

    def call(self, x, train=True):
        x = cnn_preprocess_input(x, self.pretrained_type)
        x = self.pretrained_cnn(x)
        x = self.conv(x)
        x = tf.keras.layers.Dropout(0.5, trainable=train)(x)
        x = tf.keras.layers.Flatten()(x)
        x = self.fc1(x)
        x = tf.keras.layers.Dropout(0.5, trainable=train)(x)
        x = self.fc2(x)
        x = self.clf(x)
        return x
