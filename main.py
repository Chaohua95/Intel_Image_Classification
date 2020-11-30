import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
from utils.data_creator import DataLoader
from models.simple_net import SimpleCNN

EPOCHS = 15
BATCH_SIZE = 128
IMAGE_AUG = True
TRAIN_DIR = './data/seg_train'
TEST_DIR = './data/seg_test'
PRED_DIR = './data/seg_pred'

train_set = DataLoader(TRAIN_DIR).create_dataset()
batched_train_set = train_set.batch(BATCH_SIZE, drop_remainder=True)
test_set = DataLoader(TEST_DIR, type='test').create_dataset()
batched_test_set = test_set.batch(BATCH_SIZE, drop_remainder=True)

if IMAGE_AUG:
    aug = tf.keras.Sequential([tf.keras.layers.experimental.preprocessing.RandomFlip(),
                               tf.keras.layers.experimental.preprocessing.RandomRotation(0.05)])

model = SimpleCNN(num_class=6, pretrained='dense_net')
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

loss_function = tf.keras.losses.categorical_crossentropy

optimizer = tf.keras.optimizers.Adam()


@tf.function
def train_step(image, target):
    if IMAGE_AUG:
        image = aug(image)

    with tf.GradientTape() as tape:
        prediction = model(image)
        loss = loss_function(target, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(target, prediction)


val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')


def evaluate(image, target):
    prediction = model(image, train=False)
    loss = loss_function(target, prediction)
    val_loss(loss)
    val_accuracy(target, prediction)


with tf.device('/cpu:0'):
    for epoch in range(EPOCHS):
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()
        val_loss.reset_states()
        val_accuracy.reset_states()

        for (batch, (image, label)) in enumerate(batched_train_set):
            train_step(image, label)
            if batch % 50 == 0:
                print('Epoch {} Batch {} Train Loss {:.4f} Train Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))
        print('Epoch {} Train Loss {:.4f} Train Accuracy {:.4f}'.format(epoch + 1,
                                                                        train_loss.result(),
                                                                        train_accuracy.result()))

        for (batch, (image, label)) in enumerate(batched_test_set):
            evaluate(image, label)
        print('Epoch {} Val Loss {:.4f} Val Accuracy {:.4f}'.format(epoch + 1,
                                                                    val_loss.result(),
                                                                    val_accuracy.result()))

        print('Time taken for epoch: {:.4f} secs\n'.format(time.time() - start))
