import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pathlib
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle


class DataLoader:
    def __init__(self, location, type='train', image_size=None, num_class=6):
        if image_size is None:
            image_size = [150, 150]
        self.image_size = image_size
        self.num_class = num_class
        self.root = pathlib.Path(location)
        self.pic_folder = self.root.joinpath('pic')
        if type == 'train':
            json_file = 'seg_train_label'
        else:
            json_file = 'seg_test_label'
        self.label_df = shuffle(pd.read_json(self.root.joinpath(json_file).with_suffix('.json')),
                                random_state=1)
        self.label_df['img_path'] = self.label_df['image'].map(lambda file: str(self.pic_folder.joinpath(str(file))))

    def load_and_preprocess_image(self, path):
        image_file = tf.io.read_file(path)
        image_file = tf.image.decode_jpeg(image_file, channels=3)
        image_file = tf.image.resize(image_file, self.image_size)
        return image_file

    def one_hot(self, img_label):
        vec = tf.one_hot(img_label, self.num_class)
        return vec

    def create_dataset(self):
        label_ds = tf.data.Dataset.from_tensor_slices(self.label_df['label'])
        one_hot_label_ds = label_ds.map(self.one_hot, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        img_path_ds = tf.data.Dataset.from_tensor_slices(self.label_df['img_path'])
        img_ds = img_path_ds.map(self.load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return tf.data.Dataset.zip((img_ds, one_hot_label_ds))


if __name__ == '__main__':
    train_loader = DataLoader('../data/seg_train')
    train_set = train_loader.create_dataset()
    batched_train_set = train_set.batch(128, drop_remainder=True)
    for image, label in batched_train_set.take(1):
        print(image.shape, label.shape)
