import os
import json
import shutil

label_dict = {'buildings': 0,
              'forest': 1,
              'glacier': 2,
              'mountain': 3,
              'sea': 4,
              'street': 5}

data_dir = '../data/'

# create label files
for train_test in ['seg_train', 'seg_test']:
    img_dir = os.path.join(data_dir, train_test)
    label_list = []
    pic_folder = os.path.join(img_dir, 'pic')
    if not os.path.exists(pic_folder):
        os.mkdir(pic_folder)
    for label_dir in label_dict.keys():
        label_folder = os.path.join(img_dir, 'raw/'+label_dir)
        image_names = os.listdir(label_folder)
        label = label_dict[label_dir]
        for image_name in image_names:
            img_label = {'image': image_name, 'label': label}
            label_list.append(img_label)
            shutil.copyfile(os.path.join(label_folder, image_name),
                            os.path.join(pic_folder, image_name))
    json.dump(label_list, open(os.path.join(img_dir, train_test+'_label.json'), 'w', encoding='utf-8'))


