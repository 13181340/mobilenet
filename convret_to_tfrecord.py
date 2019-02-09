import tensorflow as tf
import numpy as np
import os
import cv2

dir_list = '/Users/younghun-kim/working/TensorFlow_for_Deep_Learning/mobileNet/training'
#dir_list = '/home/ironman/working/dl_projects/dataset/ilsvrc2012/dataset'
#dir_list_test = '/Users/younghun-kim/working/TensorFlow_for_Deep_Learning/mobileNet/test'


def get_label_from_path(path):
    return int(path.split('/')[-2])


def read_image(path):
    origin_image = cv2.imread(path)
    reshape_img = cv2.resize(origin_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    return reshape_img


def search(dirname):
    global folder_cnt
    folder_cnt += 1
    try:
        filenames = os.listdir(dirname)
        for filename in filenames:
            full_filename = os.path.join(dirname, filename)
            if os.path.isdir(full_filename):
                #print(">>>",full_filename)
                search(full_filename)
            else:
                if full_filename.replace("JPEG", "jpg").endswith(".jpg"):
                    data_list.append(full_filename)
                    print(full_filename)
                    label.append([str(folder_cnt-1)])

        return np.asarray(data_list, dtype=np.string_), np.asarray(label, dtype=np.int64)
    except PermissionError:
        pass



data_list = []
label = []

SEED = 66478
input_size = 224
input_channel = 3
folder_cnt = 0

(x_data, y_data) = search(dir_list)
print(x_data.shape, y_data.shape)
x_data_shape = x_data.shape[0]

#Shuffle
idx = np.arange(0, x_data_shape)
np.random.shuffle(idx)
x_data = np.array([x_data[i] for i in idx])
y_data = np.array([y_data[i] for i in idx])


test_data_len = int(0.2* x_data_shape) # test data  = 20%
x_test = x_data[0:test_data_len]
y_test = y_data[0:test_data_len]
print(x_test.shape, y_test.shape)

x_train = np.delete(x_data, obj=np.arange(test_data_len), axis=0) # train data = 80%
y_train = np.delete(y_data, obj=np.arange(test_data_len), axis=0)
print(x_train.shape, y_train.shape)
"""
위 data 파일을 받아올 때 위 처럼 분류 할 수 있도록 설계할 것..."""

#print(">>>>>>", x_train.shape, y_train.shape)
#(x_test, y_test) = search(dir_list_test) # 이미지 사이즈 변환까지 완료.

train = zip(x_train, y_train)
test = zip(x_test, y_test)

split = dict(train=train, test=test)

options = tf.python_io.TFRecordOptions(compression_type=tf.python_io.TFRecordCompressionType.GZIP)

for key in split.keys():
    dataset = split.get(key)
    writer = tf.python_io.TFRecordWriter(path='./mobilenet_{}.tfrecords'.format(key), options=options)

    for file_path, label in dataset:
        bytes_to_string = file_path.decode('UTF-8')
        data = read_image(bytes_to_string)
        image = data.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])), # label은 int64 list로 저장
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))
        }))
        writer.write(example.SerializeToString())
    else:
        writer.close()
        print('{} was converted to TFRecords.\n'.format(key))

print("Finish convert to tfrecord")