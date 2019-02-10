import tensorflow as tf
import numpy as np
import math


def model(data, training=False):
    conv = tf.nn.conv2d(data, conv1_w_s2, strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC")
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_b))

    conv = tf.nn.separable_conv2d(relu, conv2_dw_s1, conv2_pw_s1, strides=[1, 1, 1, 1], padding="SAME")
    wb = tf.nn.bias_add(conv, conv2_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv3_dw_s2, conv3_pw_s1, strides=[1, 2, 2, 1], padding="SAME")
    wb = tf.nn.bias_add(conv, conv3_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv4_dw_s1, conv4_pw_s1, strides=[1, 1, 1, 1], padding="SAME")
    wb = tf.nn.bias_add(conv, conv4_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv5_dw_s2, conv5_pw_s1, strides=[1, 2, 2, 1], padding="SAME")
    wb = tf.nn.bias_add(conv, conv5_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv6_dw_s1, conv6_pw_s1, strides=[1, 1, 1, 1], padding="SAME")
    wb = tf.nn.bias_add(conv, conv6_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv7_dw_s2, conv7_pw_s1, strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv7_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv8_dw_s1, conv8_pw_s1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv8_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))
    conv = tf.nn.separable_conv2d(relu, conv9_dw_s1, conv9_pw_s1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv9_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))
    conv = tf.nn.separable_conv2d(relu, conv10_dw_s1, conv10_pw_s1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv10_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))
    conv = tf.nn.separable_conv2d(relu, conv11_dw_s1, conv11_pw_s1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv11_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))
    conv = tf.nn.separable_conv2d(relu, conv12_dw_s1, conv12_pw_s1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv12_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv13_dw_s2, conv13_pw_s1, strides=[1, 2, 2, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv13_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    conv = tf.nn.separable_conv2d(relu, conv14_dw_s2, conv14_pw_s1, strides=[1, 1, 1, 1], padding="SAME", data_format="NHWC")
    wb = tf.nn.bias_add(conv, conv14_b)
    relu = tf.nn.relu(tf.layers.batch_normalization(wb, training=training))

    avg_pool = tf.nn.avg_pool(relu, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding="VALID")
    reshape = tf.reshape(avg_pool, [-1, 1024])

    #pool = tf.nn.dropout(reshape, keep_prob)

    logits = tf.nn.relu(tf.matmul(reshape, FC_w))
    y_pred = tf.nn.softmax(logits)

    return y_pred, logits


def parse_single_example(record):
    features = {'label': tf.FixedLenFeature((), tf.int64),
                'image': tf.FixedLenFeature((), tf.string, '')}
    parsed_features = tf.parse_single_example(serialized=record, features=features)
    image = tf.decode_raw(parsed_features.get('image'), out_type=tf.uint8)
    image = tf.cast(tf.reshape(tensor=image, shape=[224, 224, 3]), dtype=tf.float32)
    image = tf.image.per_image_standardization(image)

    label = tf.cast(parsed_features.get('label'), dtype=tf.int64)
    # tf.print(label, [label])
    return image, label


def next_batch(index, num, data, labels):
    """
    :param num : 가져올 batch 수
    :param data: 이미지 data
    :param labels: 이미지 label
    :return: np.asarray 로 반환.
    """
    idx = np.arange(0, len(data)) # 데이터 크기 만큼 가져 온다.

    begin_index = index * num
    end_index = ((index+1) * num)
    idx = idx[begin_index:end_index]

    batch_data = [data[i] for i in idx]
    batch_labels = [labels[i] for i in idx]
    return np.array(batch_data), np.array(batch_labels)


def test_pick(num, data, labels):
    idx = np.arange(0, len(data))
    idx = idx[:num]
    batch_data = [data[i] for i in idx]
    batch_labels = [labels[i] for i in idx]

    return np.array(batch_data), np.array(batch_labels)


def data_shuffle(data, labels):
    """
    넘겨온 data를 shuffle 합니다.
    :param data: img_data
    :param labels: img_label
    :return: shuffled img, labels
    """
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    shuffled_data = [data[i] for i in idx]
    shuffled_labels = [labels[i] for i in idx]

    return shuffled_data, shuffled_labels

data_list = []
label = []

SEED = 66478
input_size = 224
input_channel = 3
class_num = 3

conv1_w_s2 = tf.Variable(tf.truncated_normal([3, 3, input_channel, 32], stddev=0.1, seed=SEED, dtype=tf.float32))
conv1_b = tf.Variable(tf.constant(0.1, shape=[32]))

conv2_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 32, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv2_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32))
conv2_b = tf.Variable(tf.constant(0.1, shape=[64]))


conv3_dw_s2 = tf.Variable(tf.truncated_normal([3, 3, 64, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv3_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 64, 128], stddev=0.1, seed=SEED, dtype=tf.float32))
conv3_b = tf.Variable(tf.constant(0.1, shape=[128]))


conv4_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 128, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv4_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 128, 128], stddev=0.1, seed=SEED, dtype=tf.float32))
conv4_b = tf.Variable(tf.constant(0.1, shape=[128]))


conv5_dw_s2 = tf.Variable(tf.truncated_normal([3, 3, 128, 1], stddev=0.1, dtype=tf.float32))
conv5_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 128, 256], stddev=0.1, dtype=tf.float32))
conv5_b = tf.Variable(tf.constant(0.1, shape=[256]))


conv6_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 256, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv6_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 256, 256], stddev=0.1, seed=SEED, dtype=tf.float32))
conv6_b = tf.Variable(tf.constant(0.1, shape=[256]))


conv7_dw_s2 = tf.Variable(tf.truncated_normal([3, 3, 256, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv7_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 256, 512], stddev=0.1, seed=SEED, dtype=tf.float32))
conv7_b = tf.Variable(tf.constant(0.1, shape=[512]))


conv8_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 512, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv8_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.1, seed=SEED, dtype=tf.float32))
conv8_b = tf.Variable(tf.constant(0.1, shape=[512]))

conv9_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 512, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv9_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.1, seed=SEED, dtype=tf.float32))
conv9_b = tf.Variable(tf.constant(0.1, shape=[512]))

conv10_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 512, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv10_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.1, seed=SEED, dtype=tf.float32))
conv10_b =tf.Variable(tf.constant(0.1, shape=[512]))

conv11_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 512, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv11_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.1, seed=SEED, dtype=tf.float32))
conv11_b = tf.Variable(tf.constant(0.1, shape=[512]))

conv12_dw_s1 = tf.Variable(tf.truncated_normal([3, 3, 512, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv12_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 512, 512], stddev=0.1, seed=SEED, dtype=tf.float32))
conv12_b = tf.Variable(tf.constant(0.1, shape=[512]))


conv13_dw_s2 = tf.Variable(tf.truncated_normal([3, 3, 512, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv13_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 512, 1024], stddev=0.1, seed=SEED, dtype=tf.float32))
conv13_b = tf.Variable(tf.constant(0.1, shape=[1024]))


conv14_dw_s2 = tf.Variable(tf.truncated_normal([3, 3, 1024, 1], stddev=0.1, seed=SEED, dtype=tf.float32))
conv14_pw_s1 = tf.Variable(tf.truncated_normal([1, 1, 1024, 1024], stddev=0.1, seed=SEED, dtype=tf.float32))
conv14_b = tf.Variable(tf.constant(0.1, shape=[1024]))

FC_w = tf.Variable(tf.truncated_normal([1024, class_num], stddev=0.1, dtype=tf.float32))




train = tf.data.TFRecordDataset(filenames='./mobilenet_train.tfrecords', compression_type='GZIP')
train = train.map(lambda record: parse_single_example(record))
train = train.shuffle(buffer_size=1000 + 3 * 128)
train = train.repeat()
train = train.batch(batch_size=128)


test = tf.data.TFRecordDataset(filenames='./mobilenet_test.tfrecords', compression_type='GZIP')
test = test.map(lambda record: parse_single_example(record))
test = test.repeat()
test = test.shuffle(buffer_size=1000 + 3 * 128)
test = test.batch(batch_size=128)

train_iterator = train.make_initializable_iterator()
test_iterator = test.make_initializable_iterator()

get_next_train = train_iterator.get_next()
get_next_test = test_iterator.get_next()




# with tf.Session() as sess:
    # sess.run([train_iterator.initializer, test_iterator.initializer])
    # # X_data = sess.run(X)
    # # y_data = sess.run(y) 이렇게 따로 sess를 하게 되면 iterator가 각 자 돌아 총 데이터 개수인 75개에서 x_data : 50, y_label: 25개로 되는
    # # 오류가 생기게 된다.
    # x_train_data, y_train_labels = sess.run(get_next_train)
    # # print(len(y_train_labels))
    # x_test_data, y_test_labels = sess.run(get_next_test)


# print(">>>>>", x_train_data.shape)

# print(x_data.shape, x_data.dtype) # (50, 224, 224, 3) float32
# print(y_data.shape, y_data.dtype) # (50, ) int64

x = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
y = tf.placeholder(tf.int64, shape=[None, class_num])
keep_prob = tf.placeholder(tf.float32)

y_pred, logits = model(x, training=False)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)) # [ True, False , .... 반환]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # 정확도 계산


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run([train_iterator.initializer, test_iterator.initializer])


    # print(">>>>", y_train_one_hot)

    # Epoch
    # dataset_size = len(x_train_data)
    # batch_size = 100

    for i in range(10000):
        x_train_data, y_train_labels = sess.run(get_next_train)
        x_test_data, y_test_labels = sess.run(get_next_test)

        y_train_one_hot = tf.one_hot(y_train_labels, class_num)
        y_test_one_hot = tf.one_hot(y_test_labels, class_num)

        y_train_one_hot = y_train_one_hot.eval()
        y_test_one_hot = y_test_one_hot.eval()

        #print(">>>>>>>>", y_train_one_hot)
        # Shuffle
        #x_train_data, y_train_one_hot = data_shuffle(x_train_data, y_train_one_hot)
        # Batch (min-batch size : 10)
        # total_count = dataset_size // batch_size
        # total_count = (total_count+1 if total_count > math.floor(total_count) else total_count)
        #
        # for batch_index in range(total_count):
        #     batch = next_batch(batch_index, batch_size, x_train_data, y_train_one_hot)
        #     sess.run(train_step, feed_dict={x: batch[0], y: batch[1], keep_prob: 0.8})
        if i % 10 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: x_train_data, y: y_train_one_hot})#, keep_prob: 1.0})

            y_p = y_pred.eval(feed_dict={x: x_train_data, y: y_train_one_hot})#, keep_prob: 1.0})
            l = loss.eval(feed_dict={x: x_train_data, y: y_train_one_hot})#, keep_prob: 1.0})
            print("Epoch: %d, 트레이닝 정확도 %f, Loss: %f"%(i, train_accuracy, l))
        sess.run(train_step, feed_dict={x: x_train_data, y: y_train_one_hot})#, keep_prob: 0.8})
    test_accuracy = 0.0
    for i in range(10):
        x_test_data, y_test_one_hot = data_shuffle(x_test_data, y_test_one_hot)
        test_batch = test_pick(10, x_test_data, y_test_one_hot)
        test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
    test_accuracy = test_accuracy / 10
    print("테스트 데이터 정확도: %f" %test_accuracy)