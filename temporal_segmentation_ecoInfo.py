import random
import sys
import math
from os import listdir
import os
import warnings

import numpy as np
import tensorflow as tf
import scipy.misc
from skimage import img_as_float
from sklearn.metrics import f1_score

# from tensorflow.python.framework import ops

NUM_CLASSES = 2


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_params(list_params):
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for i in xrange(1, len(sys.argv)):
        print list_params[i - 1] + '= ' + sys.argv[i]
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


def memory_usage_psutil():
    # return the memory usage in percentage like top
    process = psutil.Process(os.getpid())
    mem = process.memory_percent()
    return mem


def select_batch(shuffle, batch_size, it, total_size):
    batch = shuffle[it:min(it + batch_size, total_size)]
    if min(it + batch_size, total_size) == total_size or total_size == it + batch_size:
        shuffle = np.asarray(random.sample(xrange(total_size), total_size))
        # print "in", shuffle
        it = 0
        if len(batch) < batch_size:
            diff = batch_size - len(batch)
            batch_c = shuffle[it:it + diff]
            batch = np.concatenate((batch, batch_c))
            it = diff
            # print 'c', batch_c, batch, it
    else:
        it += batch_size
    return shuffle, batch, it


def normalize_images(patches, mean_full, std_full):
    # warnings.filterwarnings('error')

    for i in xrange(len(patches)):
        for j in xrange(39):
            patches[i, :, :, :, j] = np.subtract(patches[i, :, :, :, j], mean_full[i, j])
            patches[i, :, :, :, j] = np.divide(patches[i, :, :, :, j], (std_full[i, j] if std_full[i, j] != 0.0 else 1.0))


def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0, :]

    return mean_full, std_full


def calculate_mean_and_std(data, indexes, crop_size):
    mean_full = [[[] for i in range(0)] for i in xrange(len(data))]
    std_full = [[[] for i in range(0)] for i in xrange(len(data))]
    mask = int(crop_size / 2)

    for img_num in xrange(len(data)):
        all_patches = []
        for i in xrange(len(indexes)):
            cur_x = indexes[i][0]
            cur_y = indexes[i][1]

            patches = data[img_num, (cur_x + mask) - mask:(cur_x + mask) + mask + 1,
                           (cur_y + mask) - mask:(cur_y + mask) + mask + 1, :]
            if len(patches) != crop_size or len(patches[0]) != crop_size:
                print BatchColors.FAIL + "Error! Current patch size: " + str(len(patches)) + "x" + \
                      str(len(patches[0])) + BatchColors.ENDC
                return

            all_patches.append(patches)

        mean, std = compute_image_mean(np.asarray(all_patches))
        # print mean.shape, std.shape
        mean_full[img_num].append(mean)
        std_full[img_num].append(std)

    # print mean_full, std_full
    return np.squeeze(np.asarray(mean_full)), np.squeeze(np.asarray(std_full))


def manipulate_border_array(data, crop_size):
    mask = int(crop_size / 2)
    # print data.shape

    h, w = len(data), len(data[0])
    # print h, w
    crop_left = data[0:h, 0:crop_size, :]
    crop_right = data[0:h, w - crop_size:w, :]
    crop_top = data[0:crop_size, 0:w, :]
    crop_bottom = data[h - crop_size:h, 0:w, :]
    # print crop_left.shape, crop_right.shape, crop_top.shape, crop_bottom.shape

    mirror_left = np.fliplr(crop_left)
    mirror_right = np.fliplr(crop_right)
    flipped_top = np.flipud(crop_top)
    flipped_bottom = np.flipud(crop_bottom)
    # print mirror_left.shape, mirror_right.shape, flipped_top.shape, flipped_bottom.shape

    h_new, w_new = h + mask * 2, w + mask * 2
    data_border = np.zeros((h_new, w_new, len(data[0][0])))
    # print data_border.shape
    data_border[mask:h + mask, mask:w + mask, :] = data
    # print h_new, w_new, data_border.shape

    data_border[mask:h + mask, 0:mask, :] = mirror_left[:, mask + 1:, :]
    data_border[mask:h + mask, w_new - mask:w_new, :] = mirror_right[:, 0:mask, :]
    data_border[0:mask, mask:w + mask, :] = flipped_top[mask + 1:, :, :]
    data_border[h + mask:h + mask + mask, mask:w + mask, :] = flipped_bottom[0:mask, :, :]

    data_border[0:mask, 0:mask, :] = flipped_top[mask + 1:, 0:mask, :]
    data_border[0:mask, w + mask:w + mask + mask, :] = flipped_top[mask + 1:, w - mask:w, :]
    data_border[h + mask:h + mask + mask, 0:mask, :] = flipped_bottom[0:mask, 0:mask, :]
    data_border[h + mask:h + mask + mask, w + mask:w + mask + mask, :] = flipped_bottom[0:mask, w - mask:w, :]

    # scipy.misc.imsave('C:\\Users\\Keiller\\Desktop\\outfile.jpg', data_border)
    return data_border


def load_images(path, specie_type, operation, crop_size):
    mask = int(crop_size / 2)

    data = []
    day = np.zeros((960+(mask*2), 1280+(mask*2), 3 * 13), dtype=np.float)

    old_img_num = 242

    for f in sorted(listdir(path + "Imagens")):
        try:
            img = img_as_float(scipy.misc.imread(path + "Imagens/" + f))
            # print f, img.shape
        except IOError:
            print BatchColors.FAIL + "Could not open file: ", path + "Images/" + f + BatchColors.ENDC
            return

        # print(memory_usage_psutil())
        year, img_num, hour, stamp = f[:-4].split('_')
        if old_img_num != int(img_num):
            # print 'new day', f
            data.append(day)
            # print 'data', np.asarray(data).shape
            day = np.zeros((960 + mask * 2, 1280 + mask * 2, 3 * 13), dtype=np.float)
            old_img_num = int(img_num)

        if int(stamp) == 1:
            day[:, :, (int(hour) - 6) * 3:((int(hour) - 6) * 3) + 3] = manipulate_border_array(img, crop_size)
        '''else:
        hour_5img = np.concatenate((hour_5img, img), axis=2)
        if int(stamp) == 5:
            # print 'new img', hour_5img.shape
            day.append(hour_5img)
            hour_5img = []'''

    # add the last day into data
    data.append(day)

    try:
        if operation == 'generate_pred_images':
            img = scipy.misc.imread(path + "Mask/whole_mask_int.png")
        else:
            img = scipy.misc.imread(path + "Mask/" + specie_type + "_mask_int_balanced.png")  # _balanced2
    except IOError:
        print BatchColors.FAIL + "Could not open file: " + path + "Mask/" + specie_type + "_mask_int.png" + BatchColors.ENDC
        return

    mask = img

    return np.asarray(data), np.asarray(mask)


def create_distributions_over_pixel_classes(labels):
    train_class_distribution = [[[] for i in range(0)] for i in range(NUM_CLASSES)]
    test_class_distribution = [[[] for i in range(0)] for i in range(NUM_CLASSES)]

    w, h = labels.shape

    for i in xrange(0, w):
        for j in xrange(0, h):
            if labels[i, j] == 1:
                train_class_distribution[labels[i, j]-1].append((i, j))
            elif labels[i, j] == 2:
                train_class_distribution[labels[i, j]-1].append((i, j))
            elif labels[i, j] == 3:
                test_class_distribution[labels[i, j]-2].append((i, j))

    for i in xrange(len(train_class_distribution)):
        print BatchColors.OKBLUE + "Class " + str(i) + " = " + str(len(train_class_distribution[i])) + BatchColors.ENDC
    for i in xrange(len(test_class_distribution)):
        print BatchColors.OKBLUE + "Class " + str(i) + " = " + str(len(test_class_distribution[i])) + BatchColors.ENDC

    return np.asarray(train_class_distribution[0] + train_class_distribution[1]), \
           np.asarray(test_class_distribution[0] + test_class_distribution[1])


def dynamically_create_patches(data, mask_data, crop_size, class_distribution, shuffle):
    mask = int(crop_size / 2)

    patches = []
    classes = []

    for i in shuffle:
        if i >= 2 * len(class_distribution):
            cur_pos = i - 2 * len(class_distribution)
        elif i >= len(class_distribution):
            cur_pos = i - len(class_distribution)
        else:
            cur_pos = i

        cur_x = class_distribution[cur_pos][0]
        cur_y = class_distribution[cur_pos][1]

        patch = data[:, (cur_x + mask) - mask:(cur_x + mask) + mask + 1,
                     (cur_y + mask) - mask:(cur_y + mask) + mask + 1, :]

        current_class = (mask_data[cur_x, cur_y]-1 if (mask_data[cur_x, cur_y] == 1 or mask_data[cur_x, cur_y] == 2)
                         else mask_data[cur_x, cur_y]-2)

        if len(patch[0]) != crop_size or len(patch[1]) != crop_size:
            print "Error: Current patch size ", len(patch), len(patch[0])
            return
        if current_class != 0 and current_class != 1 and current_class != 2 and current_class != 3 and current_class != 4:
            print "Error: Current class is mistaken", current_class
            return

        if i < len(class_distribution):
            patches.append(patch)
        elif i < 2 * len(class_distribution):
            patches.append(np.fliplr(patch))
        elif i >= 2 * len(class_distribution):
            patches.append(np.flipud(patch))

        classes.append(current_class)

    return np.swapaxes(np.asarray(patches), 0, 1), np.asarray(classes, dtype=np.int32)


'''
TensorFlow
'''


def leaky_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def _variable_on_cpu(name, shape, ini):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=ini, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, ini, weight_decay):
    var = _variable_on_cpu(name, shape, ini)
    # tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    # tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    # tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    # orthogonal_initializer()
    if weight_decay is not None:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _batch_norm(input_data, is_training, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=True, center=False, updates_collections=None,
                                                        scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=False, center=False,
                                                        updates_collections=None, scope=scope, reuse=True)
                   )


def _conv_layer(input_data, layer_shape, name, weight_decay, is_training, rate=1, strides=None, pad='SAME',
                activation='relu', has_batch_norm=True, has_activation=True, is_normal_conv=False):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=layer_shape,
                                              ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', layer_shape[-1], tf.constant_initializer(0.1))

        if is_normal_conv is False:
            conv_op = tf.nn.atrous_conv2d(input_data, weights, rate=rate, padding=pad)
        else:
            conv_op = tf.nn.conv2d(input_data, weights, strides=strides, padding=pad)
        conv_act = tf.nn.bias_add(conv_op, biases)

        if has_batch_norm == True:
            conv_act = _batch_norm(conv_act, is_training, scope=scope)
        if has_activation == True:
            if activation == 'relu':
                conv_act = tf.nn.relu(conv_act, name=name)
            else:
                conv_act = leaky_relu(conv_act)

        return conv_act


def _max_pool(input_data, kernel, strides, name, pad='SAME', debug=False):
    pool = tf.nn.max_pool(input_data, ksize=kernel, strides=strides, padding=pad, name=name)
    if debug:
        pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

    return pool


def convnet_initial(x, dropout, is_training, weight_decay, crop_size, name_prefix):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 13*3])  # default: 25x25
    # print x.get_shape()

    conv1 = _conv_layer(x, [4, 4, 13*3, 64], name_prefix + '_conv1', weight_decay,
                        is_training, pad='VALID', is_normal_conv=True, activation='lrelu')
    pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name=name_prefix + '_pool1', pad='VALID')

    return pool1


def convnet_25_temporal(x, dropout, is_training, crop_size, weight_decay, inputs):
    pools = []

    for i in xrange(inputs):
        pools.append(convnet_initial(x[i], dropout, is_training, weight_decay, crop_size, 'time_' + str(i)))

    # conv1 = _conv_layer(x, [4, 4, 3, 64], 'ft_conv1', weight_decay, is_training, pad='VALID')
    # pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')

    try:
        pool_concat = tf.concat(pools, 3)
    except:
        pool_concat = tf.concat(concat_dim=3, values=pools)

    # drop_connection = tf.nn.dropout(pool_concat, dropout_connection)
    conv2 = _conv_layer(pool_concat, [4, 4, 64*inputs, 128], 'ft_conv2', weight_decay,
                        is_training, pad='VALID', is_normal_conv=True, activation='lrelu')
    pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool2', pad='VALID')

    conv3 = _conv_layer(pool2, [3, 3, 128, 256], 'ft_conv3', weight_decay,
                        is_training, pad='VALID', is_normal_conv=True, activation='lrelu')
    pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='ft_pool3', pad='VALID')

    with tf.variable_scope('ft_fc1') as scope:
        reshape = tf.reshape(pool3, [-1, 1 * 1 * 256])
        weights = _variable_with_weight_decay('weights', shape=[1 * 1 * 256, 1024],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
        drop_fc1 = tf.nn.dropout(reshape, dropout)
        fc1 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc1, weights), biases), is_training, scope=scope.name))

    # Fully connected layer 2
    with tf.variable_scope('ft_fc2') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1024, 1024],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))

        # Apply Dropout
        drop_fc2 = tf.nn.dropout(fc1, dropout)
        fc2 = tf.nn.relu(_batch_norm(tf.add(tf.matmul(drop_fc2, weights), biases), is_training, scope=scope.name))

    with tf.variable_scope('fc3_logits') as scope:
        weights = _variable_with_weight_decay('weights', [1024, NUM_CLASSES],
                                              ini=tf.contrib.layers.xavier_initializer(dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.1))
        logits = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return logits


def loss_def(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def validate(sess, data, labels, test_distribution, crop_size,
             mean_full, std_full, n_input_data, batch_size, x, y, keep_prob,  # keep_prob_connection,
             is_training, pred, acc_mean, step):
    all_predcs = []
    cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    true_count = 0.0

    total = len(test_distribution)

    index = np.arange(total)

    for i in range(0, int(math.ceil((total / float(batch_size))))):
        batch = index[i * batch_size:min(i * batch_size + batch_size, total)]
        bx, by = dynamically_create_patches(data, labels, crop_size, test_distribution, batch)
        normalize_images(bx, mean_full, std_full)
        bx = np.reshape(bx, (data.shape[0], -1, n_input_data))

        preds_val, acc_mean_val = sess.run([pred, acc_mean], feed_dict={x: bx, y: by, keep_prob: 1.,
                                                                        # keep_prob_connection: 1.,
                                                                        is_training: False})
        true_count += acc_mean_val

        all_predcs = np.concatenate((all_predcs, preds_val))

        for j in range(len(preds_val)):
            cm_test[by[j]][preds_val[j]] += 1

    _sum = 0.0
    for i in xrange(len(cm_test)):
        _sum += (cm_test[i][i] / float(np.sum(cm_test[i])) if np.sum(cm_test[i]) != 0 else 0)

    print("---- Iter " + str(step) + " -- Validate: Overall Accuracy= " + str(int(true_count)) +
          " Overall Accuracy= " + "{:.6f}".format(true_count / float(np.sum(np.sum(cm_test)))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          # " Kappa= " + "{:.4f}".format(cohen_kappa_score(classes, np.asarray(all_predcs))) +
          " Confusion Matrix= " + np.array_str(cm_test).replace("\n", "")
          )


def train(data, labels, train_class_distribution, test_class_distribution,
          mean_full, std_full,
          crop_size, batch_size, niter, model_path,
          x, y, keep_prob, dropout, is_training, n_input_data,
          optimizer, loss, acc_mean, pred, output_path):
    ###################
    display_step = 50
    epoch_number = 100  # int(len(training_classes)/batch_size) # 1 epoch = images / batch
    val_inteval = 100  # int(len(training_classes)/batch_size)
    # print '1 epoch every %s iterations' % str(epoch_number)
    # print '1 validation every %s iterations' % str(val_inteval)
    # display_step = math.ceil(int(len(training_classes)/batch_size)*0.01)
    ###################

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()
    saver_restore = tf.train.Saver()
    current_iter = 1

    # Initializing the variables
    init = tf.initialize_all_variables()
    shuffle = np.asarray(random.sample(xrange(3 * len(train_class_distribution)), 3 * len(train_class_distribution)))

    # Launch the graph
    with tf.Session() as sess:
        if 'model' in model_path:
            current_iter = int(model_path.split('-')[-1])
            print BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC
            saver_restore.restore(sess, model_path)
        else:
            sess.run(init)
            print BatchColors.OKBLUE + 'Model totally initialized!' + BatchColors.ENDC

        # aux variables
        it = 0
        epoch_mean = 0.0
        epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
        batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

        # Keep training until reach max iterations
        for step in xrange(current_iter, niter + 1):
            shuffle, batch, it = select_batch(shuffle, batch_size, it, 3 * len(train_class_distribution))

            b_x, batch_y = dynamically_create_patches(data, labels, crop_size, train_class_distribution, batch)
            normalize_images(b_x, mean_full, std_full)
            batch_x = np.reshape(b_x, (data.shape[0], -1, n_input_data))

            # Run optimization op (backprop)
            _, batch_loss, batch_correct, batch_predcs = sess.run([optimizer, loss, acc_mean, pred],
                                                                  feed_dict={x: batch_x, y: batch_y,
                                                                             keep_prob: dropout,
                                                                             # keep_prob_connection: dropout_connection,
                                                                             is_training: True})

            epoch_mean += batch_correct
            for j in range(len(batch_predcs)):
                epoch_cm_train[batch_y[j]][batch_predcs[j]] += 1

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                for j in range(len(batch_predcs)):
                    batch_cm_train[batch_y[j]][batch_predcs[j]] += 1

                _sum = 0.0
                for i in xrange(len(batch_cm_train)):
                    _sum += (batch_cm_train[i][i] / float(np.sum(batch_cm_train[i])) if np.sum(
                        batch_cm_train[i]) != 0 else 0)

                print("Iter " + str(step) + " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                      " Absolut Right Pred= " + str(int(batch_correct)) +
                      " Overall Accuracy= " + "{:.4f}".format(batch_correct / float(np.sum(np.sum(batch_cm_train)))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                      )
                batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            if step % epoch_number == 0:
                _sum = 0.0
                for i in xrange(len(epoch_cm_train)):
                    _sum += (epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i])) if np.sum(
                        epoch_cm_train[i]) != 0 else 0)

                print("-- Iter " + str(step) + " -- Training Epoch:" +
                      " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(np.sum(np.sum(epoch_cm_train)))) +
                      " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                      )

                epoch_mean = 0.0
                epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            if step % val_inteval == 0:
                # Test
                saver.save(sess, output_path + 'model_' + str(step))
                validate(sess, data, labels, test_class_distribution, crop_size,
                         mean_full, std_full, n_input_data, batch_size, x, y, keep_prob,  # keep_prob_connection,
                         is_training, pred, acc_mean, step)

        print BatchColors.OKGREEN + "Optimization Finished!" + BatchColors.ENDC

        # Test: Final
        saver.save(sess, output_path + 'model', global_step=step)
        validate(sess, data, labels, test_class_distribution, crop_size,
                 mean_full, std_full, n_input_data, batch_size, x, y, keep_prob,  # keep_prob_connection,
                 is_training, pred, acc_mean, step)


def test(data, labels, test_class_distribution, mean_full, std_full,
         crop_size, batch_size, model_path, x, y, keep_prob,  # keep_prob_connection,
         is_training, n_input_data, acc_mean, pred):

    # Add ops to save and restore all the variables.
    saver_restore = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        print BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC
        saver_restore.restore(sess, model_path)

        validate(sess, data, labels, test_class_distribution, crop_size,
                 mean_full, std_full, n_input_data, batch_size, x, y, keep_prob,
                 is_training, pred, acc_mean, 200000)


def color_mask(prob):
    if prob[0] >= 0.5:
        return (255 * prob[0], 0, 0)
    else:
        return (0, 255 * prob[1], 0)


def generate_pred_images(data, labels, test_class_distribution, mean_full, std_full,
                         crop_size, batch_size, model_path, x, y, keep_prob,
                         is_training, n_input_data, logits, output_path, specie_type):

    w, h = labels.shape
    mask = np.zeros((w, h, 2), dtype=np.float)
    print mask.shape

    # Add ops to save and restore all the variables.
    saver_restore = tf.train.Saver()
    softmax = tf.nn.softmax(logits)

    # Launch the graph
    with tf.Session() as sess:
        print BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC
        saver_restore.restore(sess, model_path)

        total = len(test_class_distribution)
        index = np.arange(total)

        for i in range(0, int(math.ceil((total / float(batch_size))))):
            batch = index[i * batch_size:min(i * batch_size + batch_size, total)]
            bx, by = dynamically_create_patches(data, labels, crop_size, test_class_distribution, batch)
            normalize_images(bx, mean_full, std_full)
            bx = np.reshape(bx, (39, -1, n_input_data))

            _softmax = sess.run(softmax, feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})
            for j in xrange(len(_softmax)):
                # print _softmax[j]
                mask[int(test_class_distribution[j][0]), int(test_class_distribution[j][1]), :] = _softmax[j]
                # color_mask(_softmax[j])

    np.save(output_path + specie_type + '_whole_pred', mask)
    # scipy.misc.imsave(output_path + specie_type + '_whole_pred_mask.png', mask)


'''
python temporal_segmentation_ecoInfo.py /home/Biologist_Dataset/ecoInfo_dataset/ /home/Biologist_Dataset/ecoInfo_dataset/output/ /home/Biologist_Dataset/ecoInfo_dataset/output/ 0.01 0.005 100 200000 25 
rubiginosa training

'''


def main():
    list_params = ['input_path', 'output_path (for model, images, etc)', 'model_path',
                   'learning_rate', 'weight_decay', 'batch_size', 'niter', 'crop_size',
                   'specie_type [aspidosperma|rubiginosa]', 'operation [training|testing|generate_pred_images]']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    # images path
    index = 1
    input_path = sys.argv[index]
    # output_path
    index = index + 1
    output_path = sys.argv[index]
    index = index + 1
    model_path = sys.argv[index]

    # Parameters
    index = index + 1
    lr_initial = float(sys.argv[index])
    index = index + 1
    weight_decay = float(sys.argv[index])
    index = index + 1
    batch_size = int(sys.argv[index])
    index = index + 1
    niter = int(sys.argv[index])
    index = index + 1
    crop_size = int(sys.argv[index])

    index = index + 1
    specie_type = sys.argv[index]
    index = index + 1
    operation = sys.argv[index]

    print BatchColors.OKBLUE + 'Reading images...' + BatchColors.ENDC
    data, labels = load_images(input_path, specie_type, operation, crop_size)
    print data.shape, labels.shape

    print BatchColors.OKBLUE + 'Creating class distribution...' + BatchColors.ENDC
    train_class_distribution, test_class_distribution = create_distributions_over_pixel_classes(labels)

    if os.path.isfile(output_path + 'mean.npy'):
        mean_full = np.squeeze(np.load(output_path + 'mean.npy'))
        std_full = np.squeeze(np.load(output_path + 'std.npy'))
        print BatchColors.OKGREEN + 'Loaded Mean/Std from training instances' + BatchColors.ENDC
    else:
        mean_full, std_full = calculate_mean_and_std(data, train_class_distribution, crop_size)
        np.save(output_path + 'mean.npy', mean_full)
        np.save(output_path + 'std.npy', std_full)
        print BatchColors.OKGREEN + 'Created Mean/Std from training instances' + BatchColors.ENDC
    print mean_full.shape, std_full.shape

    # Network Parameters
    n_input_data = crop_size * crop_size * 13*3  # 13 timestamps * RGB
    dropout = 0.5  # Dropout, probability to keep units
    # dropout_connection = 1.0 # (1/13.0)

    # tf Graph input_data
    x = tf.placeholder(tf.float32, [data.shape[0], None, n_input_data], name='ph_data')
    y = tf.placeholder(tf.int32, [None], name='ph_labels')

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    # keep_prob_connection = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # CONVNET
    logits = convnet_25_temporal(x, keep_prob, is_training, crop_size, weight_decay, data.shape[0])

    # Define loss and optimizer
    loss = loss_def(logits, y)

    lr = tf.train.exponential_decay(lr_initial, global_step, 50000, 0.1, staircase=True)

    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

    # Evaluate model
    correct = tf.nn.in_top_k(logits, y, 1)
    acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
    pred = tf.argmax(logits, 1)

    if operation == 'training':
        train(data, labels, train_class_distribution, test_class_distribution,
              mean_full, std_full,
              crop_size, batch_size, niter, model_path,
              x, y, keep_prob, dropout, is_training, n_input_data,
              optimizer, loss, acc_mean, pred, output_path)
    elif operation == 'testing':
        test(data, labels, test_class_distribution, mean_full, std_full,
             crop_size, batch_size, model_path, x, y, keep_prob,  # keep_prob_connection,
             is_training, n_input_data, acc_mean, pred)
    elif operation == 'generate_pred_images':
        generate_pred_images(data, labels, test_class_distribution, mean_full, std_full,
                             crop_size, batch_size, model_path, x, y, keep_prob,
                             is_training, n_input_data, logits, output_path, specie_type)
    else:
        print BatchColors.FAIL + "Operation not found: " + operation + BatchColors.ENDC


if __name__ == "__main__":
    main()
