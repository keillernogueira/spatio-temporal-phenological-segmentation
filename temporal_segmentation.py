import random
import sys
import os
import math
import datetime
from PIL import Image
import numpy as np
import tensorflow as tf
import scipy.misc
from skimage import img_as_float
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedShuffleSplit

# from tensorflow.python.framework import ops

NUM_CLASSES = 4


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
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for i in range(1, len(sys.argv)):
        print(list_params[i - 1] + '= ' + sys.argv[i])
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')


def select_batch(shuffle, batch_size, it, total_size):
    batch = shuffle[it:min(it + batch_size, total_size)]
    if min(it + batch_size, total_size) == total_size or total_size == it + batch_size:
        shuffle = np.asarray(random.sample(range(total_size), total_size))
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


def manipulate_border_array(data, crop_size):
    mask = int(crop_size / 2)
    # print data.shape

    h, w = len(data), len(data[0])
    crop_left = data[0:h, 0:crop_size, :]
    crop_right = data[0:h, w - crop_size:w, :]
    crop_top = data[0:crop_size, 0:w, :]
    crop_bottom = data[h - crop_size:h, 0:w, :]

    mirror_left = np.fliplr(crop_left)
    mirror_right = np.fliplr(crop_right)
    flipped_top = np.flipud(crop_top)
    flipped_bottom = np.flipud(crop_bottom)

    h_new, w_new = h + mask * 2, w + mask * 2
    data_border = np.zeros((h_new, w_new, len(data[0][0])))
    # print data_border.shape
    data_border[mask:h + mask, mask:w + mask, :] = data

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


def normalize_images(data, mean_full, std_full):
    if mean_full.ndim >= 2:
        for i in range(len(data)):
            data[i, :, :, :, 0] = np.subtract(data[i, :, :, :, 0], mean_full[i, 0])
            data[i, :, :, :, 1] = np.subtract(data[i, :, :, :, 1], mean_full[i, 1])
            data[i, :, :, :, 2] = np.subtract(data[i, :, :, :, 2], mean_full[i, 2])

            data[i, :, :, :, 0] = np.divide(data[i, :, :, :, 0], std_full[i, 0])
            data[i, :, :, :, 1] = np.divide(data[i, :, :, :, 1], std_full[i, 1])
            data[i, :, :, :, 2] = np.divide(data[i, :, :, :, 2], std_full[i, 2])
    else:
        for i in range(len(data)):
            data[i, :, :, :, 0] = np.subtract(data[i, :, :, :, 0], mean_full[0])
            data[i, :, :, :, 1] = np.subtract(data[i, :, :, :, 1], mean_full[1])
            data[i, :, :, :, 2] = np.subtract(data[i, :, :, :, 2], mean_full[2])

            data[i, :, :, :, 0] = np.divide(data[i, :, :, :, 0], std_full[0])
            data[i, :, :, :, 1] = np.divide(data[i, :, :, :, 1], std_full[1])
            data[i, :, :, :, 2] = np.divide(data[i, :, :, :, 2], std_full[2])


def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0, :]

    return mean_full, std_full


def calculate_mean_and_std(data, indexes, crop_size):
    mean_full = [[[] for i in range(0)] for i in range(len(data))]
    std_full = [[[] for i in range(0)] for i in range(len(data))]
    mask = int(crop_size / 2)

    for cur_map in range(len(data)):
        all_patches = []
        for i in range(len(indexes)):
            cur_x = indexes[i][0]
            cur_y = indexes[i][1]

            patches = data[cur_map, (cur_x + mask) - mask:(cur_x + mask) + mask + 1,
                           (cur_y + mask) - mask:(cur_y + mask) + mask + 1, :]
            if len(patches) != crop_size or len(patches[1]) != crop_size:
                print(BatchColors.FAIL + "Error! Current patch size: " + str(len(patches)) + "x" + \
                      str(len(patches[0])) + BatchColors.ENDC)
                return

            all_patches.append(patches)

        mean, std = compute_image_mean(np.asarray(all_patches))
        mean_full[cur_map].append(mean)
        std_full[cur_map].append(std)

    # print mean_full, std_full
    return np.squeeze(np.asarray(mean_full)), np.squeeze(np.asarray(std_full))


def load_images(path,  crop_size, instances):
    data = []
    mask = []

    for name in instances:
        try:
            img = img_as_float(scipy.misc.imread(os.path.join(path, name)))
        except IOError:
            print(BatchColors.FAIL + "Could not open file: ", path + name + BatchColors.ENDC)

        data.append(manipulate_border_array(img, crop_size))

    try:
        img = scipy.misc.imread(os.path.join(path, "mask_train_test_int.png"))
    except IOError:
        print(BatchColors.FAIL + "Could not open file: ", path + "mask_train_test_int.png" + BatchColors.ENDC)

    mask = img

    return np.asarray(data), np.asarray(mask)


def create_distributions_over_pixel_classes(labels):
    training_instances = [[[] for i in range(0)] for i in range(NUM_CLASSES)]
    testing_instances = [[[] for i in range(0)] for i in range(NUM_CLASSES)]
    no_classes_instances = []

    w, h = labels.shape

    for i in range(0, w):
        for j in range(0, h):
            if labels[i, j] != 8:
                if labels[i, j] == 0 or labels[i, j] == 1 or labels[i, j] == 2 or labels[i, j] == 3:
                    training_instances[labels[i, j]].append((i, j))
                else:
                    testing_instances[labels[i, j]-4].append((i, j))
            else:
                no_classes_instances.append((i, j))

    for i in range(len(training_instances)):
        print(BatchColors.OKBLUE + "Training class " + str(i) + " = " + str(len(training_instances[i])) + BatchColors.ENDC)
        print(BatchColors.OKBLUE + "Testing class " + str(i) + " = " + str(len(testing_instances[i])) + BatchColors.ENDC)
    print(BatchColors.OKBLUE + 'No class = ' + str(len(no_classes_instances)) + BatchColors.ENDC)

    train_data = np.asarray(training_instances[0] + training_instances[1] +
                            training_instances[2] + training_instances[3])
    test_data = np.asarray(testing_instances[0] + testing_instances[1] +
                           testing_instances[2] + testing_instances[3])
    return train_data, test_data, no_classes_instances


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
        current_class = mask_data[cur_x, cur_y]
        if current_class > 3:
            current_class = current_class - 4

        if len(patch[0]) != crop_size or len(patch[0][0]) != crop_size:
            print("Error: Current patch size ", len(patch[0]), len(patch[0][0]))
            return
        if current_class != 0 and current_class != 1 and current_class != 2 and current_class != 3:
            print("Error: Current class is mistaken", current_class)
            return

        if i < len(class_distribution):
            patches.append(patch)
            classes.append(current_class)
        elif i < 2 * len(class_distribution):
            patches.append(np.fliplr(patch))
            classes.append(current_class)
        elif i >= 2 * len(class_distribution):
            patches.append(np.flipud(patch))
            classes.append(current_class)

    return np.swapaxes(np.asarray(patches), 0, 1), np.asarray(classes, dtype=np.int32)


def class_to_color(value):
    if value == 0:
        return (0, 255, 255)
    elif value == 1:
        return (0, 0, 255)
    elif value == 2:
        return (255, 0, 0)
    elif value == 3:
        return (0, 255, 0)
    else:
        print("Class did not find!!! ", value)


def create_prediction_map(output_path, new_labels, new_logits_map):
    h, w = new_labels.shape

    im_array = np.empty([h, w, 3], dtype=np.uint8)
    im_array_prob = np.empty([h, w, 3], dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            color = class_to_color(int(new_labels[i, j]))
            im_array[i, j, :] = color
            im_array_prob[i, j, :] = (np.asarray(color) * new_logits_map[i, j, int(new_labels[i, j])]).astype(int)

    img = Image.fromarray(im_array)
    img.save(os.path.join(output_path, 'predMap.jpeg'))

    img2 = Image.fromarray(im_array_prob)
    img2.save(os.path.join(output_path, 'predMap_prob.jpeg'))


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
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25
    # print x.get_shape()

    conv1 = _conv_layer(x, [4, 4, 3, 64], name_prefix + '_conv1', weight_decay,
                        is_training, pad='VALID', is_normal_conv=True, activation='lrelu')
    pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name=name_prefix + '_pool1', pad='VALID')

    return pool1


def convnet_25_temporal(x, dropout, is_training, crop_size, weight_decay):
    pools = []

    for i in range(x.get_shape()[0]):
        pools.append(convnet_initial(x[i], dropout, is_training, weight_decay, crop_size, 'time_' + str(i)))

    # conv1 = _conv_layer(x, [4, 4, 3, 64], 'ft_conv1', weight_decay, is_training, pad='VALID')
    # pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='ft_pool1', pad='VALID')

    try:
        pool_concat = tf.concat(pools, 3)
    except:
        pool_concat = tf.concat(concat_dim=3, values=pools)

    conv2 = _conv_layer(pool_concat, [4, 4, 64*int(x.get_shape()[0]), 128], 'ft_conv2', weight_decay,
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


def validate(sess, data, labels, test_distribution, crop_size, mean_full, std_full,
             n_input_data, batch_size, x, y, keep_prob, is_training,
             pred, acc_mean, step):
    all_predcs = []
    cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    true_count = 0.0

    index = np.arange(len(test_distribution))

    for i in range(0, (len(test_distribution) / batch_size if len(test_distribution) % batch_size == 0
                                                            else (len(test_distribution) / batch_size) + 1)):
        batch = index[i * batch_size:min(i * batch_size + batch_size, len(test_distribution))]
        bx, by = dynamically_create_patches(data, labels, crop_size, test_distribution, batch)
        normalize_images(bx, mean_full, std_full)
        bx = np.reshape(bx, (len(data), -1, n_input_data))

        preds_val, acc_mean_val = sess.run([pred, acc_mean],
                                           feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})
        true_count += acc_mean_val

        all_predcs = np.concatenate((all_predcs, preds_val))

        for j in range(len(preds_val)):
            cm_test[by[j]][preds_val[j]] += 1

    _sum = 0.0
    for i in range(len(cm_test)):
        _sum += (cm_test[i][i] / float(np.sum(cm_test[i])) if np.sum(cm_test[i]) != 0 else 0)

    print("---- Iter " + str(step) + " -- Validate: Overall Accuracy= " + str(int(true_count)) +
          " Overall Accuracy= " + "{:.6f}".format(true_count / float(len(test_distribution))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          # " Kappa= " + "{:.4f}".format(cohen_kappa_score(classes, np.asarray(all_predcs))) +
          " Confusion Matrix= " + np.array_str(cm_test).replace("\n", "")
          )


def train(data, labels, training_distribution, test_distribution, mean_full, std_full,
          crop_size, batch_size, niter, model_path,
          x, y, keep_prob, dropout, is_training, n_input_data,
          optimizer, loss, acc_mean, pred, output_path):
    # TRAIN NETWORK
    ###################
    display_step = 50
    epoch_number = 1000  # int(len(training_classes)/batch_size) # 1 epoch = images / batch
    val_inteval = 1000  # int(len(training_classes)/batch_size)
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
    shuffle = np.asarray(random.sample(range(3 * len(training_distribution)), 3 * len(training_distribution)))

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # Launch the graph
    with tf.Session(config=tfconfig) as sess:
        if 'model' in model_path:
            current_iter = int(model_path.split('_')[-1])
            print(BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC)
            saver_restore.restore(sess, model_path)
        else:
            sess.run(init)
            print(BatchColors.OKBLUE + 'Model totally initialized!' + BatchColors.ENDC)

        # aux variables
        it = 0
        epoch_mean = 0.0
        epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
        batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

        # Keep training until reach max iterations
        for step in range(current_iter, niter + 1):
            shuffle, batch, it = select_batch(shuffle, batch_size, it, 3 * len(training_distribution))

            b_x, batch_y = dynamically_create_patches(data, labels, crop_size, training_distribution, batch)
            normalize_images(b_x, mean_full, std_full)
            batch_x = np.reshape(b_x, (len(data), -1, n_input_data))

            # Run optimization op (backprop)
            _, batch_loss, batch_correct, batch_predcs = sess.run([optimizer, loss, acc_mean, pred],
                                                                  feed_dict={x: batch_x, y: batch_y,
                                                                             keep_prob: dropout, is_training: True})

            epoch_mean += batch_correct
            for j in range(len(batch_predcs)):
                epoch_cm_train[batch_y[j]][batch_predcs[j]] += 1

            if step % display_step == 0:
                # Calculate batch loss and accuracy
                for j in range(len(batch_predcs)):
                    batch_cm_train[batch_y[j]][batch_predcs[j]] += 1

                _sum = 0.0
                for i in range(len(batch_cm_train)):
                    _sum += (batch_cm_train[i][i] / float(np.sum(batch_cm_train[i])) if np.sum(
                        batch_cm_train[i]) != 0 else 0)

                print("Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
                      " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                      " Absolut Right Pred= " + str(int(batch_correct)) +
                      " Overall Accuracy= " + "{:.4f}".format(batch_correct / float(len(batch_y))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                      )
                batch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            if step % epoch_number == 0:
                _sum = 0.0
                for i in range(len(epoch_cm_train)):
                    _sum += (epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i])) if np.sum(
                        epoch_cm_train[i]) != 0 else 0)

                print("-- Iter " + str(step) + " -- Training Epoch:" +
                      " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(batch_size * epoch_number)) +
                      " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                      )

                epoch_mean = 0.0
                epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            if step % val_inteval == 0:
                # Test
                saver.save(sess, output_path + 'model_' + str(step))
                validate(sess, data, labels, test_distribution, crop_size, mean_full, std_full,
                         n_input_data, batch_size, x, y, keep_prob, is_training, pred, acc_mean, step)

        print(BatchColors.OKGREEN + "Optimization Finished!" + BatchColors.ENDC)

        # Test: Final
        saver.save(sess, output_path + 'model', global_step=step)
        validate(sess, data, labels, test_distribution, crop_size, mean_full, std_full,
                 n_input_data, batch_size, x, y, keep_prob, is_training, pred, acc_mean, step)


def full_test(data, labels, non_classified_pixels_distribution, mean_full, std_full,
              x, y, keep_prob, dropout, is_training, n_input_data, logits, pred,
              batch_size, crop_size, model_path, output_path):

    h, w = labels.shape
    new_labels = labels
    new_logits_map = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
    for i in range(h):
        for j in range(w):
            if labels[i, j] != 4:
                new_logits_map[i, j, labels[i, j]] = 1.0

    list_index = np.arange(len(non_classified_pixels_distribution))
    saver_restore = tf.train.Saver()

    with tf.Session() as sess:
        # current_iter = int(model_path.split('_')[-1])
        print(BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC)
        saver_restore.restore(sess, model_path)

        for i in range(0, ((len(non_classified_pixels_distribution) / batch_size)
                            if (len(non_classified_pixels_distribution) % batch_size) == 0
                            else (len(non_classified_pixels_distribution) / batch_size) + 1)):
            batch = list_index[i * batch_size:min(i * batch_size + batch_size, len(non_classified_pixels_distribution))]
            b_x, _ = dynamically_create_patches(data, labels, crop_size, non_classified_pixels_distribution, batch)
            normalize_images(b_x, mean_full, std_full)

            bx = np.reshape(b_x, (len(data), -1, n_input_data))
            by = np.zeros(len(b_x))

            _logits, _pred = sess.run([logits, pred], feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})
            for j in range(len(batch)):
                cur_x = non_classified_pixels_distribution[batch[j]][0]
                cur_y = non_classified_pixels_distribution[batch[j]][1]
                new_labels[cur_x, cur_y] = _pred[j]
                new_logits_map[cur_x, cur_y, :] = _logits[j, :]

    # with open(output_path+'final_logits','w') as f_handle:
    # np.savetxt(f_handle,all_logits,delimiter=',',newline='\n',fmt='%.5f')

    # with open(output_path+'logits_map','w') as f_handle:
    # np.savetxt(f_handle, new_logits_map, delimiter=',', newline='\n', fmt='%.5f %.5f %.5f')

    # with open(output_path+'map_labels','w') as f_handle:
    # np.savetxt(f_handle,new_labels,delimiter=',',newline='\n',fmt='%d')

    np.save(output_path + 'logits_map', new_logits_map)
    np.save(output_path + 'map_labels', new_labels)

    create_prediction_map(output_path, new_labels, new_logits_map)
    tf.reset_default_graph()


def testing_per_map(data, labels, class_dist, instances, mean_full, std_full,
                    x, y, keep_prob, dropout, is_training, n_input_data, logits, pred,
                    batch_size, crop_size, model_path, output_path):
    list_index = np.arange(len(class_dist))
    saver_restore = tf.train.Saver()
    first = True

    with tf.Session() as sess:
        print(BatchColors.OKBLUE + 'Model restored from ' + model_path + BatchColors.ENDC)
        saver_restore.restore(sess, model_path)

        for m in range(0, len(data)):  # for each map
            cur_data = data[m:m+1, :, :, :]
            all_preds = []
            all_gts = []
            print('Map', instances[m])
            print(cur_data.shape)

            for i in range(0, ((len(class_dist) / batch_size)
                              if (len(class_dist) % batch_size) == 0 else (len(class_dist) / batch_size) + 1)):
                batch = list_index[i * batch_size:min(i * batch_size + batch_size, len(class_dist))]
                b_x, by = dynamically_create_patches(cur_data, labels, crop_size, class_dist, batch)
                normalize_images(b_x, mean_full, std_full)
                bx = np.reshape(b_x, (len(cur_data), -1, n_input_data))

                _logits, _pred = sess.run([logits, pred], feed_dict={x: bx, y: by, keep_prob: 1., is_training: False})
                all_preds = np.concatenate((all_preds, _pred))
                all_gts = np.concatenate((all_gts, by))

            print(all_preds.shape)
            print(all_gts.shape)
            np.save(os.path.join(output_path, 'test_all_preds_' + instances[m].split('.')[0]), all_preds)
            if first is True:
                np.save(os.path.join(os.getcwd(), 'test_all_gts'), all_gts)
                first = False

    tf.reset_default_graph()


'''
Method for spatio-temporal (with branch nets) segmentation using whole time series
'''


def main():
    list_params = ['input_path', 'output_path (for model, images, etc)', 'model_path', 'instances',
                   'learning_rate', 'weight_decay', 'batch_size', 'niter', 'crop_size',
                   'operation [training|testing|full_test]']
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

    # image instances
    index = index + 1
    instances = sys.argv[index].split(',')

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
    operation = sys.argv[index]

    print(BatchColors.OKBLUE + 'Reading images...' + BatchColors.ENDC)
    data, labels = load_images(input_path, crop_size, instances)
    print(data.shape, labels.shape)

    training_class_dist, testing_class_dist, no_class_dist = create_distributions_over_pixel_classes(labels)

    if os.path.isfile(os.path.join(output_path, 'mean.npy')):
        mean_full = np.squeeze(np.load(os.path.join(output_path, 'mean.npy')))
        std_full = np.squeeze(np.load(os.path.join(output_path, 'std.npy')))
        print(BatchColors.OKGREEN + 'Loaded Mean/Std from training instances' + BatchColors.ENDC)
    else:
        mean_full, std_full = calculate_mean_and_std(data, training_class_dist, crop_size)
        np.save(os.path.join(output_path, 'mean.npy'), mean_full)
        np.save(os.path.join(output_path, 'std.npy'), std_full)
        print(BatchColors.OKGREEN + 'Created Mean/Std from training instances' + BatchColors.ENDC)

    print(mean_full.shape, std_full.shape)

    # Network Parameters
    n_input_data = crop_size * crop_size * 3  # RGB
    dropout = 0.5  # Dropout, probability to keep units

    # tf Graph input_data
    if operation == 'testing_per_map':
        x = tf.placeholder(tf.float32, [1, None, n_input_data], name='ph_data')
    else:
        x = tf.placeholder(tf.float32, [len(data), None, n_input_data], name='ph_data')
    y = tf.placeholder(tf.int32, [None], name='ph_labels')

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # CONVNET
    logits = convnet_25_temporal(x, keep_prob, is_training, crop_size, weight_decay)

    # Define loss and optimizer
    loss = loss_def(logits, y)
    lr = tf.train.exponential_decay(lr_initial, global_step, 50000, 0.1, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

    # Evaluate model
    correct = tf.nn.in_top_k(logits, y, 1)
    acc_mean = tf.reduce_sum(tf.cast(correct, tf.int32))
    pred = tf.argmax(logits, 1)

    if operation == 'training':
        train(data, labels, training_class_dist, testing_class_dist, mean_full, std_full,
              crop_size, batch_size, niter, model_path,
              x, y, keep_prob, dropout, is_training, n_input_data,
              optimizer, loss, acc_mean, pred, output_path)
    elif operation == 'testing_per_map':
        testing_per_map(data, labels, testing_class_dist, instances, mean_full, std_full,
                        x, y, keep_prob, dropout, is_training, n_input_data, logits, pred,
                        batch_size, crop_size, model_path, output_path)
    elif operation == 'test_Full_map':
        full_test(data, labels, no_class_dist, mean_full, std_full,
                  x, y, keep_prob, dropout, is_training, n_input_data, logits, pred,
                  batch_size, crop_size, model_path, output_path)
    else:
        print(BatchColors.FAIL + "Operation not found: " + operation + BatchColors.ENDC)


if __name__ == "__main__":
    main()
