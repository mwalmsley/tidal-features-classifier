'''This script goes along the blog post
"Building powerful image classification models using very little data"
'''
import os
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from meta_CNN import fold_tables, custom_flow_from_directory, recordMetrics, define_parameters
from helper_funcs import ThreadsafeIter, shuffle_df, load_lines
import keras.optimizers
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import seaborn as sns


from keras import backend as K
K.set_image_dim_ordering('th')

def load_VGG(instruct): # only loads CNN base layers
    channels = instruct['channels']
    img_width = instruct['img_width']
    img_height = instruct['img_height']
    weights_path = instruct['weights_path']

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(channels, img_width, img_height)))

    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # load the weights of the VGG16 networks
    # (trained on ImageNet, won the ILSVRC competition in 2014)
    assert os.path.exists(weights_path), 'Model weights not found (see "weights_path" variable in script).'
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    return model

def save_bottleneck_features(train_table, val_table, instruct):

    model = load_VGG(instruct)

    instruct['save_gen_output'] = True

    print(instruct)
    print(train_table.head())
    custom_gen_train = custom_flow_from_directory(train_table, instruct, gen_name='train_labels')
    custom_gen_val = custom_flow_from_directory(val_table, instruct, gen_name='val_labels')
    custom_gen_train = ThreadsafeIter(custom_gen_train)
    custom_gen_val = ThreadsafeIter(custom_gen_val)

    print('begin bottleneck predictions')
    bottleneck_features_train = model.predict_generator(custom_gen_train, instruct['nb_train_samples'])
    print('bottleneck predictions made')
    print('shape: ', bottleneck_features_train.shape)
    np.save(open('/exports/eddie/scratch/s1220970/bottleneck_features_train.npy', 'w'), bottleneck_features_train)
    print('bottleneck predictions saved')

    bottleneck_features_validation = model.predict_generator(custom_gen_val, instruct['nb_validation_samples'])
    print('bottleneck validation predictions made')
    np.save(open('/exports/eddie/scratch/s1220970/bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    print('bottleneck predictions saved')
    return train_table, val_table


def define_top_model(instruct):

    train_data = np.load(open('/exports/eddie/scratch/s1220970/bottleneck_features_train.npy'))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # model.load_weights(instruct['top_model_weights_path'])

    return model

def train_top_model(instruct):

    train_data = np.load(open('/exports/eddie/scratch/s1220970/bottleneck_features_train.npy'))
    # train_labels = load_lines(instruct['directory'] + 'train_labels' + '_' + str(instruct['run']) + '_label.txt')[:instruct['nb_train_samples']]
    train_labels = np.ravel(np.loadtxt(instruct['directory'] + 'train_labels' + '_' + str(instruct['run']) + '_label.txt'))[:instruct['nb_train_samples']]

    validation_data = np.load(open('/exports/eddie/scratch/s1220970/bottleneck_features_validation.npy'))
    validation_labels = np.ravel(np.loadtxt(instruct['directory'] + 'val_labels' + '_' + str(instruct['run']) + '_label.txt'))[:instruct['nb_validation_samples']]

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

    print('training top model')
    print('training data example:')
    # print(train_data[0])
    # print(train_data[0].max())
    print(train_data.shape)
    print('train labels:')
    # print(train_labels)
    print(train_labels.shape)
    # print('')
    print('val data example:')
    # print(validation_data[0])
    # print(validation_data[0].max())
    print(validation_data.shape)
    print('validation labels:')
    # print(validation_labels)
    print(validation_labels.shape)
    # print('')
    hist = model.fit(train_data, train_labels,
              nb_epoch=instruct['nb_epoch'], batch_size=instruct['batch_size'],
              validation_data=(validation_data, validation_labels),verbose=2)
    model.save_weights(instruct['top_model_weights_path'])


    print('Top model training complete')
    hist_dict = hist.history
    acc = np.array([v for v in hist_dict['acc']])
    val_acc = np.array([v for v in hist_dict['val_acc']])
    loss = np.array([v for v in hist_dict['loss']])
    val_loss = np.array([v for v in hist_dict['val_loss']])

    return acc, loss, val_acc, val_loss

def train_both(train_table, val_table, instruct):
    # be careful to use the same train and val table as top model was already trained on, don't mix!

    runs = instruct['runs']
    nb_epoch = instruct['nb_epoch']
    batch_size = instruct['batch_size']
    nb_train_samples = instruct['nb_train_samples']
    nb_validation_samples = instruct['nb_validation_samples']

    model = load_VGG(instruct)
    top_model = define_top_model(instruct)

    top_model.load_weights(instruct['top_model_weights_path'])

    model.add(top_model)

    for layer in model.layers[:25]:
        layer.trainable = False

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    print('Begin fine-tuning')

    # same label file locations as before, don't run in parallel TODO
    instruct['save_gen_output'] = True
    custom_gen_train = custom_flow_from_directory(train_table, instruct, gen_name='train_labels')
    custom_gen_val = custom_flow_from_directory(val_table, instruct, gen_name='val_labels')
    custom_gen_train = ThreadsafeIter(custom_gen_train)
    custom_gen_val = ThreadsafeIter(custom_gen_val)

    hist = model.fit_generator(
        custom_gen_train,
        samples_per_epoch=nb_train_samples,
        nb_epoch=nb_epoch,
        validation_data=custom_gen_val,
        nb_val_samples=nb_validation_samples,
        verbose=2,
    )

    print('Fine-tuning complete')
    hist_dict = hist.history
    acc = np.array([v for v in hist_dict['acc']])
    val_acc = np.array([v for v in hist_dict['val_acc']])
    loss = np.array([v for v in hist_dict['loss']])
    val_loss = np.array([v for v in hist_dict['val_loss']])

    return acc, loss, val_acc, val_loss, model


def plot_metrics(acc, loss, val_acc, val_loss, name):
    # don't change this until important tests complete
    plt.figure(1)
    plt.subplot(121)
    sns.tsplot(val_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])
    plt.title('Validation')

    plt.subplot(122)
    sns.tsplot(acc)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1.0])
    # plt.legend(['Train set', 'Validation set'], loc=0)
    plt.title('Training')

    plt.figure(1).subplots_adjust(left=0.1, right=0.9, wspace=0.25)
    plt.savefig(name + '_acc.png')

    plt.show()


def runVGG():
    eddie=True

    print('Begin function')

    instruct, meta = define_parameters(eddie)

    instruct['nb_epoch'] = 100 # how many epochs to train on fixed dataset
    instruct['batch_size'] = 32
    instruct['nb_train_samples'] = 500*instruct['batch_size'] # size of fixed dataset
    instruct['nb_validation_samples'] = 100*instruct['batch_size']

    instruct['crop'] = True
    instruct['w'] = 112
    instruct['img_width'] = 224
    instruct['img_height'] = 224
    # instruct['vgg_zoom'] = 0.7 # TODO: test with shrinking the image before cropping, to effectively resize 256->150
    instruct['vgg_zoom'] = None

    instruct['run_n'] = 0

    instruct['name'] = 'vgg_l4'
    # path to the model weights file.
    if eddie != False: weights_dir = '/exports/eddie/scratch/s1220970/'
    else: weights_dir = '/media/mike/SandiskLinux/'
    instruct['weights_path'] = weights_dir + 'vgg16_weights.h5'
    print(instruct['weights_path'])
    instruct['top_model_weights_path'] = weights_dir + 'bottleneck_fc_model.h5'

    instruct['input_mode'] = 'threshold_color_3sig'
    instruct['channels'] = 3
    instruct['scale'] = 'log'
    instruct['clip'] = False
    instruct['convolve'] = None

    # shuffle, fold, and choose first will give a unique fold each time
    instruct['runs'] = 2
    instruct['folds'] = 5
    runs = instruct['runs']

    top_epochs = 150
    finetune_epochs = 2

    acc = np.zeros((runs, finetune_epochs))
    val_acc = np.zeros((runs,finetune_epochs))
    loss = np.zeros((runs,finetune_epochs))
    val_loss = np.zeros((runs,finetune_epochs))

    top_acc = np.zeros((runs,top_epochs))
    top_val_acc = np.zeros((runs,top_epochs))
    top_loss = np.zeros((runs,top_epochs))
    top_val_loss = np.zeros((runs,top_epochs))

    print('reached loop')
    for run in range(runs):
        instruct['run'] = run
        meta = shuffle_df(meta)
        folded_train_tables, folded_val_tables = fold_tables(meta, instruct)
        train_table = folded_train_tables[0]
        val_table = folded_val_tables[0]
        instruct['nb_epoch'] = top_epochs
        print('saving initial features')
        train_table, val_table = save_bottleneck_features(train_table, val_table, instruct)
        print('training bottleneck')
        top_acc, top_loss, top_val_acc, top_val_loss = train_top_model(instruct)
        print('training both')
        instruct['nb_epoch'] = finetune_epochs
        acc[run, :], loss[run, :], val_acc[run, :], val_loss[run, :], model = train_both(train_table, val_table, instruct)
    print(acc)
    print(loss)
    print(val_acc)
    print(val_loss)

    instruct['name'] = instruct['name'] + '_top'
    recordMetrics(top_acc, top_loss, top_val_acc, top_val_loss, instruct)
    plot_metrics(top_acc, top_loss, top_val_acc, top_val_loss, instruct)

    instruct['name'] = instruct['name'][-4] + '_finetune'
    recordMetrics(acc, loss, val_acc, val_loss, instruct['name'])
    plot_metrics(acc, loss, val_acc, val_loss, instruct['name'])


runVGG()