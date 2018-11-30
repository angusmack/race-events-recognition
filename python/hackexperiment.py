# Experiment to predict collisions

# the following two lines are to make REPL find all modules
import sys
sys.path.append('c:\\work\\race-events-recognition')

import math
import os
from pipe import *
import pypln as pp
from pypln.utils.pipeutils import *
from pypln.utils.video import *
import importlib
import numpy as np
import cv2
import os
from keras import layers
from keras import models
from keras import optimizers
from keras.regularizers import l2
import keras
import functools

# configuration variables
data_dir = 'd:\\data'
classes = { 'collisions' : 1, 'not-collisions': 0 }


#### EXPERIMENT 1: OPTICAL FLOW

# Functions to compute optical flow descriptors

def compute_flow_descriptor(video,bins=100,mag_min=-7,mag_max=7,stabilize=True):
    def descr(f1,f2,abins,mbins):
        print(f1.shape)
        f = dense_optical_flow(f1,f2)
        if stabilize:
              f = naive_stabilization(f)
        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
        h1,bins1 = np.histogram(ang.ravel(),bins=abins)
        h2,bins2 = np.histogram(mag.ravel(),bins=mbins)
        return [[x,y] for x,y in zip(h1,np.log(1+h2))] # we take log of histograms to make range smaller
    abins = [i*2*math.pi/bins for i in range(0,bins+1)]
    mbins = np.arange(bins+1)/bins*(mag_max-mag_min)+mag_min
    return np.array([ descr(video[i],video[i+1],abins=abins,mbins=mbins) for i in range(0,len(video)-1)])

def normalize_histograms(video,ang_diap,mag_diap):
    video[:,:,0] = (video[:,:,0]-ang_diap[0]) / (ang_diap[1]-ang_diap[0])
    video[:,:,1] = (video[:,:,1]-mag_diap[0]) / (mag_diap[0]-mag_diap[1])
    return video

## Compute Optical Flow Descriptors

videostream = \
    pp.get_datastream(data_dir,'mp4',classes,split_filename='split.txt')\
    | pp.lzapply('filename','video',lambda fn: pp.load_video(fn,video_size=(None,300)))\
    | pp.apply_npy('video','optflow',compute_flow_descriptor,file_ext='.flows.npy')\
    | pp.delfield('video')

videostream = videostream | as_list # this will cause all optical flows to computed on disk

# Compute max no of frames per video
max_frames = videostream | select (lambda x : x['optflow'].shape[0]) | max

# Compute the mix/max values of angle and magnitude of optical flow vectors
maxs = videostream | select(lambda x : x['optflow'].max(axis=(0,1))) | as_npy
mins,maxs,avgs = maxs.min(axis=0),maxs.max(axis=0),np.average(maxs,axis=0)
ang_diap = (mins[0],avgs[0])
mag_diap = (mins[1],maxs[1])

#ang_diap=(7112.0, 80842.35460992908)
#mag_diap=(9.06126014896203, 11.982310152711182)

train_flow,test_flow = \
    videostream \
    | pp.apply('optflow','optflow',functools.partial(pp.zero_pad,max_frames=max_frames))\
    | pp.apply('optflow','optflow',functools.partial(normalize_histograms,ang_diap=ang_diap,mag_diap=mag_diap))\
    | pp.make_train_test_split

model = models.Sequential()
model.add(layers.AveragePooling2D((2, 2),input_shape=(max_frames, 100, 2)))
model.add(layers.Conv2D(8, (3, 3), data_format='channels_last',activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(layers.Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.compile(loss='binary_crossentropy',
              #optimizer=optimizers.Nadam(lr=0.005),
              optimizer=optimizers.adagrad(),
              metrics=['acc'])
model.summary()

checkpointer = keras.callbacks.ModelCheckpoint(filepath='d:/data/optflow-{epoch}-{val_acc}.hdf5', verbose=1, save_best_only=True)
#estopper = keras.callbacks.EarlyStopping(monitor='val_acc', patience=30, baseline=0.8) # restore_best_weights=True,
hist = model.fit_generator(
      train_flow | infshuffle() | pp.as_batch(feature_field_name='optflow',label_field_name='class_id'),
      steps_per_epoch=100,
      validation_data= test_flow | infshuffle() | pp.as_batch(feature_field_name='optflow',label_field_name='class_id'),
      use_multiprocessing=False, # has to be false on windows..
      validation_steps = 10,
      callbacks=[checkpointer],
      epochs=30)

hist.history['val_acc'] | max
# 0.78
# 0.81875

#### EXPERIEMENT 2: VGG Embeddings

video_size=(177,100)

vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=video_size[::-1] + (3,))

videostream = \
    pp.get_datastream(data_dir,'mp4',classes,split_filename='split.txt')\
    | pp.lzapply('filename','video',lambda fn: pp.load_video(fn,video_size=(177,100)))\
    | as_list

videostream | select(lambda x: x['video'].shape[0]) | max
videostream = videostream \
 | pp.apply_npy('video','vgg16',lambda x: vgg.predict(keras.applications.vgg16.preprocess_input(x)),file_ext='.vgg16.npy')\
 | as_list

(videostream | first)['vgg16'].shape
# 125x5x3x512

max_frames = videostream | select (lambda x : x['vgg16'].shape[0]) | max

train_flow = videostream | pp.filtersplit('T') \
             | pp.apply('vgg16','vgg16',functools.partial(pp.zero_pad,max_frames=max_frames)) \
             | select(lambda x: { "features": x['vgg16'].reshape(max_frames,-1,1), "label" : x["class_id"] }) \
             | infshuffle() | pp.as_batch()
test_flow = videostream | pp.filtersplit('V') \
             | pp.apply('vgg16','vgg16',functools.partial(pp.zero_pad,max_frames=max_frames)) \
             | select(lambda x: { "features": x['vgg16'].reshape(max_frames,-1,1), "label" : x["class_id"] }) \
             | infshuffle() | pp.as_batch()

model = models.Sequential()
model.add(layers.AveragePooling2D((2, 2),input_shape=(max_frames, 7680, 1)))
model.add(layers.Conv2D(8, (3, 3), data_format='channels_last',activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(layers.AveragePooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.add(layers.Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model.compile(loss='binary_crossentropy',
              #optimizer=optimizers.Nadam(lr=0.005),
              optimizer=optimizers.adam(),
              metrics=['acc'])
model.summary()

checkpointer = keras.callbacks.ModelCheckpoint(filepath='d:/data/vgg16-{epoch}-{val_acc}.hdf5', verbose=1, save_best_only=True)
estopper = keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, baseline=0.8) # restore_best_weights=True,
hist = model.fit_generator(
      train_flow,
      steps_per_epoch=100,
      validation_data= test_flow,
      callbacks=[checkpointer,estopper],
      use_multiprocessing=False, # has to be false on windows..
      validation_steps = 10,
      epochs=20)

hist.history['val_acc'] | max
# 0.84

model.save(os.path.join(data_dir,'vgg_model.hdf5'))

### EXPERIMENT 3: Combined experiment

videostream = \
    pp.get_datastream(data_dir,'mp4',classes,split_filename='split.txt')\
    | pp.apply('filename','optflow', lambda x: np.load(x+'.flows.npy')) \
    | pp.apply('filename', 'vgg16', lambda x: np.load(x + '.vgg16.npy')) \
    | as_list

max_frames = videostream | select (lambda x : x['vgg16'].shape[0]) | max

train_flow,test_flow = videostream \
                       | pp.apply('optflow','optflow',functools.partial(pp.zero_pad,max_frames=max_frames)) \
                       | pp.apply('optflow', 'optflow', functools.partial(normalize_histograms, ang_diap=ang_diap, mag_diap=mag_diap)) \
                       | pp.apply('vgg16', 'vgg16', functools.partial(pp.zero_pad,max_frames=max_frames)) \
                       | pp.apply('vgg16', 'vgg16', lambda x: x.reshape(max_frames, -1, 1)) \
                       | pp.make_train_test_split

model1 = models.Sequential()
model1.add(layers.AveragePooling2D((2, 2),input_shape=(max_frames, 100, 2)))
model1.add(layers.Conv2D(8, (3, 3), data_format='channels_last',activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model1.add(layers.AveragePooling2D((2, 2)))
model1.add(layers.Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model1.add(layers.AveragePooling2D((2, 2)))
model1.add(layers.Flatten())
model1.add(layers.Dropout(0.5))
model1.add(layers.Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))

model2 = models.Sequential()
model2.add(layers.AveragePooling2D((4, 4),input_shape=(max_frames, 7680, 1)))
model2.add(layers.Conv2D(8, (3, 3), data_format='channels_last',activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model2.add(layers.AveragePooling2D((4, 4)))
model2.add(layers.Conv2D(16, (3, 3) ,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))
model2.add(layers.AveragePooling2D((4, 4)))
model2.add(layers.Flatten())
model2.add(layers.Dropout(0.5))
model2.add(layers.Dense(100,activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01)))

input1 = keras.Input(shape=(max_frames,100,2))
input2 = keras.Input(shape=(max_frames,7680,1))
concat = keras.layers.concatenate([model1(input1),model2(input2)])
output = layers.Dense(1, activation='sigmoid',kernel_initializer='glorot_uniform',kernel_regularizer=l2(0.01))(concat)

model = keras.Model(input=[input1,input2],output=output)

model.summary()

model.compile(loss='binary_crossentropy',
              #optimizer=optimizers.Nadam(lr=0.005),
              optimizer=optimizers.adam(),
              metrics=['acc'])


hist = model.fit_generator(
      train_flow | infshuffle() | pp.as_batch(feature_field_name=['optflow','vgg16'],label_field_name='class_id'),
      steps_per_epoch=100,
      validation_data= test_flow | infshuffle() | pp.as_batch(feature_field_name=['optflow','vgg16'],label_field_name='class_id'),
      use_multiprocessing=False, # has to be false on windows..
      validation_steps = 10,
      epochs=20)

hist.history['val_acc'] | max
# 0.88

model.save(os.path.join(data_dir,'model.hdf5'))
print(max_frames)