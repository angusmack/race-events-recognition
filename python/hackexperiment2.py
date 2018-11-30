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
import keras
import functools

data_dir = 'd:\\data'
classes = { 'collisions' : 1, 'not-collisions': 0 }

vgg = keras.applications.vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(100,177,3))
model = keras.models.load_model(os.path.join(data_dir,'vgg16.hdf5'))
flowmodel = keras.models.load_model(os.path.join(data_dir,'optflow.hdf5'))

def imprint(frames,values):
    for i,x in enumerate(frames):
        for j,z in enumerate(values):
            dx = 10+15*j
            clr = (255,0,0) if z<0.5 else (0,255,0)
            cv2.rectangle(x,(dx,10),(dx+10,90),clr,1)
            cv2.line(x,(dx+5,90),(dx+5,90-int(80*z)),clr,10)
    return frames

def compute_flow_descriptor(video,bins=100,mag_min=-7,mag_max=7,stabilize=True):
    def descr(f1,f2,abins,mbins):
        f = cv2.cvtColor(f1,cv2.COLOR_BGR2GRAY)
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

ang_diap=(7112.0, 80842.35460992908)
mag_diap=(9.06126014896203, 11.982310152711182)

videostream = pp.videosource_chunked('d:\\data\\Berlin-Grand-Prix.mp4',video_size=(533,300))
res = []
graph = ( videostream
  | pp.chunk_slide(5)
  | take(100)
  | pp.to_field('video')
  | pp.apply('video', 'optflow', compute_flow_descriptor)
  | pp.apply('video','rvideo',lambda x: resize_video(x,video_size=(177,100)))
  | pp.apply('rvideo','score',lambda x: model.predict(vgg.predict(keras.applications.vgg16.preprocess_input(x)).reshape(1,125,7680,1))[0][0])
  | pp.apply('optflow','optflow',functools.partial(normalize_histograms,ang_diap=ang_diap,mag_diap=mag_diap))\
  | pp.apply('optflow','optscore',lambda x: flowmodel.predict(np.expand_dims(x,0))[0][0])
  | pp.iter(['score','optscore'],lambda x: res.append(x))
  | pp.iter(['score','optscore'],lambda x: print(x))
  | pp.apply(['video','score','optscore'],'video',lambda args: imprint(args[0][50:75],args[1:]))
  | pp.extract_field('video')
  | pp.collect_video('d:\\data\\out.mp4')
  )

res

([1,2,3]
 | pp.to_field('data')
 | pp.lzapply('data','data', lambda x: x*2)
 | as_list
 )