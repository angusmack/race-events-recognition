## Advanced video streaming utils that use optical flow, VGG embeddings, etc

import cv2
import numpy as np
import utils.streams as ss
import mPyPl.utils.video as vid
from pipe import *
import keras as K
import math

def get_flow_descriptor(video,bins=100,mag_min=-7,mag_max=7,stabilize=True):
    def descr(f1,f2,abins,mbins):
        f = vid.dense_optical_flow(f1,f2)
        if stabilize:
              f = vid.naive_stabilization(f)
        mag, ang = cv2.cartToPolar(f[..., 0], f[..., 1])
        h1,bins1 = np.histogram(ang.ravel(),bins=abins)
        h2,bins2 = np.histogram(mag.ravel(),bins=mbins)
        return [[x,y] for x,y in zip(h1,np.log(1+h2))] # we take log of histograms to make range smaller
    abins = [i*2*math.pi/bins for i in range(0,bins+1)]
    mbins = np.arange(bins+1)/bins*(mag_max-mag_min)+mag_min
    return np.array([ descr(video[i],video[i+1],abins=abins,mbins=mbins) for i in range(0,len(video)-1)])


def adorn_opticalflow(filename, stabilize=True, height=300, bins=100):
    frames = vid.video_to_npy(filename,height=height)
    return get_flow_descriptor(frames,stabilize=stabilize,bins=bins)

    
def zero_pad(x,max_frames,axis=0):
    npad = [(0, 0) for x in range(len(x.shape))]
    npad[axis] = (0, max_frames-x.shape[axis])
    return np.pad(x,npad,'constant',constant_values=0)


def normalize_histograms(video,ang_diap,mag_diap):
    video[:,:,0] = (video[:,:,0]-ang_diap[0]) / (ang_diap[1]-ang_diap[0])
    video[:,:,1] = (video[:,:,1]-mag_diap[0]) / (mag_diap[0]-mag_diap[1])
    return video