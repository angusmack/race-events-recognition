from os import listdir
from os.path import isfile, join, basename
from itertools import *
from pipe import *
import uuid
import pickle
import math
import numpy as np
from typing import TypeVar, Iterable, Tuple, Union
from mPyPl.utils.flowutils import *
from mPyPl.utils.video import video_to_npy
import mPyPl.utils.fileutils as ff
import random
import builtins
from keras.preprocessing.image import ImageDataGenerator
import math


def get_filestream(data_dir: str, ext: str) -> Iterable[str]:
    """
    Get a list of files from the given directory with specified extension  }
    :rtype: Iterable[str]
    """
    return listdir(data_dir) \
        | where(lambda p: p.endswith(ext)) \
        | select( lambda p: join(data_dir,p))

        
def get_datastream(data_dir, ext, classes):
    """
    Get a stream of objects for a number of classes specified as dict of the form { 'dir0' : 0, 'dir1' : 1, ... }
    Returns stream of dictionaries of the form { class_id: ... , class_name: ..., filename: ... }
    """
    return list(classes.items()) \
            | select(lambda kv: get_filestream(join(data_dir,kv[0]),ext)\
            | select(lambda x: { "filename": x, "class_id": kv[1], "class_name": kv[0] })) \
            | chain
        
        
@Pipe
def apply(datastream, src_field, dst_field, func):
    """
    Applies a function to the specified field of stream and adds appropriate field 
    """
    def applier(x):
        if isinstance(src_field, list) or isinstance(src_field, np.ndarray):
            x[dst_field] = func([x[key] for key in x.keys() if key in src_field])
        else:
            x[dst_field] = func(x[src_field])
        return x
    return datastream | select (applier) 


@Pipe
def adorn_video(video, video_size=(100,100), squarecrop=False, fps=25, maxlength=5, use_cache=False):
    """
    Adorn video content 
    """
    return video_to_npy(
        video,
        width=video_size[0],
        height=video_size[1], 
        squarecrop=squarecrop,
        fps=fps,
        maxlength=maxlength,
        use_cache=use_cache
    )


@Pipe
def resize_video(video, video_size=(100,100)):
    """
    Resize video content
    """
    height, width = video_size
    width = width if width else int(height / video[0].shape[0] * video[0].shape[1])
    height = height if height else int(width / video[0].shape[1] * video[0].shape[0])
    video = [ cv2.resize(frame, (width, height)) for frame in video ]
    return video


@Pipe
def adorn_npy(args, func, file_ext='.npy'):
    """
    Adorn dataset with custom numpy data.
    Func takes input dictionary and returns numpy array
    Results are cached on disk with `file_ext` extension
    """
    npfile, video = args
    fn=npfile + file_ext
    if isfile(fn):
        return np.load(fn)
    else:
        res = func(video)
        np.save(fn,res)
        return res


def count_classes(datastream):
    dic = {}
    for x in datastream:
        if x["class_name"] in dic:
            dic[x["class_name"]] += 1
        else:
            dic[x["class_name"]] = 1
    return dic


def make_split(datastream,split=0.2):
    cls = count_classes(datastream)
    mx = builtins.min(cls.values())
    n = int(mx*split)
    res = list(cls.keys()) | select (lambda c : datastream | where (lambda x: x["class_name"]==c) | as_list | pshuffle | take(n)) | chain
    return res | select(lambda x: basename(x["filename"])) | as_list


class StreamSplitter:
    """
    Split datastream into train and test datasets
    Supports writing 'split.txt' file if it does not exist.
    It expects the following structure on disk:
    base_dir
       class1
          file1.ext
          ...
       class2
          ...
       split.txt
    """

    def __init__(self,datastream,splitfile='split.txt',split=0.2):
        self.datastream = list(datastream) ## dsh -- All data is stored in memory! I did not find a way around it :(
        if isfile(splitfile):
            self.test = ff.readlines(splitfile)
        else:
            self.test = make_split(self.datastream,split=split)
            ff.writelines(splitfile,self.test)

    def get_train(self):
        return self.datastream | where (lambda x: basename(x['filename']) not in self.test)

    def get_test(self):
        return self.datastream | where (lambda x: basename(x['filename']) in self.test)

    def get_all(self):
        return self.datastream

@Pipe
def as_batch(flow, npsize, feature_field_name='features', label_field_name='label', batchsize=16):
    while (True):
        batch = np.zeros((batchsize,)+npsize)
        labels = np.zeros((batchsize, 1))
        for i in range(batchsize):
            data = next(flow)
            batch[i] = data[feature_field_name]
            labels[i] = data[label_field_name]
        yield (batch, labels)

class VideoStream:

    def __init__(self,
                 video_size=64,
                 max_length=5,
                 framerate=25,
                 squarecrop=True,
                 use_cache=False,
                 batch_size=16
                 ):
        self.data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=5,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.1,
                             horizontal_flip=True)
        self.video_size=video_size
        self.max_length=max_length
        self.framerate=framerate
        self.squarecrop = squarecrop
        self.batchsize=batch_size,
        self.use_cache = use_cache

    def dump_class_distribution(self,file_name: str = "class_structure.p"):
        """
        Dump the current class distribution of the data into a pickle file
        so that others can use the file to harmonise.

        :param file_name:
        """
        # dump a classes file so the others can see which classes I have assigned
        classes = self.get_videos(self.not_collisions, 1) \
                  | chain_with(self.get_videos(self.collisions, 0)) \
                  | select(lambda i: (i[1].replace(self.collisions, '').replace(self.not_collisions, ''), i[0])) \
                  | as_dict()

        print('Written %s' % file_name)
        pickle.dump(classes, open(file_name, "wb"))


    def createValidationSplit(self):
        """
        Create the validation split randomly and write it to a pickle file, which can them be fed into
        the videoStream method. Note that Tim has already generated a canonical validation split file
        and checked into SCC which we can use as a baseline.
        """
        collisions = self.get_videos(self.collisions, 1) | as_list() | pshuffle()
        ncollisions = self.get_videos(self.not_collisions, 0) | as_list() | pshuffle()

        # what is the min number in either class?
        minn = [collisions | count, ncollisions | count] | min

        # now let's validate on 33% of that
        vs = math.floor(minn * 0.33)

        # run this when you need to
        valset = collisions | take(vs) | chain_with(ncollisions | take(vs)) | as_list()
        # let's save this valsplit to the filesystem for use later!
        pickle_file_name = "valsplit_%s.p" % uuid.uuid4().hex

        pickle.dump(valset, open(pickle_file_name, "wb"))
        print( 'written to %s' % (pickle_file_name) )

    def video_stream(self,
                     val_split_file='valsplit_1b46b8e12c244afdaf09f83241fb443a.p',
                     validation_data=False):
        """
        Get an infinite stream of training videos from the respective paths. Will randomise
        all the videos but stream out in a 50:50 label distribution on average. It will load in a
        separate validation set from the supplied pickle file. This pickle file has the format;

        [(1, './videos/collisions\\output003.mp4'),
         (1, './videos/collisions\\output070-24132.mp4'),
         (0, './videos/collisions\\output074-26961.mp4'),
         (1, './videos/collisions\\output008-26961.mp4'),
         [...]
         ]

        :param val_split_file: Your validation split file, the default value is one which tim loaded into SCC and uses
        :param cpath: collisions path
        :param ncpath: non-collisions path
        """
        valset = pickle.load(open(val_split_file, "rb"))

        # handy dict to see if a file is in the validation set
        vd = valset | select(lambda t: (t[1], t[0])) | as_dict()

        collisions = self.get_videos(self.collisions, 1) | as_list() | pshuffle()
        ncollisions = self.get_videos(self.not_collisions, 0) | as_list() | pshuffle()

        while True:

            # create an infinite sequence of both, taking the same from both to ensure
            # same average label distribution

            ic = collisions | where(lambda v: v[1] in vd if validation_data else v[1] not in vd) | as_list() | pshuffle() | take(100)
            inc = ncollisions | where(lambda v: v[1] in vd if validation_data else v[1] not in vd) | as_list() | pshuffle() | take(100)

            mixed_stream = (ic
                            | chain_with(inc)
                            | as_list()
                            # shuffle it so we get good label mix
                            | pshuffle())

            for e in mixed_stream:
                yield e

    def adorn_video(self, iterator):
        """
        Load videos for the supplied flow/iterator, this will happen lazily
        :param iterator: the flow of videos
        :use_cache cache the numpy in a dictionary, only use this if you have enough RAM on your machine
        to hold all of the videos! At 25fps 60^2 they are all 600MB, this blows up very fast.
        :return: iterator
        """
        return iterator \
               | select(lambda f: f + (
            video_to_npy(f[1],
                         # note weird thing here, width doesn't work they appear to be inverted
                         height=self.video_size,
                         squarecrop=self.squarecrop,
                         fps=self.framerate,
                         maxlength=self.max_length,
                         # save a npy replacement
                         outfile=self.get_numpy_filename(f[1]),
                         use_cache=self.use_cache
                         ),))

    def augmentation(
        self,
        flow,
        normalize=True):
        """
        Take an existing flow of data and augment it with consistent random transformations uniformly accross all
        frames in the video. Also supports normalization /in [0,1]

        :param flow:  the iterator of video data
        :param normalize: do you want to scale the data to [0,1] using a linear map
        """

        image_datagen = ImageDataGenerator(**self.data_gen_args)

        for video in flow:
            # for every frame in this video generate the same transformation
            # and yield it all back in sequence order
            trans = image_datagen.get_random_transform(video[2].shape)
            augmentedVideo = np.zeros(video[2].shape)
            for i in range(video[2].shape[0]):
                augmentedVideo[i] = image_datagen.apply_transform(video[2][i], trans)

                # now is a good time to transform the video onto 0-1
                # we need to do this to get convergence when we train i.e. homogenise features
                if normalize:
                    augmentedVideo[i] = augmentedVideo[i] / 255

            yield video[:-1] + (augmentedVideo,)


    def zero_pad_videos(self, videoStream):
        for video in videoStream:
            zeroPadded = np.zeros((self.max_length * self.framerate, self.video_size, self.video_size, 3))
            zeroPadded[0:video[2].shape[0], :, :, :] = video[2]
            yield video[:-1] + (zeroPadded,)

    # we will need to do the same thing to the validation stream as what happened to the train stream!
    @staticmethod
    def normalize_videos(videoStream):
        for video in videoStream:
            yield (video[0], video[1], video[2] / 255)

    def batch_flow(self, flow):
        while (True):
            batch = np.zeros((self.batchsize, self.max_length * self.framerate, self.video_size, self.video_size, 3))
            labels = np.zeros((self.batchsize, 1))
            for i in range(self.batchsize):
                video = next(flow)
                batch[i] = video[2]
                labels[i] = video[0]
            yield (batch, labels)

    def video_stream_with_loading(self, validation_data=False):
        return self.adorn_video( self.video_stream(validation_data=validation_data) )

    def video_stream_with_augmentation(self, validation_data=False):
        return self.augmentation( self.video_stream_with_loading(validation_data) )

    # no augmentation
    def video_stream_with_zeropad(self, validation_data=False):
        return self.zero_pad_videos( self.normalize_videos( self.video_stream_with_loading(validation_data) ) )

    # no augmentation
    def video_stream_with_zeropad_batch(self, validation_data=False):
        return self.batch_flow( self.video_stream_with_zeropad(validation_data) )

    def video_stream_with_augmentation_zeropad(self, validation_data=False):
        return self.zero_pad_videos( self.video_stream_with_augmentation(validation_data) )

    def video_stream_with_augmentation_zeropad_batch(self, validation_data=False):
        return self.batch_flow( self.video_stream_with_augmentation_zeropad(validation_data) )

    def get_numpy_filename(self, path):
        return path.replace('.mp4', 'sq%s_fp%d_vs%d.npy' % \
                                              ('1' if self.squarecrop else '0', self.framerate, self.video_size))

    def precompute_numpy_video_files(self):
        """
        Compute numpy files for the videos given the class parameters, note that this is incorporated into the
        file name. Doing this *significantly* improves the performance of training.
        """
        videos = self.get_videos(self.not_collisions, 1) \
                  | chain_with(self.get_videos(self.collisions, 0)) \
                  | where(lambda f: not isfile(self.get_numpy_filename(f[1])))

        for v in videos:
            path = self.get_numpy_filename(v[1])

            video_to_npy(v[1],
                         # note weird thing here, width doesn't work they appear to be inverted
                         height=self.video_size,
                         squarecrop=self.squarecrop,
                         fps=self.framerate,
                         maxlength=self.max_length,
                         # save a npy replacement
                         outfile=path)

            print('%s written' % (path))

