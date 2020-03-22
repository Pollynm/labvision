import os
import cv2
import numpy as np
from slicerator import Slicerator
from .. import images

__all__ = [
    'ReadVideo',
    'WriteVideo'
]

class WriteVideo:

    def __init__(
            self, 
            filename,
            frame_size=None, 
            frame=None,
            fps=50.0, 
            codec='XVID'):

        fourcc = cv2.VideoWriter_fourcc(*list(codec))

        assert (frame_size is None or frame is None) and not (frame_size is None and frame is None), "One of frame or frame_size must be supplied"

        if frame_size is None:
            self.frame_size = np.shape(frame)

        if frame is None:
            self.frame_size = frame_size

        self.vid = cv2.VideoWriter(
            filename,
            fourcc,
            fps,
            (self.frame_size[1], self.frame_size[0]))

        assert self.vid.isOpened(), 'Video failed to open'

    def add_frame(self, im):
        assert np.shape(im) == self.frame_size, "Added frame is wrong shape"
        self.vid.write(im)

    def close(self):
        self.vid.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



@Slicerator.from_class
class ReadVideo:

    def __init__(self, filename=None, grayscale=False, frame_range=(0,None,1), return_function=None):
        self.filename = filename
        self.grayscale = grayscale
        self.frame_range = frame_range
        self.return_func = return_function
        self._detect_file_type()
        self.init_video()
        self.get_vid_props()

    def close(self):
        if self.filetype == 'video':
            self.vid.release()

    def _detect_file_type(self):
        self.ext = os.path.splitext(self.filename)[1]
        if self.ext in ['.MP4', '.mp4', '.m4v', '.avi']:
            self.filetype = 'video'
        else:
            raise NotImplementedError('File extension is not implemented')

    def init_video(self):
        self.frame_num = 0
        self.vid = cv2.VideoCapture(self.filename)

    def get_vid_props(self):
        self.frame_num = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
        self.num_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_time = self.vid.get(cv2.CAP_PROP_POS_MSEC)
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if self.vid.get(cv2.CAP_PROP_MONOCHROME) == 0.0:
            self.colour = 3
            self.frame_size = (self.height, self.width, 3)
        else:
            self.colour = 1
            self.frame_size = (self.width, self.height)
        self.fps = self.vid.get(cv2.CAP_PROP_FPS)
        self.format = self.vid.get(cv2.CAP_PROP_FORMAT)
        self.codec = self.vid.get(cv2.CAP_PROP_FOURCC)

        self.file_extension = self.filename.split('.')[1]

    def read_frame(self, n=None):
        if n is None:
            return self.read_next_frame()

        assert (n >= self.frame_range[0]) & (n < self.frame_range[1]) \
                    & ((n-self.frame_range[0]) % self.frame_range[2] == 0)\
                    , 'Frame not in frame_range'

        self.set_frame(n)
        return self.read_next_frame()

    def set_frame(self, n):
        if n != self.frame_num:
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, float(n))
            self.frame_num = n

    def read_next_frame(self):
        ret, im = self.vid.read()
        self.frame_num += self.frame_range[2]
        if self.frame_range[2] != 1:
            self.set_frame(self.frame_num)

        if ret:
            if self.grayscale:
                return images.bgr_to_gray(im)
            if self.return_func:
                return self.return_func(im)
            else:
                return im
        else:
            raise Exception('Cannot read frame')

    def __getitem__(self, frame_num):
        return self.read_frame(n=frame_num)

    def __len__(self):
        return self.num_frames

    def __enter__(self):
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if
        return self.read_frame()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


