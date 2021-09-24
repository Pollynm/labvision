import sys

from labvision import camera
from sh import gphoto2
import time
import subprocess
import os
import cv2
#
# cam1 = camera.Panasonic()
# cam1.take_frame()
# filename = cam1.save_file_onto_computer()
# print(filename)


cam2 = camera.Panasonic(mode='Movie')
cam2.delete_multiple_files_from_camera()
cam2.start_movie(duration=3)
cam2.save_file_onto_computer()