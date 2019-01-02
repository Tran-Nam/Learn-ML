from imutils.video import VideoStream
import argparse
import datetime
import imutils
import time
import cv2 

ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', \
                help='Path to video')
ap.add_argument('-a', '--min_area', type=int, \
                default=500, help='Minimum area size')
args = vars(ap.parse_args())

# if argument is none, then we are reading from webcam
if args.get('video', None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

# otherwise, we wre reading from a video file
else:
    vs = cv2.VideoCapture(args['video'])

# initialize the first frame in the video stream
firstFrame = None
