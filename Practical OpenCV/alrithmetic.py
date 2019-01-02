import numpy as np 
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

# print('max of 255: {}'.format(cv2.add(np.uint8([200]), np.uint8([100]))))
# print('wrap around: {}'.format(np.uint8([200]) + np.uint8([100])))

M = np.ones(image.shape, dtype='uint8') * 100
added = cv2.add(image, M)
cv2.imshow('Added', added)
cv2.waitKey(0)

M = np.ones(image.shape, dtype='uint8') * 50
subed = cv2.subtract(image, M)
cv2.imshow('Subed', subed)
cv2.waitKey(0)