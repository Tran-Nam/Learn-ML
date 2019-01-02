import numpy as np 
import imutils
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, 
                help='Path to image')

args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

# translation
M = np.float32([[1, 0, -50], [0, 1, -90]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Shifted up and left', shifted)
cv2.waitKey(0)

M = np.float32([[1, 0, 25], [0, 1, 50]])
shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
cv2.imshow('Shifted Down and Right', shifted)
cv2.waitKey(0)

shifted = imutils.translate(image, 0, 100)
cv2.imshow('Shifted down', shifted)
cv2.waitKey(0)

# rotate
(h, w) = image.shape[:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, angle=90, scale=1.0)
rotated = cv2.warpAffine(image, M, (w, h))
cv2.imshow('Rotate 90', rotated)
cv2.waitKey(0)

rotated = imutils.rotate(image, 180)
cv2.imshow('Rotate 180', rotated)
cv2.waitKey(0)

# resize
r = 150. / image.shape[1]
dim = (150, int(image.shape[0]*r))
resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
print(resized.shape)
print(image.shape)
cv2.imshow('Resize width', resized)
cv2.waitKey(0)

# flipping
flipped = cv2.flip(image, 1)
cv2.imshow('Flipped Horizontally', flipped)
cv2.waitKey(0)

flipped = cv2.flip(image, 0)
cv2.imshow('Flipped Vertically', flipped)
cv2.waitKey(0)

flipped = cv2.flip(image, -1)
cv2.imshow('Flipped', flipped)
cv2.waitKey(0)


