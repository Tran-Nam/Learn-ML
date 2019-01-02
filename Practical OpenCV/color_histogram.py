import numpy as np 
import matplotlib.pyplot as plt 
import argparse 
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help='Path to image')
args = vars(ap.parse_args())

image = cv2.imread(args['image'])
cv2.imshow('Original', image)
cv2.waitKey(0)

chans = cv2.split(image)
colors = ('b', 'g', 'r')
plt.figure()
plt.title('Flatten Color Histogram')
plt.xlabel('Bin')
plt.ylabel('# of pixels')

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
# plt.show()

fig = plt.figure()

ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1],
                    None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and B')
plt.colorbar(p)

hist = cv2.calcHist([chans[1], chans[2]], [0, 1],
                    None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for G and R')
plt.colorbar(p)

hist = cv2.calcHist([chans[0], chans[2]], [0, 1],
                    None, [32, 32], [0, 256, 0, 256])
p = ax.imshow(hist, interpolation='nearest')
ax.set_title('2D Color Histogram for B and R')
plt.colorbar(p)