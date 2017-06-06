import cv2
import numpy as np
from matplotlib import pyplot as plt

gray_image = cv2.imread("fig.tif",cv2.IMREAD_GRAYSCALE)   #read gray-scale image

#create and apply laplacian mask to gray_image to get laplacian edges
mask1 = np.array([[0,1,0],[1,-4,1],[0,1,0]])
laplacian_edges = cv2.filter2D(gray_image,-1,mask1)

#substract laplacian edges from gray_image to get sharp image
sharped_image = gray_image - laplacian_edges

#put gray_image's value where values are negative in sharp image
locations = np.where(gray_image<laplacian_edges)
for pt in zip(*locations):
    sharped_image.itemset((pt[0],pt[1]),gray_image[pt[0]][pt[1]])

plt.subplot(1,3,1),plt.imshow(gray_image,'gray'),plt.title('gray image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(laplacian_edges,'gray'),plt.title('Laplacian image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(sharped_image,'gray'),plt.title('sharped image'),plt.xticks([]), plt.yticks([])

plt.show()
