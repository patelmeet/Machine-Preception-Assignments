import cv2
import numpy as np
from matplotlib import pyplot as plt

binary_image = cv2.imread('binary_image2.jpg',cv2.IMREAD_GRAYSCALE)	#read gray-scale image

#define prewitt and sobel masks of size 3x3
prewittxmask = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
prewittymask = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])

sobelxmask = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
sobelymask = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

#apply all masks on binary image using in-built filter2D function
prewittx = cv2.filter2D(binary_image,-1,prewittxmask)
prewitty = cv2.filter2D(binary_image,-1,prewittymask)

sobelx = cv2.filter2D(binary_image,-1,sobelxmask)
sobely = cv2.filter2D(binary_image,-1,sobelymask)

#plot all the images
plt.subplot(3,3,2),plt.imshow(binary_image,cmap = 'gray'),plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(sobelx,cmap = 'gray'),plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,7),plt.imshow(sobely,cmap = 'gray'),plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6),plt.imshow(prewittx,cmap = 'gray'),plt.title('Prewitt X'), plt.xticks([]), plt.yticks([])
plt.subplot(3,3,9),plt.imshow(prewitty,cmap = 'gray'),plt.title('Prewitt Y'), plt.xticks([]), plt.yticks([])

plt.show()
