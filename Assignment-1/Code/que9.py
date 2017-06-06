import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("rect1.png",cv2.IMREAD_GRAYSCALE)    #read gray-scale image

#define robert masks of size 2x2
mask_45 = np.array([[1,0],[0,-1]])
mask_135 = np.array([[0,1],[-1,0]])

#apply robert masks on binary image using in-built filter2D function
output_45 = cv2.filter2D(image,-1,mask_45)
output_135 = cv2.filter2D(image,-1,mask_135)

#apply thresolding to get interested lines
mask1,only_45 = cv2.threshold(output_45,190,255,cv2.THRESH_BINARY)
mask2,only_135 = cv2.threshold(output_135,190,255,cv2.THRESH_BINARY)

#To plot multiple images, we have used plt.subplot() function
plt.subplot(1,3,1),plt.imshow(image,'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(only_45,'gray'),plt.title('After thresold 45 degree'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(only_135,'gray'),plt.title('After thresold 135 degree'),plt.xticks([]), plt.yticks([])
plt.show()
