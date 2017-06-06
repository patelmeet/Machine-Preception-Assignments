import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("girl.jpg",cv2.IMREAD_COLOR)     #read color image

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) #convert BGR to Grayscale image

equ = cv2.equalizeHist(gray_image)      #apply histogram equalization on grayscale image

mean = np.mean(gray_image)              #get mean of image
std = np.std(gray_image)                #get standard deviation of image
whitned_image = (gray_image-mean)/std   #apply whitening to grayscale image
#whitned_image can have -ve values also
#apply scalling to get all values in range 0..1
min = np.min(whitned_image)
whitned_image = whitned_image - min
max = np.max(whitned_image)
whitned_image = whitned_image / max

plt.subplot(1,3,1),plt.imshow(gray_image,'gray'),plt.title('GrayScale Image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(whitned_image,'gray'),plt.title('Whitened Image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(equ,'gray'),plt.title('Histogram Equalized Image'),plt.xticks([]), plt.yticks([])

plt.show()
