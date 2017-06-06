import cv2
import numpy as np
import matplotlib.pyplot as plt #Collection of command style functions that make matplotlib work like MATLAB

image = cv2.imread("girl.jpg",cv2.IMREAD_COLOR)     #Loads a color image. Any transparency of image will be neglected.

hsv_image = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)   #convert original BGR image to HSV image
hls_image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)   #convert original BGR image to HLS image

hue_channel = hsv_image[:,:,0]          #extract hue channel(first plane) from HSV image
saturation_channel = hsv_image[:,:,1]   #extract saturation channel
value_channel = hsv_image[:,:,2]        #extract value channel
lightness_channel = hls_image[:,:,1]    #extract lightness channel

hsl_image = np.copy(hsv_image)          #makes copy of hsv_image
hsl_image[:,:,2] = lightness_channel    #replace third plane(value) of HSV with Lightness to get HSL image

# plot original-BGR, HSV and HSL image as color image and other channels as grayscale.
plt.subplot(3,3,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,2),plt.imshow(cv2.cvtColor(hsv_image, cv2.COLOR_BGR2RGB),'gray'),plt.title('HSV Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,3),plt.imshow(cv2.cvtColor(hsl_image, cv2.COLOR_BGR2RGB),'gray'),plt.title('HSL Image'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,4),plt.imshow(hue_channel,'gray'),plt.title('Hue Channel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,5),plt.imshow(saturation_channel,'gray'),plt.title('Saturation Channel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,6),plt.imshow(value_channel,'gray'),plt.title('Value Channel'),plt.xticks([]), plt.yticks([])
plt.subplot(3,3,8),plt.imshow(lightness_channel,'gray'),plt.title('Lightness Channel'),plt.xticks([]), plt.yticks([])

plt.show() #Display the current figure that you are working on
