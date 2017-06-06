import cv2
import matplotlib.pyplot as plt #Collection of command style functions that make matplotlib work like MATLAB

image = cv2.imread("rgb.png",cv2.IMREAD_COLOR) #Loads a color image. Any transparency of image will be neglected.
blue_channel = image[:,:,0]     #extract first BLUE plane
green_channel = image[:,:,1]    #extract second GREEN plane
red_channel = image[:,:,2]      #extract third RED plane

# Plot original image as color image and other three channel as grayscale.
# Matplotlib takes image as RGB and openCV as BGR so before passing to matplotlib, we have to convert image to RGB.
plt.subplot(2,2,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(blue_channel,'gray'),plt.title('Blue Channel'),plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3),plt.imshow(green_channel,'gray'),plt.title('Green Channel'),plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4),plt.imshow(red_channel,'gray'),plt.title('Red Channel'),plt.xticks([]), plt.yticks([])

plt.show() #Display the current figure that you are working on