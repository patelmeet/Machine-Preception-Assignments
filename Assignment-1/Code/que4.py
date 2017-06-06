import cv2 #First we import the cv2 module
from matplotlib import pyplot as plt

image = cv2.imread("1.jpg",cv2.IMREAD_COLOR)    #Read the image to a variable named “image”

gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)  #Convert to gray-scale image and store it to another variable named “gray_image”

#“gray_image” will hold the gray - scale version of the input image.

#To display the original and the gray-scale we use function “cv2.imshow()” with parameters as the “window title” and the “image variable”
plt.subplot(1,2,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),'gray'),plt.title('Original Image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2),plt.imshow(gray_image,'gray'),plt.title('Grayscale Image'),plt.xticks([]), plt.yticks([])
plt.show()

