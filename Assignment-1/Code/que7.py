import cv2
import numpy as np
from matplotlib import pyplot as plt

def add_salt_and_pepper_noise(input_image , salt_prob , pepper_prob):
    #make random_matrix of size,same as image with values between 0 and 1.
    random_matrix = np.random.rand(input_image.shape[0], input_image.shape[1])
    output_image = np.copy(input_image)         #copy input image
    output_image[random_matrix > 1- salt_prob] = 255    #add salt noise
    output_image[random_matrix < pepper_prob] = 0       #add pepper noise
    return output_image

image = cv2.imread("messy.jpg",cv2.IMREAD_COLOR)    #read color image
gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)     #convert color image to gray-scale image

#add salt and pepper noise to gray_image with probability of 0.008 for both salt and pepper
noisy_image = add_salt_and_pepper_noise(gray_image,0.008,0.008)

#apply median filtering with kernal size=3x3 to remove noise
median = cv2.medianBlur(noisy_image,3)

plt.subplot(1,3,1),plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),'gray'),plt.title('Original image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,2),plt.imshow(noisy_image,'gray'),plt.title('Salt and pepper noisy image'),plt.xticks([]), plt.yticks([])
plt.subplot(1,3,3),plt.imshow(median,'gray'),plt.title('Median filtered image'),plt.xticks([]), plt.yticks([])
plt.show()



