import numpy as np
import matplotlib.pyplot as plt
from skimage import data, filters
from skimage.segmentation import flood, flood_fill
from skimage import filters
import cv2

output = []
image = cv2.imread('cells1png.png')

# image = cv2.bitwise_not(image)

# image = cv2.equalizeHist(image)

# image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

# Create kernel
# kernel = np.array([[0, -1, 0],
#                   [-1, 5, -1],
#                   [0, -1, 0]])

# Sharpen image
# image = cv2.filter2D(image, -1, kernel)

# cv2.imshow('fastNlMeansDenoisingColored', image)

# image = cv2.Canny(image, 5, 200)

seed = (1100, 550)

cv2.floodFill(image, None, seedPoint=seed, newVal=(0, 0, 255), loDiff=(2, 2, 2, 2), upDiff=(5, 5, 5, 5))
cv2.circle(image, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

# kernel = np.ones((5,5),np.float32)/25
# image = cv2.filter2D(image, -1, kernel)

kernel = np.ones((5, 5), np.uint8)
image = cv2.dilate(image, kernel, iterations=3)

cv2.imshow('flood', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
