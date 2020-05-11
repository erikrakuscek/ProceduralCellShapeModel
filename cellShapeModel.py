import numpy as np
import cv2
import czifile
import pickle


def num_red_pixels(image):
    return image[(image == [255, 0, 0])]


# [seedX, seedY, layerFrom, layerTo, maxSize]
cells = [[1120, 600, 19, 53, 40000]]

# image = czifile.imread('CAAX_100X_20171024_1-Scene-03-P3-B02.czi')

# with open('cells.pickle', 'wb+') as f:
#    pickle.dump(image[0][0][3], f)

with open('cells.pickle', 'rb+') as f:
    images = pickle.load(f)

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

# TODO: init random seed
# TODO: init vektor landmarkov
landmarks = []

for cell in cells:
    seedX = cell[0]
    seedY = cell[1]
    layerFrom = cell[2]
    layerTo = cell[3]
    max_red = cell[4]
    for i in range(layerFrom, layerTo - 20):
        image = images[i]
        image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

        seed = (seedX, seedY)
        cv2.floodFill(image, None, seedPoint=seed, newVal=(0, 0, 255), loDiff=(3, 3, 3, 3), upDiff=(255, 255, 255, 255))
        cv2.circle(image, seed, 2, (0, 255, 0), cv2.FILLED, cv2.LINE_AA)

        # kernel = np.ones((5, 5), np.float32) / 25
        # image = cv2.filter2D(image, -1, kernel)

        reds = num_red_pixels(image)
        print(i, reds.size)

        # Glajenje slike
        kernel = np.ones((5, 5), np.uint8)
        image = cv2.dilate(image, kernel, iterations=3)

        if reds.size < max_red:
            bg = image[:, :, 0] == image[:, :, 1]  # B == G
            gr = image[:, :, 1] == image[:, :, 2]  # G == R
            image = np.bitwise_and(bg, gr, dtype=np.uint8) * 255

            image = cv2.Canny(image, 5, 200)

            a = np.argwhere(image == 255)

            cv2.imshow('layer ' + str(i), image)

        # TODO: določimo landmarke

        # TODO: poravnaj landmarke v (0, 0)

        # TODO: dodaj landmarke v vekor landmarkov

# TODO: izračunaj povprečno obliko celice

# TODO: izračunaj eigenvalues in eigenvectors

# TODO: generate new cells WOOHOO!

cv2.waitKey(0)
cv2.destroyAllWindows()
