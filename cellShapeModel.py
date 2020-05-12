import numpy as np
import cv2
import czifile
import pickle
from numpy import linalg as LA
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import Rbf


def num_red_pixels(img):
    return img[(img == [255, 0, 0])]


LANDMARKS_PER_LAYER = 15
LAYER_HEIGHT = 5
# [seedX, seedY, layerFrom, layerTo, minSize, maxSize]
cells = [[1120, 600, 19, 53, LANDMARKS_PER_LAYER, 40000],
         [340, 670, 19, 53, LANDMARKS_PER_LAYER, 40000],
         [460, 920, 20, 54, LANDMARKS_PER_LAYER, 35000],
         [1010, 610, 19, 53, LANDMARKS_PER_LAYER, 35000],
         [540, 515, 17, 51, LANDMARKS_PER_LAYER, 40000],
         [1400, 1180, 19, 53, LANDMARKS_PER_LAYER, 40000]]
# cells = [[540, 515, 17, 51, LANDMARKS_PER_LAYER, 40000]]

# image = czifile.imread('CAAX_100X_20171024_1-Scene-03-P3-B02.czi')

# with open('cells.pickle', 'wb+') as f:
#    pickle.dump(image[0][0][3], f)


if None:

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
    cells_landmarks = []

    for cell in cells:
        layerFrom = cell[2]
        layerTo = cell[3]
        min_red = cell[4]
        max_red = cell[5]

        b = 0
        landmarks = []
        prev_image = images[0]
        prev_edge = np.zeros((LANDMARKS_PER_LAYER, 2))
        z = 0
        for i in range(layerFrom, layerTo):
            image = images[i]
            image = cv2.normalize(image, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            image = cv2.convertScaleAbs(image, alpha=(255.0 / 65535.0))

            seed = (cell[0], cell[1])
            cv2.floodFill(image, None, seedPoint=seed, newVal=(0, 0, 255), loDiff=(3, 3, 3, 3), upDiff=(255, 255, 255, 255))

            reds = num_red_pixels(image)
            cv2.imwrite('newImages/' + str(i) + '.png', image)

            edge = []
            if max_red > reds.size > min_red:
                # Glajenje slike
                kernel = np.ones((5, 5), np.uint8)
                image = cv2.dilate(image, kernel, iterations=4)

                bg = image[:, :, 0] == image[:, :, 1]  # B == G
                gr = image[:, :, 1] == image[:, :, 2]  # G == R
                image = np.bitwise_and(bg, gr, dtype=np.uint8) * 255

                # poiščemo samo rob clice
                image = cv2.Canny(image, 5, 200)

                # pridobimo vse točke na robu
                edge = np.argwhere(image == 255)

                # cv2.imshow('i ' + str(i), image)
                if edge.shape[0] < LANDMARKS_PER_LAYER:
                    # ce je slika neuporabna vzamemo prejsnjo
                    edge = prev_edge
                    image = prev_image
            else:
                # ce je slika neuporabna vzamemo prejsnjo
                b += 1
                edge = prev_edge
                image = prev_image
                print("too big or too small ", b)

            # vzamemo naključnih LANDMARKS_PER_LAYER točk na robu, jih poravnamo v središče koordinatnega sistema
            # in jih dodamo v landmarks
            i_landmarks = np.random.choice(np.arange(edge.shape[0]), LANDMARKS_PER_LAYER, replace=False)
            i_landmarks.sort()
            a = 0
            for j in i_landmarks:
                landmarks.append(edge[j][1] - seed[0])
                landmarks.append(edge[j][0] - seed[1])
                landmarks.append(z)

                cv2.circle(image, (int(edge[j][1]), int(edge[j][0])), 4, (2 * a, 0, 0), cv2.FILLED, cv2.LINE_AA)
                a += 1

            # cv2.imshow('layer ' + str(i), image)

            prev_image = image
            prev_edge = edge
            z += LAYER_HEIGHT

        cells_landmarks.append(landmarks)

    # TODO: izračunaj povprečno obliko celice
    cells_landmarks = np.asarray(cells_landmarks)
    avg = np.sum(cells_landmarks, axis=0) / cells_landmarks.shape[0]

    # TODO: izračunaj eigenvalues in eigenvectors
    c_matrix = np.zeros((cells_landmarks.shape[1], cells_landmarks.shape[1]))
    for x in cells_landmarks:
        c_matrix += np.outer((x - avg), (x - avg))
    c_matrix = c_matrix / (len(cells_landmarks) - 1)

    eigenvalues, eigenvectors = LA.eig(c_matrix)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenvalues = np.asarray(list(map(lambda y: y.real, eigenvalues)))
    eigenvectors = np.asarray(list(map(lambda y: y.real, eigenvectors)))

    eigen_matrix = np.zeros((cells_landmarks.shape[1], cells_landmarks.shape[1]))
    for i in range(cells_landmarks.shape[1]):
        eigen_matrix[i] = eigenvectors[:, i]


    with open('eigenvalues.pickle', 'wb+') as f:
        pickle.dump(eigenvalues, f)
    with open('eigenvectors.pickle', 'wb+') as f:
        pickle.dump(eigenvectors, f)
    with open('avg.pickle', 'wb+') as f:
        pickle.dump(avg, f)

with open('eigenvalues.pickle', 'rb+') as f:
    eigenvalues = pickle.load(f)
with open('eigenvectors.pickle', 'rb+') as f:
    eigenvectors = pickle.load(f)
with open('avg.pickle', 'rb+') as f:
    avg = pickle.load(f)



# TODO: generate new cells WOOHOO!
# weights = np.ones(cells_landmarks.shape[1])
# for i in range(weights.size):
#     weights[i] *= np.sign(eigenvalues[0]) * random.random() * 3 * math.sqrt(abs(eigenvalues[0].real))

x_kaca = avg + (random.random() * 2 * math.sqrt(abs(eigenvalues[0])) * eigenvectors[0])
x_kaca2 = avg + (random.random() * 2 * math.sqrt(abs(eigenvalues[0])) * eigenvectors[0])

fig = plt.figure()
ax = Axes3D(fig)
ax.set_xlim(-300, 300)
ax.set_ylim(-300, 300)
ax.set_zlim(-300, 300)

new_cell_x = x_kaca2[0::3]
new_cell_y = x_kaca2[1::3]
new_cell_z = x_kaca2[2::3]
ax.scatter(new_cell_x, new_cell_y, new_cell_z, c='b', s=1)
new_cell_x = x_kaca[0::3]
new_cell_y = x_kaca[1::3]
new_cell_z = x_kaca[2::3]
ax.scatter(new_cell_x, new_cell_y, new_cell_z, c='r', s=1)
plt.show()

new_cell_x = np.asarray(list(map(lambda y: int(y), new_cell_x)))
new_cell_y = np.asarray(list(map(lambda y: int(y), new_cell_y)))
new_cell_z = np.asarray(list(map(lambda y: int(y), new_cell_z)))

idx = np.argsort(new_cell_z)
new_cell_z = np.array(new_cell_z)[idx]
new_cell_y = np.array(new_cell_y)[idx]
new_cell_x = np.array(new_cell_x)[idx]

print(new_cell_x)
print(new_cell_y)
print(new_cell_z)

x, y, z, d = np.random.rand(4, 50)
rbfi = Rbf(new_cell_x, new_cell_y, new_cell_z, np.ones(new_cell_z.size))  # radial basis function interpolator instance
xi = yi = zi = np.linspace(0, 1, 20)
di = rbfi(xi, yi, zi)   # interpolated values
print(di.shape)

# grid = np.zeros((new_cell_x.size, new_cell_y.size, new_cell_z.size))
# cur_z = 0
# with open('cell.raw', 'ab+') as f:
#     for z in range(-300, 300):
#         if z == cur_z:
#             for y in range(-300, 300):
#                 for x in range(-300, 300):
#                     appearances_z = [i for i, e in new_cell_z if e == z]
#                     for appearance in appearances_z:
#                         if new_cell_x[appearance] = x and new_cell_y[appearance] = y:
#
#                     if new_cell_z[i]:
#
#                     else if:
#
#                     else:
#                         f.write(\x00)
#         else:

# cv2.waitKey(0)
# cv2.destroyAllWindows()
