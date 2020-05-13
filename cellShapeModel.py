import numpy as np
import cv2
import czifile
import pickle
from numpy import linalg as LA
import random
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def num_red_pixels(img):
    return img[(img == [255, 0, 0])]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def clockwiseangle_and_distance(point):
    refvec = [0, 1]
    origin = [0, 0]

    # Vector between point and the origin: v = p - o
    vector = [point[0] - origin[0], point[1] - origin[1]]
    # Length of vector: ||v||
    lenvector = math.hypot(vector[0], vector[1])
    # If length is zero there is no angle
    if lenvector == 0:
        return -math.pi, 0
    # Normalize vector: v/||v||
    normalized = [vector[0] / lenvector, vector[1] / lenvector]
    dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
    diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
    angle = math.atan2(diffprod, dotprod)
    # Negative angles represent counter-clockwise angles so we need to subtract them
    # from 2*pi (360 degrees)
    if angle < 0:
        return 2 * math.pi + angle, lenvector
    # I return first the angle because that's the primary sorting criterium
    # but if two vectors have the same angle then the shorter distance should come first.
    return angle, lenvector


LANDMARKS_PER_LAYER = 15
LAYER_HEIGHT = 1
NUM_LAYERS = 34

# [seedX, seedY, layerFrom, layerTo, minSize, maxSize]
cells = [[1120, 600, 19, 53, LANDMARKS_PER_LAYER, 40000],
         [340, 670, 19, 53, LANDMARKS_PER_LAYER, 40000],
         [460, 920, 20, 54, LANDMARKS_PER_LAYER, 35000],
         [1010, 610, 19, 53, LANDMARKS_PER_LAYER, 35000],
         [540, 515, 17, 51, LANDMARKS_PER_LAYER, 40000],
         [1400, 1180, 19, 53, LANDMARKS_PER_LAYER, 40000]]

# image = czifile.imread('CAAX_100X_20171024_1-Scene-03-P3-B02.czi')

# with open('cells.pickle', 'wb+') as f:
#    pickle.dump(image[0][0][3], f)

if True:

    with open('cells.pickle', 'rb+') as f:
        images = pickle.load(f)

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
            cv2.floodFill(image, None, seedPoint=seed, newVal=(0, 0, 255), loDiff=(3, 3, 3, 3),
                          upDiff=(255, 255, 255, 255))

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

            cell_landmarks = []
            for j in i_landmarks:
                cell_landmarks.append(edge[j][1] - seed[0])
                cell_landmarks.append(edge[j][0] - seed[1])
                cell_landmarks.append(z)

            cell_landmarks = np.asarray(list(chunks(cell_landmarks, 3)))
            cell_landmarks = sorted(cell_landmarks, key=clockwiseangle_and_distance)
            cell_landmarks = np.asarray(cell_landmarks).flatten().tolist()

            for a in range(LAYER_HEIGHT):
                landmarks += cell_landmarks
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

random.seed(123123)
new_cell = avg + (random.random() * 2 * math.sqrt(abs(eigenvalues[0])) * eigenvectors[0])

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.set_xlim(-300, 300)
# ax.set_ylim(-300, 300)
# ax.set_zlim(-300, 300)
#
new_cell_x = new_cell[0::3]
new_cell_y = new_cell[1::3]
new_cell_z = new_cell[2::3]
# ax.scatter(new_cell_x, new_cell_y, new_cell_z, c='r', s=1)
# plt.show()

new_cell_x = np.asarray(list(map(lambda e: int(e), new_cell_x)))
new_cell_y = np.asarray(list(map(lambda e: int(e), new_cell_y)))
new_cell_z = np.asarray(list(map(lambda e: int(e), new_cell_z)))

idx = np.argsort(new_cell_z)
new_cell_z = np.array(new_cell_z)[idx]
new_cell_y = np.array(new_cell_y)[idx]
new_cell_x = np.array(new_cell_x)[idx]

layers = np.asarray(list(chunks(new_cell, 3 * LANDMARKS_PER_LAYER)))
print(layers)

inside = 0
outside = 0
on_edge = 0
print(len(layers))
with open('cell.raw', 'ab+') as f:
    for z in range(len(layers)):
        layers[z][2::3] = z
        points = list(map(tuple, np.asarray(list(chunks(layers[z], 3)))))
        points = np.asarray(list(map(lambda t: tuple(map(int, t)), points)))

        points_clockwise = sorted(points, key=clockwiseangle_and_distance)
        points_clockwise = np.asarray(points_clockwise).tolist()
        print(points_clockwise)

        poly = Polygon(points_clockwise)
        # p = points_clockwise
        # p.append(points_clockwise[0])  # repeat the first point to create a 'closed loop'
        # xs, ys, zs = zip(*p)  # create lists of x and y values
        # plt.figure()
        # plt.plot(xs, ys, zs)
        # plt.show()

        # p = np.asarray([[item[0], item[1]] for item in points])
        # hull = ConvexHull(p)
        # plt.plot(p[:, 0], p[:, 1], 'o')
        # for simplex in hull.simplices:
        #     plt.plot(p[simplex, 0], p[simplex, 1], 'k-')
        # plt.plot(p[hull.vertices, 0], p[hull.vertices, 1], 'r--', lw=2)
        # plt.plot(p[hull.vertices[0], 0], p[hull.vertices[0], 1], 'ro')
        # plt.show()

        output = []
        for y in range(-200, 200):
            for x in range(-200, 200):
                if any(np.equal(points_clockwise, [x, y, z]).all(1)):
                    # print(np.equal(points, [x, y, cur_z]).all(1))
                    # print([x, y, cur_z])
                    on_edge += 1
                    output.append(128)
                elif poly.contains(Point(x, y, z)):
                    inside += 1
                    #plt.scatter(x, y, z, c='r')
                    output.append(255)
                else:
                    outside += 1
                    output.append(0)

        print(on_edge)
        print(inside)
        print(outside)
        f.write(bytearray(output))

# cv2.waitKey(0)
# cv2.destroyAllWindows()
