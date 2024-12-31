import cv2
import matplotlib.pyplot as plt
import json
import numpy as np


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def plot_ransac_revised(segment_data_x, segment_data_y):
    from skimage.measure import LineModelND, ransac
    data = np.column_stack([segment_data_x, segment_data_y])

    # fit line using all data
    model = LineModelND()
    model.estimate(data)

    # robustly fit line only using inlier data with RANSAC algorithm
    model_robust, inliers = ransac(data, LineModelND, min_samples=2,
                                   residual_threshold=10, max_trials=1000)
    outliers = inliers == False

    # generate coordinates of estimated models
    line_x = np.array([segment_data_x.min(), segment_data_x.max()])
    line_y = model.predict_y(line_x)
    line_y_robust = model_robust.predict_y(line_x)
    k = (line_y_robust[1] - line_y_robust[0]) / (line_x[1] - line_x[0])
    m = line_y_robust[0] - k * line_x[0]
    x0 = (segment_data_y.min() - m) / k
    x1 = (segment_data_y.max() - m) / k
    line_x_y = np.array([x0, x1])
    line_y_robust_y = model_robust.predict_y(line_x_y)
    if (distance(line_x[0], line_y_robust[0], line_x[1], line_y_robust[1]) <
            distance(line_x_y[0], line_y_robust_y[0], line_x_y[1], line_y_robust_y[1])):
        # plt.plot(line_x, line_y_robust, '-b', label='Robust line model')
        line_twopoint = (line_x, line_y_robust)
    else:
        # plt.plot(line_x_y, line_y_robust_y, '-b', label='Robust line model')
        line_twopoint = (line_x_y, line_y_robust_y)

    return inliers, outliers, line_twopoint


def line_intersection(line1, line2, x_min, x_max, y_min, y_max):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    # 범위 내의 값인지 체크
    if x_min - 100 <= x <= x_max + 100 and y_min - 100 <= y <= y_max + 100:
        return x, y
    else:
        return -12345, -12345


def coord_sort_n(x):
    x = np.array(x)
    k = x[:, 0]
    s = k.argsort()
    centers_sorted = x[s]
    for i in range(len(centers_sorted) // 2):
        b = centers_sorted[2 * i:2 * (i + 1), :]
        k = b[:, 1]
        s = k.argsort()
        centers_sorted[2 * i:2 * (i + 1), :] = b[s]
    return centers_sorted


def homography(img, points, width, height):
    pts1 = np.float32(coord_sort_n(points))
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # img = img_to_coord(img)
    img = cv2.warpPerspective(img, matrix, (width, height))
    return img


def cluster(point_data, n_cluster=8):
    '''
    <input>
    point_data: 중심점을 구할 포인트 집합 ( np.array([[x1,y1], [x2,y2], [x3,y3]....]]) )
    n_cluster: 클러스터링할 그룹의 수

    <output>
    centers: 중심점 집합 ( np.array([[x1,y1], [x2,y2], [x3,y3]....]]) )
    '''
    model = KMeans(n_clusters=n_cluster)
    model.fit(point_data)
    predict = model.predict(point_data)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(set(predict))))
    k_means_labels = model.labels_
    k_means_cluster_centers = model.cluster_centers_
    for k, col in zip(range(n_cluster), colors):
        my_members = (k_means_labels == k)

        # 중심 정의
        cluster_center = k_means_cluster_centers[k]

    centers = np.array(k_means_cluster_centers, dtype=int)
    return centers

def ribbon_gap(ribbon_homo, axis):
    if axis == "0":
        y_diff = []
        y_cut_ribbon = ribbon_homo[:, 0:int(ribbon_homo.shape[1] / 2)]
        for y in range(y_cut_ribbon.shape[0]):
            y_cut_idx = np.where(y_cut_ribbon[y, :] > 0)
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff.append(y_max_value - y_min_value)
        y_cut_ribbon = ribbon_homo[:, int(ribbon_homo.shape[1] / 2):]
        for y in range(y_cut_ribbon.shape[0]):
            y_cut_idx = np.where(y_cut_ribbon[y, :] > 0)
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff.append(y_max_value - y_min_value)
    elif axis == "1":
        y_diff = []
        y_cut_ribbon = ribbon_homo[0:int(ribbon_homo.shape[0] / 2), :]
        for x in range(y_cut_ribbon.shape[1]):
            y_cut_idx = np.where(y_cut_ribbon[:, x] > 0)
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff.append(y_max_value - y_min_value)
        y_cut_ribbon = ribbon_homo[int(ribbon_homo.shape[0] / 2):, :]
        for x in range(y_cut_ribbon.shape[1]):
            y_cut_idx = np.where(y_cut_ribbon[:, x] > 0)
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff.append(y_max_value - y_min_value)
    y_diff = np.array(y_diff)
    # print(y_diff)
    # print(np.min(y_diff))

    return y_diff


def plot_points_with_labels(points):
    for i, point in enumerate(points):
        plt.text(point[0], point[1], str(i), color='white')


def draw_dab_distances(centers):
    centers = coord_sort_z(centers)
    centers_x = centers[:, 0]
    centers_y = centers[:, 1]
    distances_vert = []
    distances_horiz = []
    a = 1
    for k in range(0, 8, 2):  # 수정된 범위 (0부터 8까지, 간격은 2)
        dist_vert = calculate_distance(centers[k], centers[k + 1])
        distances_vert.append(round(dist_vert * a, 2))
        plt.plot((centers_x[k], centers_x[k + 1]), (centers_y[k], centers_y[k + 1]), marker="o", linestyle=":")
        plt.text((centers_x[k] + centers_x[k + 1]) / 2, (centers_y[k] + centers_y[k + 1]) / 2, distances_vert[-1],
                 color="white")

    for k in range(0, 6):  # 수정된 범위 (0부터 6까지)
        dist_horiz = calculate_distance(centers[k], centers[k + 2])
        distances_horiz.append(round(dist_horiz * a, 2))
        plt.plot((centers_x[k], centers_x[k + 2]), (centers_y[k], centers_y[k + 2]), marker="s", linestyle="--")
        plt.text((centers_x[k] + centers_x[k + 2]) / 2, (centers_y[k] + centers_y[k + 2]) / 2, distances_horiz[-1],
                 color="white")
    return distances_horiz, distances_vert


def coord_sort_z(x):
    # Convert input to a NumPy array
    x = np.array(x)

    # Sort by y-coordinate
    k = x[:, 1]  # Selecting the second column (y-coordinate)
    s = k.argsort()
    centers_sorted = x[s]

    # Within each pair, sort by x-coordinate
    for i in range(len(centers_sorted) // 2):
        b = centers_sorted[2 * i:2 * (i + 1), :]
        k = b[:, 0]  # Selecting the first column (x-coordinate)
        s = k.argsort()
        centers_sorted[2 * i:2 * (i + 1), :] = b[s]

    return centers_sorted
def is_clockwise(contour):
    value = 0
    num = len(contour)
    for i, point in enumerate(contour):
        p1 = contour[i]
        if i < num - 1:
            p2 = contour[i + 1]
        else:
            p2 = contour[0]
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1]);
    return value < 0

def get_merge_point_idx(contour1, contour2):
    idx1 = 0
    idx2 = 0
    distance_min = -1
    for i, p1 in enumerate(contour1):
        for j, p2 in enumerate(contour2):
            distance = pow(p2[0][0] - p1[0][0], 2) + pow(p2[0][1] - p1[0][1], 2);
            if distance_min < 0:
                distance_min = distance
                idx1 = i
                idx2 = j
            elif distance < distance_min:
                distance_min = distance
                idx1 = i
                idx2 = j
    return idx1, idx2

def merge_contours(contour1, contour2, idx1, idx2):
    contour = []
    for i in list(range(0, idx1 + 1)):
        contour.append(contour1[i])
    for i in list(range(idx2, len(contour2))):
        contour.append(contour2[i])
    for i in list(range(0, idx2 + 1)):
        contour.append(contour2[i])
    for i in list(range(idx1, len(contour1))):
        contour.append(contour1[i])
    contour = np.array(contour)
    return contour

def merge_with_parent(contour_parent, contour):
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    return merge_contours(contour_parent, contour, idx1, idx2)





with open("GT.json", "r") as file:
    raw_data = json.load(file)
dir = "../{}".format(raw_data['key'])
img = cv2.imread(dir)

print(raw_data['key'])

image = cv2.imread("../{}".format(raw_data['key']))
scratch = np.zeros((raw_data['height'], raw_data['width'], 3), dtype=np.uint8)

ribbon_1 = np.array(raw_data['boxes'][0]['points'])
ribbon_2 = np.array(raw_data['boxes'][1]['points'])

contour = merge_contours(ribbon_1, ribbon_2, 0, 0)
contour.astype(np.int32)
GT_total = cv2.polylines(scratch, np.int32([contour]), isClosed=True, color=(0,0,255))
cv2.fillPoly(GT_total, np.int32([contour]), (255,255,255)) # 안에 색칠
for k in range(2, len(raw_data['boxes'])):
    points = np.array(raw_data['boxes'][k]['points'])
    cv2.polylines(GT_total, np.int32([points]), isClosed=True, color=(0,0,255))
    cv2.fillPoly(GT_total, np.int32([points]), (255,255,255))
# plt.imshow(GT_total)
# plt.show()

GT_dab = np.zeros((raw_data['height'], raw_data['width'], 3), dtype=np.uint8)
for k in range(2, 10):
    points = np.array(raw_data['boxes'][k]['points'], dtype=np.int32)
    points = points.reshape((-1, 1, 2))  # 점들을 (N, 1, 2) 형식으로 변환합니다.
    cv2.polylines(GT_dab, [points], isClosed=True, color=(0, 0, 255))
    cv2.fillPoly(GT_dab, [points], color=(255, 255, 255))
# plt.imshow(GT_total)
# plt.show()

GT_ribbon = GT_total - GT_dab

