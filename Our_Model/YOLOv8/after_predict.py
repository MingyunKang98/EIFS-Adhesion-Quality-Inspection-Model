from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from predict import predict_dir
from predict import predicted_dir
from GT import GT_dab
from GT import GT_total
from GT import GT_ribbon
from Case_study import GT_total_homo
from Case_study import GT_dab_homo
from skimage.measure import label, regionprops
import pandas as pd


def distance(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def calculate_distance(point1, point2):
    """
    두 점 간의 유클리드 거리를 계산하는 함수입니다.

    :param point1: 첫 번째 점의 좌표 (x1, y1)
    :param point2: 두 번째 점의 좌표 (x2, y2)
    :return: 두 점 간의 거리
    """
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


def ribbon_gap_4_directions(ribbon_homo):
    directions = ["left", "right", "top", "bottom"]
    gaps = {}

    # Left direction
    y_diff_left = []
    y_cut_ribbon_left = ribbon_homo[:, 0:int(ribbon_homo.shape[1] / 2)]
    for y in range(y_cut_ribbon_left.shape[0]):
        y_cut_idx = np.where(y_cut_ribbon_left[y, :] > 0)[0]
        if y_cut_idx.size > 0:
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff_left.append([y_min_value, y_max_value, y_max_value - y_min_value])
        else:
            y_diff_left.append([0, 0, 0])
    gaps["left"] = y_diff_left

    # Right direction
    y_diff_right = []
    y_cut_ribbon_right = ribbon_homo[:, int(ribbon_homo.shape[1] / 2):]
    for y in range(y_cut_ribbon_right.shape[0]):
        y_cut_idx = np.where(y_cut_ribbon_right[y, :] > 0)[0]
        if y_cut_idx.size > 0:
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff_right.append([y_min_value, y_max_value, y_max_value - y_min_value])
        else:
            y_diff_right.append([0, 0, 0])
    gaps["right"] = y_diff_right

    # Top direction
    y_diff_top = []
    y_cut_ribbon_top = ribbon_homo[0:int(ribbon_homo.shape[0] / 2), :]
    for x in range(y_cut_ribbon_top.shape[1]):
        y_cut_idx = np.where(y_cut_ribbon_top[:, x] > 0)[0]
        if y_cut_idx.size > 0:
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff_top.append([y_min_value, y_max_value, y_max_value - y_min_value])
        else:
            y_diff_top.append([0, 0, 0])
    gaps["top"] = y_diff_top

    # Bottom direction
    y_diff_bottom = []
    y_cut_ribbon_bottom = ribbon_homo[int(ribbon_homo.shape[0] / 2):, :]
    for x in range(y_cut_ribbon_bottom.shape[1]):
        y_cut_idx = np.where(y_cut_ribbon_bottom[:, x] > 0)[0]
        if y_cut_idx.size > 0:
            y_min_value = np.min(y_cut_idx)
            y_max_value = np.max(y_cut_idx)
            y_diff_bottom.append([y_min_value, y_max_value, y_max_value - y_min_value])
        else:
            y_diff_bottom.append([0, 0, 0])
    gaps["bottom"] = y_diff_bottom

    return gaps


def plot_ribbon_gaps(ribbon_homo, gaps):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(ribbon_homo)

    for index, value in enumerate(gaps["left"]):
        if value[2] < 50:
            ax.plot((value[0], value[1]), (index, index), linewidth=3, color='red')
            ax.text(value[1] + 5, index, "under 50", verticalalignment='center', color='white')

    for index, value in enumerate(gaps["right"]):
        if value[2] < 50:
            ax.plot((ribbon_homo.shape[1] / 2 + value[0], ribbon_homo.shape[1] / 2 + value[1]), (index, index),
                    linewidth=3, color='red')
            ax.text(ribbon_homo.shape[1] - value[1] - 5, index, "under 50", verticalalignment='center', color='white')

    for index, value in enumerate(gaps["top"]):
        if value[2] < 50:
            ax.plot((index, index), (value[0], value[1]), linewidth=3, color='red')
            # ax.text(index, value[1] + 50, "under 50", horizontalalignment='center', color='white')
    for index, value in enumerate(gaps["bottom"]):
        if value[2] < 50:
            ax.plot((index, index), (ribbon_homo.shape[0] / 2 + value[0], ribbon_homo.shape[0] / 2 + value[1]),
                    linewidth=3, color='red')

    ax.axis('off')
    plt.savefig(f"{predicted_dir}/ribbon_gap.png", bbox_inches='tight', pad_inches=0, transparent=True)
    plt.show()

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

def  calculate_percent_errors_of_list(list1, list2):
    percent_error = []
    for i in range(len(list1)):
        value1 = float(list1[i])
        value2 = float(list2[i])
        error = abs(value1 - value2) / value2 * 100
        percent_error.append(error)
    return percent_error

if __name__ == "__main__":


    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #몰라 오류 뜨는데 chatgpt한테 물어보니 이거 하라고 함


    """
    1. segmented0~9.npy를 load 하여서 모두 합침 -> total mask
    """



    total_mask = np.zeros_like(np.load(f"{predicted_dir}/segmented0.npy"))
    img = cv2.imread(predict_dir)
    for idx in range(9):
        segmented = np.load(f"{predicted_dir}/segmented{idx}.npy")
        total_mask += segmented


    """
    2. mask 중 가장 큰 친구를 ribbon_mask로 저장하였음 역시 load 함
    """

    ribbon_mask = np.load(f"{predicted_dir}/ribbon_mask.npy")
    # plt.imshow(ribbon_mask)
    # plt.show()
    #
    """
    3. total - ribbon = dab 만 남겠죠??
    """
    dab_mask = total_mask - ribbon_mask
    # plt.imshow(dab_mask)
    # plt.show()


    """
    4. YOLO method 중 xy 추출하는 기능을 사용하여 ribbon 의 외곽선 xy 를 검출하여 RANSAC
    """

    xy = np.load(f"{predicted_dir}/ribbon_point.npy")
    # plt.imshow(img)
    # plt.scatter(xy[:,0], xy[:,1])
    # plt.show() ## Scatter 확인 코드

    x_data = xy[:, 0]
    y_data = xy[:, 1]
    x_tmp = x_data.copy()
    y_tmp = y_data.copy()
    ransac_line = []
    intersection_points = []
    while True:
        inliers, outliers, line_twopoint = plot_ransac_revised(x_tmp, y_tmp)

        if x_tmp[inliers].shape[0] >= 2:
            # inliers, two points for line 기록 저장
            ransac_line.append((x_tmp[inliers], y_tmp[inliers], line_twopoint))

        # 나머지 점들 (outliers)
        x_tmp = x_tmp[outliers]
        y_tmp = y_tmp[outliers]

        # if x_tmp.shape[0] <= 2 or len(ransac_line) == 4:
        if len(ransac_line) == 4:
            break
    x_min, x_max, y_min, y_max = x_data.min(), x_data.max(), y_data.min(), y_data.max()
    for i in range(len(ransac_line)):
        for j in range(i + 1, len(ransac_line)):
            (x1, x2), (y1, y2) = ransac_line[i][2]
            (x3, x4), (y3, y4) = ransac_line[j][2]
            x, y = line_intersection([[x1, y1], [x2, y2]], [[x3, y3], [x4, y4]], x_min, x_max, y_min, y_max)
            if x != -12345 or y != -12345:
                intersection_points.append(np.array((x, y)))
    if not len(intersection_points) == 4:
        raise Exception("intersection points doesn't satifies 4, correct the threshold of RANSAC")

    """
    5. YOLO result mask는 이상하게 원본의 image size를 반영하지 않음. 따라서 모든 mask를 resize
    """

    total_mask = cv2.resize(total_mask, (img.shape[1], img.shape[0]))
    ribbon_mask = cv2.resize(ribbon_mask, (img.shape[1], img.shape[0]))
    dab_mask = cv2.resize(dab_mask, (img.shape[1], img.shape[0]))

    """
    6. Homography!
    """

    img_homo = homography(img, intersection_points, width=600, height=1200)
    ribbon_homo = homography(ribbon_mask, intersection_points, width=600, height=1200)
    dab_homo = homography(dab_mask, intersection_points, width=600, height=1200)
    total_mask_homo = homography(total_mask, intersection_points, width=600, height=1200)
    # plt.imshow(dab_homo)
    # plt.show()

######################################################################################################
# 거리구하기
######################################################################################################
    """
    7. dab 간의 거리를 구해봅시다
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8,12))

    indices = np.where(dab_homo > 0)
    points = list(zip(indices[1], indices[0]))
    cluster_dab = cluster(points)
    # dab_distances_horiz, dab_distances_vert =  draw_dab_distances(cluster_dab)
    img_homo_dab_distance = cv2.cvtColor(img_homo, cv2.COLOR_BGR2RGB)
    # print(dab_distances_horiz)
    # print(dab_distances_vert)
    # plt.imshow(img_homo_dab_distance)
    # plt.show()


    centers = coord_sort_z(cluster_dab)
    centers_x = centers[:, 0]
    centers_y = centers[:, 1]
    distances_vert = []
    distances_horiz = []
    a = 1
    for k in range(0, 8, 2):  # 수정된 범위 (0부터 8까지, 간격은 2)
        dist_vert = calculate_distance(centers[k], centers[k + 1])
        distances_vert.append(round(dist_vert * a, 2))
        ax1.plot((centers_x[k], centers_x[k + 1]), (centers_y[k], centers_y[k + 1]), marker="o", linestyle=":")
        ax1.text((centers_x[k] + centers_x[k + 1]) / 2, (centers_y[k] + centers_y[k + 1]) / 2, distances_vert[-1],
                 color="white")

    for k in range(0, 6):  # 수정된 범위 (0부터 6까지)
        dist_horiz = calculate_distance(centers[k], centers[k + 2])
        distances_horiz.append(round(dist_horiz * a, 2))
        ax1.plot((centers_x[k], centers_x[k + 2]), (centers_y[k], centers_y[k + 2]), marker="s", linestyle="--")
        ax1.text((centers_x[k] + centers_x[k + 2]) / 2, (centers_y[k] + centers_y[k + 2]) / 2, distances_horiz[-1],
                 color="white")
    ax1.set_title("distance each dab")
    ax1.imshow(img_homo_dab_distance)





    """
    8. insulation으로 부터의 거리
    """

    width = 600
    height = 1200
    cluster_dab = coord_sort_z(cluster_dab)


    for k in range(0, len(cluster_dab), 2):
        ax2.plot((0, cluster_dab[k][0]), (cluster_dab[k][1], cluster_dab[k][1]))
        ax2.text(cluster_dab[k][0] / 2, cluster_dab[k][1], cluster_dab[k][0], color='white')  # y축으로 부터의 거리를 text
    for k in range(1, len(cluster_dab), 2):
        ax2.plot((cluster_dab[k][0], width), (cluster_dab[k][1], cluster_dab[k][1]))
        ax2.text((width + cluster_dab[k][0]) / 2, cluster_dab[k][1], width - cluster_dab[k][0],
                 color='white')  # y축으로 부터의 거리를 text
    for k in range(2):
        ax2.text(cluster_dab[k][0], cluster_dab[k][1] / 2, cluster_dab[k][1], color='white')  # x축으로 부터의 거리를 text
        ax2.plot((cluster_dab[k][0], cluster_dab[k][0]), (0, cluster_dab[k][1]))
    for k in range(6, len(cluster_dab, )):
        ax2.text(cluster_dab[k][0], (height + cluster_dab[k][1]) / 2, height - cluster_dab[k][1],
                 color='white')  # width로 부터의 거리를 text
        ax2.plot((cluster_dab[k][0], cluster_dab[k][0]), (cluster_dab[k][1], height))


    img_homo_dab_distance_from_insulation = cv2.cvtColor(img_homo, cv2.COLOR_BGR2RGB)
    ax2.set_title("distance from insulation")
    ax2.imshow(img_homo_dab_distance_from_insulation)

    indices = np.where(GT_dab_homo > 0)
    points = list(zip(indices[1], indices[0]))
    cluster_dab = cluster(points)
    dab_distance_GT = cv2.cvtColor(GT_dab_homo, cv2.COLOR_BGR2RGB)

    centers = coord_sort_z(cluster_dab)
    centers_x = centers[:, 0]
    centers_y = centers[:, 1]
    distances_vert = []
    distances_horiz = []
    a = 1
    for k in range(0, 8, 2):  # 수정된 범위 (0부터 8까지, 간격은 2)
        dist_vert = calculate_distance(centers[k], centers[k + 1])
        distances_vert.append(round(dist_vert * a, 2))
        ax3.plot((centers_x[k], centers_x[k + 1]), (centers_y[k], centers_y[k + 1]), marker="o", linestyle=":")
        ax3.text((centers_x[k] + centers_x[k + 1]) / 2, (centers_y[k] + centers_y[k + 1]) / 2, distances_vert[-1],
                 color="magenta")

    for k in range(0, 6):  # 수정된 범위 (0부터 6까지)
        dist_horiz = calculate_distance(centers[k], centers[k + 2])
        distances_horiz.append(round(dist_horiz * a, 2))
        ax3.plot((centers_x[k], centers_x[k + 2]), (centers_y[k], centers_y[k + 2]), marker="s", linestyle="--")
        ax3.text((centers_x[k] + centers_x[k + 2]) / 2, (centers_y[k] + centers_y[k + 2]) / 2, distances_horiz[-1],
                 color="magenta")
    ax3.set_title("distance each dab GT")
    ax3.imshow(dab_distance_GT)

    width = 600
    height = 1200
    cluster_dab = coord_sort_z(cluster_dab)

    for k in range(0, len(cluster_dab), 2):
        ax4.plot((0, cluster_dab[k][0]), (cluster_dab[k][1], cluster_dab[k][1]))
        ax4.text(cluster_dab[k][0] / 2, cluster_dab[k][1], cluster_dab[k][0], color='magenta')  # y축으로 부터의 거리를 text
    for k in range(1, len(cluster_dab), 2):
        ax4.plot((cluster_dab[k][0], width), (cluster_dab[k][1], cluster_dab[k][1]))
        ax4.text((width + cluster_dab[k][0]) / 2, cluster_dab[k][1], width - cluster_dab[k][0],
                 color='magenta')  # y축으로 부터의 거리를 text
    for k in range(2):
        ax4.text(cluster_dab[k][0], cluster_dab[k][1] / 2, cluster_dab[k][1], color='magenta')  # x축으로 부터의 거리를 text
        ax4.plot((cluster_dab[k][0], cluster_dab[k][0]), (0, cluster_dab[k][1]))
    for k in range(6, len(cluster_dab, )):
        ax4.text(cluster_dab[k][0], (height + cluster_dab[k][1]) / 2, height - cluster_dab[k][1],
                 color='magenta')  # width로 부터의 거리를 text
        ax4.plot((cluster_dab[k][0], cluster_dab[k][0]), (cluster_dab[k][1], height))

    img_homo_dab_distance_from_insulation = cv2.cvtColor(dab_distance_GT, cv2.COLOR_BGR2RGB)
    ax4.set_title("distance from insulation_GT")
    ax4.imshow(img_homo_dab_distance_from_insulation)
    fig.suptitle('Dab Distance')

    ax1_texts = [text.get_text() for text in ax1.texts]
    ax2_texts = [text.get_text() for text in ax2.texts]
    ax3_texts = [text.get_text() for text in ax3.texts]
    ax4_texts = [text.get_text() for text in ax4.texts]

    average_error_each_dab = np.average(calculate_percent_errors_of_list(ax1_texts,ax3_texts))
    average_error_insulation_dab = np.average(calculate_percent_errors_of_list(ax2_texts, ax4_texts))
    # for i, error in enumerate(percent_error_dab):
    #     print(f"Percent error at index {i}: {error:.2f}%")

    # average_error_each_dab = np.average(percent_error_dab)

    # Correct syntax to format the average to two decimal places
    ax3.set_xlabel(f"{average_error_each_dab:.2f}%")
    ax4.set_xlabel(f"{average_error_insulation_dab:.2f}%")

    plt.savefig(f"{predicted_dir}/distance_dab.png")
    plt.show()

    max_length = max(len(ax1_texts), len(ax2_texts), len(ax3_texts), len(ax4_texts))

    # Pad the lists with empty strings to make them all the same length
    ax1_texts.extend([''] * (max_length - len(ax1_texts)))
    ax2_texts.extend([''] * (max_length - len(ax2_texts)))
    ax3_texts.extend([''] * (max_length - len(ax3_texts)))
    ax4_texts.extend([''] * (max_length - len(ax4_texts)))

    df_texts = pd.DataFrame({
        "predicted distance each dab" : ax1_texts,
        'predicted distance from insulation': ax2_texts,
        'ground truth distance each dab': ax3_texts,
        'ground truth distance from insulation': ax4_texts
    })

    # Save to Excel
    excel_path = f"{predicted_dir}/dab_distances.xlsx"
    df_texts.to_excel(excel_path, index=False)

    print(f"Dab distances saved to {excel_path}")


    """
    9. ribbon gap
    """

    # x_gap = ribbon_gap(ribbon_homo, axis="0")  # axis = 0 : 행으로 절단해서 x value 사이의 거리를 구함
    # print(x_gap)
    # y_gap = ribbon_gap(ribbon_homo, axis="1")
    # print(y_gap[:1000])
    gaps = ribbon_gap_4_directions(ribbon_homo)
    plot_ribbon_gaps(ribbon_homo, gaps)

    max_length = max(len(gaps["left"]), len(gaps["right"]), len(gaps["top"]), len(gaps["bottom"]))

    # Pad the lists with empty entries to make them the same length
    for direction in ["left", "right", "top", "bottom"]:
        while len(gaps[direction]) < max_length:
            gaps[direction].append(None)

        # Create DataFrame
    df_gaps = pd.DataFrame({
        'left_diff': gaps["left"][2],
        'right_diff': gaps["right"][2],
        'top_diff': gaps["top"][2],
        'bottom_diff': gaps["bottom"][2]
    })

    # Save to Excel
    df_gaps.to_excel(f"{predicted_dir}/ribbon_gaps.xlsx", index=False)


    print(f"Gaps saved to {predicted_dir}")


    ######################################################################################################
    # 면적구하기
    ######################################################################################################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,10))

    area = width*height
    predicted_area = np.count_nonzero(total_mask_homo)
    GT_area = np.count_nonzero(GT_total_homo)
    error = predicted_area/area * 100

    ax1.set_title("predicted, {}".format(predict_dir))
    ax1.imshow(cv2.cvtColor(img_homo,cv2.COLOR_RGB2BGR))
    ax1.set_xlabel(predicted_area)


    ax2.imshow(GT_total_homo)
    ax2.set_title("Ground Truth")
    ax2.set_xlabel(GT_area)


    percent_error = abs(GT_area - predicted_area) / GT_area * 100
    fig.suptitle("Percent Error : {}".format(percent_error))
    plt.savefig(f"{predicted_dir}/whole_area.png")
    plt.show()






    ######################################################################################################
    # 면적구하기 DAB간의 거리 구하기
    ######################################################################################################
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.measure import label, regionprops
    from PIL import Image

    # Assuming 'GT_dab_homo' and 'dab_homo' are loaded correctly from 'after_predict.py'
    data_reloaded_GT = GT_dab_homo
    data_reloaded_Predicted = dab_homo


    # Function to process and plot data
    def process_data(data, ax, title):
        labels = label(data, connectivity=2)
        props = regionprops(labels)

        filtered_regions = []

        region_counts = []
        for prop in props:
            if prop.area > 10:  # Area threshold to filter small noisy regions
                minr, minc, maxr, maxc = prop.bbox
                margin = 0
                minr = max(minr - margin, 0)
                minc = max(minc - margin, 0)
                maxr = min(maxr + margin, data.shape[0])
                maxc = min(maxc + margin, data.shape[1])
                filtered_regions.append((minr, minc, maxr, maxc))


                # Count the number of '1's within the region
                region_data = data[minr:maxr, minc:maxc]
                count_ones = np.count_nonzero(region_data)
                region_counts.append(count_ones)

        # Sort regions to maintain the specified order
        left_regions = sorted([r for r in filtered_regions if r[1] < data.shape[1] / 2], key=lambda x: x[0])
        right_regions = sorted([r for r in filtered_regions if r[1] >= data.shape[1] / 2], key=lambda x: x[0])
        sorted_regions = left_regions + right_regions

        ax.imshow(data, cmap='gray')
        for i, (minr, minc, maxr, maxc) in enumerate(sorted_regions):
            rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(rect)
            ax.text(minc, minr - 5, f'AREA {i} (Count: {region_counts[i]})', color='yellow', fontsize=8,
                    ha='left')


        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return region_counts


    def percent_error(Pre, GT):
        percent_error = abs(GT-Pre)/GT*100
        return percent_error

    # Set up the plot with two subplots
    fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(12,10))
    area_dab_Predicted = process_data(data_reloaded_Predicted, ax3, 'Dab area of Predicted')
    area_dab_GT = process_data(data_reloaded_GT, ax4, 'Dab area of Ground Truth')

    percent_error_dab = []
    for k in range(8):
        percent_error_dab.append(percent_error(area_dab_Predicted[k],area_dab_GT[k]))
        # print(percent_error(area_dab_Predicted[k],area_dab_GT[k]))

    percent_error_dab = np.average(percent_error_dab)
    fig.suptitle("Percent Error of Dab Area : {}%".format(round(percent_error_dab, 2)))
    plt.savefig(f"{predicted_dir}/dab_area.png")

    plt.show()


    # Create DataFrame
    df_area = pd.DataFrame({
        'area_predicted': predicted_area,
        'area_dab_predicted': area_dab_Predicted,
        'area_groundtruth': GT_area,
        'area_dab_groundtruth': area_dab_GT,
    })


    df_area.to_excel(f"{predicted_dir}/dab_area.xlsx", index=False)

    import numpy as np
    import matplotlib.pyplot as plt
    from skimage.measure import label, regionprops
    import pandas as pd


    def process_data(data):
        labels = label(data, connectivity=2)
        props = regionprops(labels)

        diameters = []
        for prop in props:
            if prop.area > 10:  # Area threshold to filter small noisy regions
                minr, minc, maxr, maxc = prop.bbox
                margin = 0
                minr = max(minr - margin, 0)
                minc = max(minc - margin, 0)
                maxr = min(maxr + margin, data.shape[0])
                maxc = min(maxc + margin, data.shape[1])
                diameters.append((maxc - minc, maxr - minr))

        return diameters


    # Ensure data_reloaded_Predicted and data_reloaded_GT are defined
    # These should be 2D numpy arrays representing the predicted and ground truth data

    # Example:
    # data_reloaded_Predicted = np.array([[...]])  # Replace with actual data
    # data_reloaded_GT = np.array([[...]])

    # Process the data to get diameters
    diameters_predicted = process_data(data_reloaded_Predicted)
    diameters_gt = process_data(data_reloaded_GT)

    # Create a DataFrame to save diameters
    df = pd.DataFrame({
        'Predicted_Diameter_Width': [d[0] for d in diameters_predicted],
        'Predicted_Diameter_Height': [d[1] for d in diameters_predicted],
        'GT_Diameter_Width': [d[0] for d in diameters_gt],
        'GT_Diameter_Height': [d[1] for d in diameters_gt]
    })

    # Save to Excel
    excel_path = f"{predicted_dir}/diameters.xlsx"
    df.to_excel(excel_path, index=False)
    0





    # Save to text file
    text_path = f"{predicted_dir}/diameters.txt"
    df.to_csv(text_path, index=False, sep='\t')

    print(f"Diameters saved to {excel_path} and {text_path}")




