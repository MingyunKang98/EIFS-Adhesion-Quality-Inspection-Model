import matplotlib.pyplot as plt
import numpy as np
import cv2

from GT import GT_dab
from GT import GT_total
from GT import GT_ribbon
from GT import homography
from GT import coord_sort_n
from GT import dir
from predict import predicted_dir



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
    plt.savefig(f"{predicted_dir}/GT_ribbon_gap.png", bbox_inches='tight', pad_inches=0, transparent=True)





intersection_points = [[673,815],[897,3461],[1884,519],[2404,3291]]
GT_dab_homo = homography(GT_dab, intersection_points, width=600, height=1200)
GT_total_homo = homography(GT_total, intersection_points, width=600, height=1200)
GT_ribbon_homo = homography(GT_ribbon, intersection_points, width=600, height=1200)

GT_total_homo = cv2.cvtColor(GT_total_homo, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(GT_total_homo, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

GT_dab_homo = cv2.cvtColor(GT_dab_homo, cv2.COLOR_BGR2GRAY)
(thresh, im_bw) = cv2.threshold(GT_dab_homo, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

image = cv2.imread(dir)
image_homo = homography(image, intersection_points, width=600, height=1200)
cv2.imwrite("{}/GT_image_homo.jpg".format(predicted_dir), image_homo)
cv2.imwrite("{}/GT_dab_homo.jpg".format(predicted_dir), GT_dab_homo)
cv2.imwrite("{}/GT_ribbon_homo.jpg".format(predicted_dir), GT_ribbon_homo)
cv2.imwrite("{}/GT_total_homo.jpg".format(predicted_dir), GT_total_homo)

gaps = ribbon_gap_4_directions(GT_ribbon_homo)
# plot_ribbon_gaps(GT_ribbon_homo, gaps)
# plt.imshow(GT_total_homo)
# plt.show()


