from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np
from GT import dir
from ultralytics.engine.predictor import BasePredictor
import ultralytics.engine.results



# Load a pretrained YOLOv8n model
model = YOLO("../../ultralytics-main/ultralytics/YOLOv8/train9/weights/best.pt")
predict_dir = "../IMG_4221_GT.JPG"




# results = model(dir, save=False, save_txt=False,save_crop=False,show_conf = False,show_labels=True,show_boxes=False)  # list of 1 Results object
results = model(predict_dir, save=True)  # list of 1 Results object

predicted_dir = results[0].save_dir

####################################
mask_arrays = results[0].masks.data.cpu().numpy()

ones_counts = [np.sum(mask) for mask in mask_arrays]
# 1의 개수가 가장 많은 마스크 배열의 인덱스를 찾기
max_ones_index = np.argmax(ones_counts)
# 1이 가장 많은 마스크 배열 추출
max_ones_mask = mask_arrays[max_ones_index]

np.save(f"{predicted_dir}/ribbon_mask.npy", max_ones_mask)

###########################################

for idx in range(9):
  # Retrieving and converting the segmentation mask to a numpy array
  segmented_mask = results[0].masks.data[idx].cpu().numpy()

  # Saving each segmented mask to a numpy file
  file_path = f"{predicted_dir}/segmented{idx}.npy"
  np.save(file_path, segmented_mask)

##################################### points 추출 -> intersection points로 활용할 예정

mask = results[0].masks.cpu()

points = mask.xy

lengths = np.array([len(sub_list) for sub_list in points])
# 가장 큰 길이의 인덱스를 찾기
max_length_index = np.argmax(lengths)
# 가장 긴 서브 리스트 추출
longest_sub_list = points[max_length_index]
# ribbon_point_file_path = f"{predicted_dir}/ribbon_point.npy"
np.save(f"{predicted_dir}/ribbon_point.npy", longest_sub_list)