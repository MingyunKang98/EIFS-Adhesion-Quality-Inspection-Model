import os
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import numpy as np

def parse_polygon_annotation(annotation_line, img_width, img_height):
    parts = list(map(float, annotation_line.split()))
    class_id = int(parts[0])
    points = [(int(x * img_width), int(y * img_height)) for x, y in zip(parts[1::2], parts[2::2])]
    return class_id, points

if __name__ == "__main__":
    # 이미지와 어노테이션 파일 경로 설정
    # image_path = "./donut/images/brooke-lark-5BbB3WPi128-unsplash.jpg"
    # annotation_path = "./donut/labels/brooke-lark-5BbB3WPi128-unsplash.txt"

    image_path = "./IMG_4238_JPG.rf.9cecaa724f2fc8111ae692201ba551ca.jpg"
    annotation_path = "./IMG_4238_JPG.rf.9cecaa724f2fc8111ae692201ba551ca_donut_merged.txt"

    # 이미지 불러오기
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # 어노테이션 파일 읽기
    with open(annotation_path, 'r') as file:
        for line in file:
            # 어노테이션 파싱
            class_id, polygon_points = parse_polygon_annotation(line, img_width, img_height)

            # 이미지에 바운딩 박스 그리기
            if class_id == 0:
                cv2.polylines(image, [np.array(polygon_points)], isClosed=True, color=(0, 255, 0), thickness=2)
            elif class_id == 1:
                cv2.polylines(image, [np.array(polygon_points)], isClosed=True, color=(0, 0, 255), thickness=2)
            elif class_id == 2:
                cv2.polylines(image, [np.array(polygon_points)], isClosed=True, color=(255, 0, 0), thickness=2)

    # 이미지 표시
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Class ID: {class_id}")
    plt.show()
