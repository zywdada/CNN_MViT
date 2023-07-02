import cv2
import matplotlib as plt
import os
import numpy as np
from tqdm import tqdm
def white_balance_2(img_input):
    img = img_input.copy()
    b, g, r = cv2.split(img)
    m, n, t = img.shape
    sum_ = np.zeros(b.shape)
    sum_ = (b.astype(int) + g.astype(int)  + r.astype(int))
#     for i in range(m):
#         for j in range(n):
#             sum_[i][j] = int(b[i][j]) + int(g[i][j]) + int(r[i][j])
    hists, bins = np.histogram(sum_.flatten(), 766, [0, 766])
    Y = 765
    num, key = 0, 0
    ratio = 0.01
    while Y >= 0:
        num += hists[Y]
        if num > m * n * ratio / 100:
            key = Y
            break
        Y = Y - 1
 
    sum_b, sum_g, sum_r = 0, 0, 0
    time = 0
    for i in range(m):
        for j in range(n):
            if sum_[i][j] >= key:
                sum_b += b[i][j]
                sum_g += g[i][j]
                sum_r += r[i][j]
                time = time + 1
    if sum_b < 0:
        sum_b = 1
    if sum_g < 0:
        sum_g = 1
    if sum_r < 0:
        sum_r = 1
    avg_b = sum_b / time
    avg_g = sum_g / time
    avg_r = sum_r / time

    if avg_b < 0:
        avg_b = 1
    if avg_g < 0:
        avg_g = 1
    if avg_r < 1:
        avg_r = 1
 
    maxvalue = float(np.max(img))

    for i in range(m):
        for j in range(n):
            b = int(img[i][j][0]) * maxvalue / int(avg_b)
            g = int(img[i][j][1]) * maxvalue / int(avg_g)
            r = int(img[i][j][2]) * maxvalue / int(avg_r)
            if b > 255:
                b = 255
            if b < 0:
                b = 0
            if g > 255:
                g = 255
            if g < 0:
                g = 0
            if r > 255:
                r = 255
            if r < 0:
                r = 0
            img[i][j][0] = b
            img[i][j][1] = g
            img[i][j][2] = r
 
    return img
DATA_DIR = ''
OUTPUT_DIR = ''

def get_videofolder(dir):
    videos=os.listdir(dir)
    changed = os.listdir(OUTPUT_DIR)
    for video in tqdm(videos, desc='Processing'):
        if video not in changed or video == changed[-1]:
            video_path = os.path.join(dir, video)
            output_dir=os.path.join(OUTPUT_DIR, video)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            frames = os.listdir(video_path)
            for frame in frames:
                frame_path = os.path.join(video_path, frame)
                img = cv2.imread(frame_path, cv2.IMREAD_COLOR)
                frame_wb = white_balance_2(img)
                output_file = os.path.join(output_dir, frame)
                cv2.imwrite(output_file, frame_wb)
    return
get_videofolder(DATA_DIR)
# img1 = cv2.imread('C:\\Users\\12533\\Desktop\\0_0.png',cv2.IMREAD_COLOR)
# img2 = cv2.imread('C:\\Users\\12533\\Desktop\\0_0.png',cv2.IMREAD_COLOR)
# img1 =white_balance_2(img)
# # IMG = np.hstack((img1, img))



