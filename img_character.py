# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:50:19 2019

@author: GeePL
"""
import os
import random
import shutil

'''
xml_dir = r'D:\dataset2018\raw_img' 
img_dir = r'D:\dataset2018\raw_img'

imgs = [s for s in os.listdir(img_dir) if s[-4:] == ".jpg"]
xmls = [s for s in os.listdir(xml_dir) if s[-4:] == ".xml"]
assert len(imgs) == len(xmls)

rare_flaws = [67, 75, 77, 80]
commom_flaws = [61, 62, 63, 64, 71, 72, 73]
lack_img_flaws = [65, 66, 68, 69, 70, 74, 76, 78, 79, 81]

print("rare flaws")
for i in rare_flaws:
    path = os.path.join(xml_dir, str(i))
    print(str(i) + " " + str(len(os.listdir(path))))
    
print("commom flaws") 
for i in commom_flaws:
    path = os.path.join(xml_dir, str(i))
    print(str(i) + " " + str(len(os.listdir(path))))
    
print("lack img flaws") 
for i in lack_img_flaws:
    path = os.path.join(xml_dir, str(i))
    print(str(i) + " " + str(len(os.listdir(path))))
'''
"""
##Mean && Std
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import imread
img_mean = 0
img_std = 0
for flaw_number in commom_flaws:
    print(flaw_number)
    img_path = os.path.join(img_dir, str(flaw_number))  
    imgs = [s for s in os.listdir(img_path) if s[-4:]=='.jpg']
    img_num = len(imgs)
    R_channel = 0
    R_channel_square = 0
    pixels_num = 0
    np.random.shuffle(imgs)
    for img_name in imgs[:300]:
        img = imread(os.path.join(img_path, img_name))
        h, w, _ = img.shape
        pixels_num += h*w       # 统计单个通道的像素数量
        R_temp = img[:, :, 0]
        R_channel += np.sum(R_temp)
        R_channel_square += np.sum(np.power(R_temp, 2.0))
    R_mean = R_channel / pixels_num
    R_std = np.sqrt(R_channel_square/pixels_num - R_mean*R_mean)
    print(R_mean)
    img_mean += R_mean
    print(R_std)
    img_std += R_std
print(img_mean/len(commom_flaws))
print(img_std/len(commom_flaws))
"""
xml_dir = r'D:\dataset2018-05-23\trian_spilted_1200_imgs'

xml_list = [s for s in os.listdir(xml_dir) if s[-4:]=='.xml']
for xml_file in xml_list:
    os.remove(os.path.join(xml_dir, xml_file))
#    shutil.copy(os.path.join(xml_dir, xml_file),
#            os.path.join(new_xml_dir, xml_file))