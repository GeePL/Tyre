# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:50:19 2019

@author: GeePL
"""
import os
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