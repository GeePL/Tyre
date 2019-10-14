# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 19:31:15 2019

@author: GeePL
"""
import os
import shutil
import xml.etree.ElementTree as ET
import random
import cv2
from PIL import Image
from create_xml import create_xml
import numpy as np
import json

rare_flaws = [67, 75, 77, 80]
commom_flaws = [61, 62, 63, 64, 71, 72, 73]
lack_img_flaws = [65, 66, 68, 69, 70, 74, 76, 78, 79, 81]
sep = os.sep

def get(root, name):
    vars = root.findall(name)
    return vars

def get_and_check(root, name, length):
    vars = root.findall(name)
    if len(vars) == 0:
        raise NotImplementedError('Can not find %s in %s.'%(name, root.tag))
    if length > 0 and len(vars) != length:
        raise NotImplementedError('The size of %s is supposed to be %d, but is %d.'%(name, length, len(vars)))
    if length == 1:
        vars = vars[0]
    return vars

#左下-右上
def intersection(ai, bi):
	x = max(ai[0], bi[0])
	y = max(ai[1], bi[1])
	w = min(ai[2], bi[2]) - x
	h = min(ai[3], bi[3]) - y
	if w < 0 or h < 0:
		return 0
	return w*h

#交并比
def iou(a, b):
	# a is the box in splited img
   # b is the gt bbox

	if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
		return 0.0
    
	area_i = intersection(a, b)
	area_u = (b[2] - b[0]) * (b[3] - b[1])

	return float(area_i) / float(area_u + 1e-6)

def divide_raw_imgs_into_train_and_val():
    data_path = r'D:\dataset2018-05-23\raw_img_after_classify'
    train_raw_data_path = r'D:\dataset2018-05-23\train_raw_imgs'
    val_raw_data_path = r'D:\dataset2018-05-23\val_raw_imgs'
    if not os.path.exists(train_raw_data_path):
        os.makedirs(train_raw_data_path)
    if not os.path.exists(val_raw_data_path):
        os.makedirs(val_raw_data_path)
    for j in commom_flaws:
        path = os.path.join(data_path, str(j))
        imgs = [s for s in os.listdir(path) if s[-4:]=='.jpg']
        xmls = [s for s in os.listdir(path) if s[-4:]=='.xml']
        assert len(imgs) == len(xmls)
        print(str(j)+ " "+ str(len(imgs)))
        for i in range(len(imgs)):
            assert imgs[i][:-4] == xmls[i][:-4]
            xml_file = os.path.join(path, xmls[i])
            tree = ET.parse(xml_file)
            root = tree.getroot()
            size = get_and_check(root, 'size', 1)
            # 图片的基本信息
            width = int(get_and_check(size, 'width', 1).text)
            if width<500 | width>5000:
                continue
            height = int(get_and_check(size, 'height', 1).text)
            if height>10000 | height<3000:
                continue
            # 处理每个标注的检测框
            for obj in get(root, 'object'):
                # 取出检测框类别名称
                category = get_and_check(obj, 'name', 1).text
                category = category[:2]
                if int(category) in commom_flaws:
                    bndbox = get_and_check(obj, 'bndbox', 1)
                    xmin = int(get_and_check(bndbox, 'xmin', 1).text) 
                    ymin = int(get_and_check(bndbox, 'ymin', 1).text)
                    xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                    ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                    assert(xmax > xmin)
                    assert(ymax > ymin)
                    if random.randint(0, 10)>=2:
                        shutil.copy(os.path.join(path, xmls[i]),
                            os.path.join(train_raw_data_path, xmls[i]))
                        shutil.copy(os.path.join(path, imgs[i]),
                            os.path.join(train_raw_data_path, imgs[i]))
                    else:
                        shutil.copy(os.path.join(path, xmls[i]),
                                os.path.join(val_raw_data_path, xmls[i])) 
                        shutil.copy(os.path.join(path, imgs[i]),
                            os.path.join(val_raw_data_path, imgs[i]))
                    break;

def resize_raw_imgs():
    new_W = 2000
    data_input_path = r'D:\dataset2018-05-23\val_raw_imgs'
    data_output_path = r'D:\dataset2018-05-23\val_resized_2000_imgs'
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
        
    imgs = [s for s in os.listdir(data_input_path) if s[-4:]=='.jpg']
    xmls = [s for s in os.listdir(data_input_path) if s[-4:]=='.xml']
    assert len(imgs) == len(xmls)
    print(data_input_path + " "+ str(len(imgs)))
    
    for i in range(len(imgs)):
        assert imgs[i][:-4] == xmls[i][:-4]
        img_path = os.path.join(data_input_path, imgs[i])
        xml_path = os.path.join(data_input_path, xmls[i])
        ## resize img
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text) 
        if width<=0 | height<=0:
            continue
        ratio = float(new_W)/width
        new_H = int(height * ratio)
        raw_img = cv2.imread(img_path) 
        resized_img = cv2.resize(raw_img, (new_W, new_H))
        
        img_details = {'height':new_H, 'width':new_W, \
                       'bboxes':[], 'filename':xmls[i][:-4]}
        ## resize xml
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            category = category[:2]
            assert (int(category) > 60) & (int(category) < 90)
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(int(get_and_check(bndbox, 'xmin', 1).text) * ratio)
            ymin = int(int(get_and_check(bndbox, 'ymin', 1).text) * ratio)
            xmax = int(int(get_and_check(bndbox, 'xmax', 1).text) * ratio)
            ymax = int(int(get_and_check(bndbox, 'ymax', 1).text) * ratio)
            assert(xmax > xmin)
            assert(ymax > ymin)
            '''
            cv2.rectangle(raw_img,(xmin, ymin), (xmax, ymax), (0,0,255),2)           
            textLabel = category          
            textOrg = (xmin, ymin)
            cv2.putText(raw_img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            Image.fromarray(raw_img).save(data_output_path+sep+imgs[i])
            '''
            img_details['bboxes'].append(
                    {'class':category, 'x1':xmin,
                     'y1':ymin, 'x2':xmax, 'y2':ymax})
            
        create_xml(img_details, data_output_path)
        '''
        xml_path = os.path.join(val_resized_imgs, xmls[i])
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in get(root, 'object'):
            category = get_and_check(obj, 'name', 1).text
            category = category[:2]
            assert (int(category) > 60) & (int(category) < 90)
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(int(get_and_check(bndbox, 'xmin', 1).text))
            ymin = int(int(get_and_check(bndbox, 'ymin', 1).text))
            xmax = int(int(get_and_check(bndbox, 'xmax', 1).text))
            ymax = int(int(get_and_check(bndbox, 'ymax', 1).text))
            assert(xmax > xmin)
            assert(ymax > ymin)
            cv2.rectangle(resized_img,(xmin, ymin), (xmax, ymax), (0,0,255),2)           
            textLabel = category          
            textOrg = (xmin, ymin)
            cv2.putText(resized_img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
        '''
        Image.fromarray(resized_img).save(data_output_path+sep+imgs[i])

def split_resized_img():
    data_path = r'D:\dataset2018-05-23\train_resized_imgs'
    data_output_path = r'D:\dataset2018-05-23\trian_spilted_1200_imgs'
    W_pieces = 2
    H_pieces = 10
    new_size = 1200 
    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
    imgs = [s for s in os.listdir(data_path) if s[-4:]=='.jpg']
    xmls = [s for s in os.listdir(data_path) if s[-4:]=='.xml']
    assert len(imgs) == len(xmls)
    print(len(imgs))
    for i in range(len(imgs)):
        assert imgs[i][:-4] == xmls[i][:-4]
        img_path = os.path.join(data_path, imgs[i])
        xml_path = os.path.join(data_path, xmls[i])
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = get_and_check(root, 'size', 1)
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text) 
        H_overlap = int((H_pieces*new_size - height)/(H_pieces-1))
        W_overlap = int((W_pieces*new_size - width)/ (W_pieces-1))
        all_splited_imgs = []
        for w in range(W_pieces):
            for h in range(H_pieces):
                #print(str(w)+" "+str(h))
                file_name = xmls[i][:-4]+'_'+str(w)+'_'+str(h)     
                #print(file_name)
                start_w = np.maximum(w*(new_size-W_overlap), 0)
                end_w = np.minimum(start_w+new_size, width)    
                start_h = np.maximum(h*(new_size-H_overlap), 0)
                end_h = np.minimum(start_h+new_size, height)
                source_img = cv2.imread(img_path)
                if(source_img.shape[0]>=source_img.shape[1]):
                    splited_img = source_img[start_h:end_h,start_w:end_w,:]
                else:
                    splited_img = source_img[start_w:end_w,start_h:end_h,:]
                img_details = {'img_data':splited_img, 'height':new_size, 'width':new_size, \
                           'bboxes':[], 'filename':file_name}
                
                has_box_iou_too_small = False
                for obj in get(root, 'object'):
                    category = get_and_check(obj, 'name', 1).text
                    category = category[:2]
                    if int(category) in commom_flaws:
                        bndbox = get_and_check(obj, 'bndbox', 1)
                        xmin = int(get_and_check(bndbox, 'xmin', 1).text) 
                        ymin = int(get_and_check(bndbox, 'ymin', 1).text)
                        xmax = int(get_and_check(bndbox, 'xmax', 1).text)
                        ymax = int(get_and_check(bndbox, 'ymax', 1).text)
                        assert(xmax > xmin)
                        assert(ymax > ymin)
                        if ymin<start_h and ymax>end_h:
                            ymin = start_h
                            ymax = end_h
                            
                        if xmin <start_w and xmax>end_w:
                            xmin = start_w
                            xmax = end_w
                        ## bbox in splited img / bbox in raw img
                        IoU = iou([start_w, start_h, end_w, end_h],
                                  [xmin, ymin, xmax, ymax])
                        if IoU<0.8:
                            has_box_iou_too_small = True
                            break
                        xmin = xmin - start_w
                        ymin = ymin - start_h
                        xmax = xmax - start_w
                        ymax = ymax - start_h
                        img_details['bboxes'].append({
                                'class':category, 'x1':xmin,
                                'y1':ymin, 'x2':xmax, 'y2':ymax})
                if(has_box_iou_too_small):
                    continue
                if(len(img_details['bboxes'])>0):
                    all_splited_imgs.append(img_details)
                else:
                    continue
 
        if(len(all_splited_imgs)>3):
            np.random.shuffle(all_splited_imgs)
            all_splited_imgs = all_splited_imgs[:3]
        for i in range(len(all_splited_imgs)):
            img_detail = all_splited_imgs[i]
            splited_img_data = img_detail['img_data']
            create_xml(img_detail, data_output_path)
            '''
            xml_path = os.path.join(data_output_path, img_detail['filename']+'.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in get(root, 'object'):
                category = get_and_check(obj, 'name', 1).text
                category = category[:2]
                assert (int(category) > 60) & (int(category) < 90)
                bndbox = get_and_check(obj, 'bndbox', 1)
                xmin = int(int(get_and_check(bndbox, 'xmin', 1).text))
                ymin = int(int(get_and_check(bndbox, 'ymin', 1).text))
                xmax = int(int(get_and_check(bndbox, 'xmax', 1).text))
                ymax = int(int(get_and_check(bndbox, 'ymax', 1).text))
                assert(xmax > xmin)
                assert(ymax > ymin)
                cv2.rectangle(splited_img_data,(xmin, ymin), (xmax, ymax), (0,255,255),1)           
                textLabel = category          
                textOrg = (xmin, ymin)
                cv2.putText(splited_img_data, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
            '''
            Image.fromarray(splited_img_data).save(data_output_path+sep+img_detail['filename']+'.jpg')

# 检测框的ID起始值
START_BOUNDING_BOX_ID = 1
# 类别列表无必要预先创建，程序中会根据所有图像中包含的ID来创建并更新
PRE_DEFINE_CATEGORIES = {}
def convert_xml_2_coco(xml_dir, json_file):
    '''
    :param xml_list: 需要转换的XML文件列表
    :param xml_dir: XML的存储文件夹
    :param json_file: 导出json文件的路径
    :return: None
    '''
    xml_list = [s for s in os.listdir(xml_dir) if s[-4:]=='.xml']
    # 标注基本结构
    json_dict = {"images":[],
                 "type": "instances",
                 "annotations": [],
                 "categories": []}
    categories = PRE_DEFINE_CATEGORIES
    bnd_id = START_BOUNDING_BOX_ID
    for line in xml_list:
        line = line.strip()
        print("Processing {}".format(line))
        # 解析XML
        xml_f = os.path.join(xml_dir, line)
        print(xml_f)
        tree = ET.parse(xml_f)
        root = tree.getroot()
        path = get(root, 'path')
        # 取出图片名字
        if len(path) == 1:
            filename = os.path.basename(path[0].text)
        elif len(path) == 0:
            filename = get_and_check(root, 'filename', 1).text
        else:
            raise NotImplementedError('%d paths found in %s'%(len(path), line))
        ## The filename must be a number
        #image_id = get_filename_as_int(filename)  # 图片ID
        image_id = filename[:-4]
        size = get_and_check(root, 'size', 1)
        # 图片的基本信息
        width = int(get_and_check(size, 'width', 1).text)
        height = int(get_and_check(size, 'height', 1).text)
        image = {'file_name': filename,
                 'height': height,
                 'width': width,
                 'id':image_id}
        json_dict['images'].append(image)
        ## Cruuently we do not support segmentation
        #  segmented = get_and_check(root, 'segmented', 1).text
        #  assert segmented == '0'
        # 处理每个标注的检测框
        for obj in get(root, 'object'):
            # 取出检测框类别名称
            category = get_and_check(obj, 'name', 1).text
            category = category[:2]
            tmp = int(category)
            assert tmp in commom_flaws
            # 更新类别ID字典
            if category not in categories:
                new_id = int(category)
                #print(new_id)
                categories[category] = new_id
            category_id = categories[category]
            bndbox = get_and_check(obj, 'bndbox', 1)
            xmin = int(get_and_check(bndbox, 'xmin', 1).text) - 1
            ymin = int(get_and_check(bndbox, 'ymin', 1).text) - 1
            xmax = int(get_and_check(bndbox, 'xmax', 1).text)
            ymax = int(get_and_check(bndbox, 'ymax', 1).text)
            assert(xmax > xmin)
            assert(ymax > ymin)
            o_width = abs(xmax - xmin)
            o_height = abs(ymax - ymin)
            annotation = dict()
            annotation['area'] = o_width*o_height
            annotation['iscrowd'] = 0
            annotation['image_id'] = image_id
            annotation['bbox'] = [xmin, ymin, o_width, o_height]
            annotation['category_id'] = category_id
            annotation['id'] = bnd_id
            annotation['ignore'] = 0
            # 设置分割数据，点的顺序为逆时针方向
            annotation['segmentation'] = [[xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin]]

            json_dict['annotations'].append(annotation)
            bnd_id = bnd_id + 1

    # 写入类别ID字典
    for cate, cid in categories.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cate}
        json_dict['categories'].append(cat)
    # 导出到json
    json_fp = open(json_file, 'w')
    json_str = json.dumps(json_dict,indent=1)
    json_fp.write(json_str)
    json_fp.close()
        

if __name__=="__main__":
    #resize_raw_imgs()
    split_resized_img()
    
    
    

