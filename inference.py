
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import mmcv
import numpy as np
import glob
import cv2
import os
import time
import json

from argparse import ArgumentParser

# submission_dir = './work_dirs/submission'
img_path = '../guangdong1_round1_testA_20190818/'
# config_file = './configs/custom/faster_rcnn_r101_fpn_dcn_ohem.py'
# # download the checkpoint from model zoo and put it in `checkpoints/`
# checkpoint_file = './work_dirs/baseline_faster_rcnn_dcn_ohem/latest.pth'

def infer(config_file, checkpoint_file, submission_dir):
	# build the model from a config file and a checkpoint file
	model = init_detector(config_file, checkpoint_file, device='cuda:0')

	imglist = glob.glob(img_path+'*.jpg')

	imglist.sort()

	num_images = len(imglist)

	result_list = []
	for i in range(num_images):
		print('processing {}/{}'.format(i+1,num_images),end='\r')
		#获取提交的图片名称
		filename = os.path.basename(imglist[i])
		list_rect = []
		im = cv2.imread(imglist[i])
		#模型获取box的相关信息,具体代码可能有所不同
		boxes = inference_detector(model, im)
		 

		#boxes,classes=model(im)....
		if boxes is not None:
			for j in range(len(boxes)):
				if boxes[j].size > 0:
					for box in boxes[j]:
						result = {}
						x,y,w,h,score = box
						category=j+1
						result["name"] = filename
						result["category"] = category
						result["bbox"] = [round(float(x),2), round(float(y), 2), round(float(w), 2), round(float(h), 2)]
						result["score"] = float(score)
						#print(result)
						result_list.append(result)
	print('Finished!')
	save_path = "%s/result_%s.json" % (submission_dir, time.strftime("%Y%m%d%H%M"))
	json.dump(result_list, open(save_path, 'w'), indent=4, separators=(',', ': '))

def main():
    parser = ArgumentParser(description='Inference the test_dataset')
    parser.add_argument('config', help='config file path')
    parser.add_argument('pth', help='pth file path')
    parser.add_argument(
        '--output',
        type=str,
        default='work_dirs/submission',
        help='output path')
    args = parser.parse_args()
    #cfg = mmcv.Config.fromfile(args.config)
    #test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
    infer(args.config, args.pth, args.output)


if __name__ == '__main__':
    main()
