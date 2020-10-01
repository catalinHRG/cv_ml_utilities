import tensorflow as tf
import numpy as np
import random
import colorsys
import re

from cv2 import imshow, waitKey, rectangle, imwrite, imread, IMREAD_GRAYSCALE, IMREAD_COLOR, putText, FONT_HERSHEY_SIMPLEX, cvtColor, COLOR_BGR2RGB, COLOR_RGB2BGR
from time import time
from argparse import ArgumentParser
from glob import glob
from os.path import splitext, join, basename, isdir, isfile
from os import mkdir
from tqdm import tqdm
from math import sqrt
from sys import float_info
from json import load
from copy import deepcopy

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util as lmu

'''

	Current implementation is a bit inefficient, since instead of batching up 'x' ammount of images to be
	forwarded throught the graph, it forwards one image at a time.

	A better implementation would be to look up the system resources, see wether 'x' ammount would be possible for
	the given graph, forward them and interpret output for 'x' images, and dump the images on disk with bboxes drawn
	on them.

'''

IMG_FORMAT = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.bmp', '.BMP', '.bitmap', '.BITMAP']

INPUT_TENSOR_NAME = 'image_tensor:0'

OUTPUT_CONF_TENSOR_NAME = 'detection_scores:0'
OUTPUT_BOX_TENSOR_NAME = 'detection_boxes:0'
OUTPUT_CLASS_TENSOR_NAME = 'detection_classes:0'

JSON_ANNOTATION_DIR = 'ForTrainingAnnotations'
JSON_ANNOT_KEY = 'Annotations'
JSON_LABEL_KEY = 'Label'
JSON_TOP_LEFT_KEY = 'PointTopLeft'
JSON_TOP_RIGHT_KEY = 'PointTopRight'
JSON_BOTTOM_LEFT_KEY = 'PointBottomLeft'
JSON_BOTTOM_RIGHT_KEY = 'PointBottomRight'
JSON_IS_BAD_IMAGE_KEY = 'IsBadImage'

def interval_overlap(interval_a, interval_b):
    
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2, x4) - x3

def jaccard_index(box1, box2):
    
    # bbox --> [xmin, ymin, xmax, ymax, ..]
    
    intersect_w = interval_overlap( [ box1[0], box1[2] ], [ box2[0], box2[2] ] )
    intersect_h = interval_overlap( [ box1[1], box1[3] ], [ box2[1], box2[3] ] )  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    
    union = w1 * h1 + w2 * h2 - intersect
    
    return float(intersect) / union

def get_metrics(image_gts, image_predictions, iou_threshold):

	# 'predictions' are already filterd by obj prob threshold

	true_pos, false_pos, false_neg = 0, 0, 0
		
	num_boxes = len(image_predictions)
	num_gts = len(image_gts)

	box_gt_matched_pairs = []

	for gt in image_gts:

		ious = []

		for box in image_predictions:

			if box[4] == gt[4]: # if they are the same class

				ious.append(jaccard_index(gt, box))
			
			else: # even if they match, iou should be 0, cuz it doesnt count as a match, since class differ, mostly doing this cuz i need to have the indices to match in order for the poping to work

				ious.append(0)

		if ious: best_iou = max(ious)
		else: best_iou = 0 

		if best_iou > iou_threshold:

			true_pos += 1

			_id = ious.index(best_iou)
			image_predictions.pop(_id)# we don't want one good prediction to be counted for more than one ground truth

	#print('num_boxes', num_boxes, 'true_pos', true_pos)
	false_pos = max(num_boxes, true_pos) - num_gts
	#print('false_pos', false_pos)
	#print('num_gts', num_gts, 'true_pos', true_pos)
	false_neg = num_gts - min(num_gts, true_pos)
	#print('false_neg', false_neg)

	return true_pos, false_pos, false_neg

def get_annots_from_json(file_):

        image_annotations = []
        
        if not isfile(file_): return image_annotations

        else:
            
            with open(file_, 'r') as handle:

            	json_data = load(handle)

            annot_dicts = json_data[JSON_ANNOT_KEY]
            
            for annot_dict in annot_dicts:
            
                class_label = annot_dict[JSON_LABEL_KEY]
                
                if class_label == '':
                    
                    class_label = '0'

                tl_xy_string = annot_dict[JSON_TOP_LEFT_KEY]
                tr_xy_string = annot_dict[JSON_TOP_RIGHT_KEY]
                bl_xy_string = annot_dict[JSON_BOTTOM_LEFT_KEY]
                br_xy_string = annot_dict[JSON_BOTTOM_RIGHT_KEY]
        
                x_string, y_string = tl_xy_string.split(',')
                tl_x, tl_y = float(x_string), float(y_string)
                
                x_string, y_string = tr_xy_string.split(',')
                tr_x, tr_y = float(x_string), float(y_string)
                
                x_string, y_string = bl_xy_string.split(',')
                bl_x, bl_y = float(x_string), float(y_string)
                
                x_string, y_string = br_xy_string.split(',')
                br_x, br_y = float(x_string), float(y_string)
    
                xmin = np.min( [ tl_x, tr_x, bl_x, br_x ] )
                xmax = np.max( [ tl_x, tr_x, bl_x, br_x ] )

                ymin = np.min( [ tl_y, tr_y, bl_y, br_y ] )
                ymax = np.max( [ tl_y, tr_y, bl_y, br_y ] )
                
                image_annotations.append( [ int(xmin), int(ymin), int(xmax), int(ymax), class_label, 0 ] ) # for consistency adding dummy value as cls_index
    
        return image_annotations

def get_class_colors(num_classes):

    HSV_tuples = [ (x * 1.0 / num_classes , 0.5, 0.5 ) for x in range(num_classes + 20)]

    class_rgb_tuples = []
    
    for rgb in HSV_tuples:
    
        rgb = map(lambda x: int(x * 255 * sqrt(x)), colorsys.hsv_to_rgb(*rgb))
        class_rgb_tuples.append(tuple(rgb))
    
    return [(255, 0, 0)] * 3

def draw_boxes(image, bboxes, class_color_map, color = None, thickness = 1):

	for bbox in bboxes :
               
		xmin, ymin, xmax, ymax, cls_name, cls_idx = bbox
		if color is None: 

			thickness = 1
			color = (255, 0, 0)


		rectangle(image, (xmin, ymin), (xmax, ymax), color, thickness)  
		putText(image, cls_name, (xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.2, (255, 255, 255), thickness = 1) 

	return image

parser = ArgumentParser(description = 'Visual assesment of Google Object Detection API generated models')

parser.add_argument('--model_file', '-m', required = True, help = 'Binary protobuf model to make predictions with')
parser.add_argument('--input_dir', '-i', required = True, help = 'Root directory for images to make predictions on')
parser.add_argument('--output_dir', '-o', required = True, help = 'Output directory')
parser.add_argument('--obj_threshold', '-obj', type = float, default = 0.5, help = 'Object probability thresold to filter out proposals. Note: boxes get filtered first by the embeded score threshold value within the model file.')
parser.add_argument('--iou_threshold', '-iou', type = float, default = 0.8, help = 'Jaccard Index threshold which dictates what is a true pos and what is not')
parser.add_argument('--labelmap_file', '-lm', required = True, help = 'used for getting class index, will take into account ssd implicit background class being at index 0')
parser.add_argument('--gray', action = 'store_true', help = 'Flag to read image as grayscale or rgb, just mention it if images are to be read in grayscale')

args = parser.parse_args()

def main():

	if not isdir(args.input_dir):

		print(args.input_dir, 'does not exist')
		exit()

	if not isdir(args.output_dir):

		print(args.output_dir, 'does not exist !!')
		exit()

	if not isfile(args.labelmap_file): 

		print('Could not find ', args.labelmap_file)
		exit()

	labelmap = lmu.create_category_index_from_labelmap(args.labelmap_file)

	keys = labelmap.keys()

	string_names = []

	for key in keys:

		string_names.append(labelmap[key]['name'])

	with tf.gfile.GFile(args.model_file, 'rb') as f: # tf 1
	#with tf.io.gfile.GFile(args.model_file, 'rb') as f: # tf 2

		graph_def = tf.GraphDef() # tf 1
		#graph_def = tf.compat.v1.GraphDef() # tf 2
		
		graph_def.ParseFromString(f.read())

	class_label_map = ['background'] + string_names
	class_color_map = get_class_colors(len(class_label_map))
	random.shuffle(class_color_map)

	config = tf.ConfigProto() # tf 1
	#config = tf.compat.v1.ConfigProto() # tf 2
	
	config.gpu_options.allow_growth = True

	with tf.Session(config = config) as sess: # tf 1
	#with tf.compat.v1.Session(config = config) as sess: # tf 2

		tf.import_graph_def(graph_def, name = '')

		g = tf.get_default_graph() # tf 1
		#g = tf.compat.v1.get_default_graph() # tf 2

		_input  = g.get_tensor_by_name(INPUT_TENSOR_NAME)     
		_scores = g.get_tensor_by_name(OUTPUT_CONF_TENSOR_NAME)
		_classes = g.get_tensor_by_name(OUTPUT_CLASS_TENSOR_NAME)
		_boxes = g.get_tensor_by_name(OUTPUT_BOX_TENSOR_NAME)

	content = glob(join(args.input_dir, '*'))

	images = []

	for item in content:

		if isdir(item): continue

		name_, _ext = splitext(item)
		if _ext not in IMG_FORMAT: continue

		images.append(item)

	annot_dir = join(args.input_dir, 'ForTrainingAnnotations')
				
	if isdir(annot_dir): 

		compute_metrics = True

	else: 

		compute_metrics = False
	
	imread_mode = IMREAD_GRAYSCALE if args.gray else IMREAD_COLOR

	with tf.Session() as sess: # tf 1
	#with tf.compat.v1.Session() as sess: # tf 2

		total = 0
		cnt = 0
		cnt_ = 0

		start_counting = False # in order to skip the first sess.run(...), when it also caches the graph, thus taking longer than the actual inference

		precisions, recalls, f1_scores = [], [], []

		for image in tqdm(images):

			image_, _ext = splitext(image)

			img = imread(image, imread_mode)
			if img is None: continue
			img = cvtColor(img, COLOR_BGR2RGB)

			if args.gray: img = np.expand_dims(img, 2) # explicit rank 3 since imread for IMREAD_GRASCALE mode yields a Matrix I.E. (height, width) instead of a Tensor (height, width, channels)

			img_h, img_w, _ = img.shape

			_img = np.expand_dims(img, 0) # put img in a batch, since the graph expects it to be so

			start = time()

			objectness, class_ids, boxes = sess.run([_scores, _classes, _boxes], feed_dict = {_input: _img} ) 
			
			#print(len(objectness), len(class_ids), len(boxes))
			#print(objectness, class_ids)

			img = cvtColor(img, COLOR_RGB2BGR)

			if start_counting:

				total += (time() - start)
				cnt += 1

			else:

				start_counting = True

			relevant_boxes = []
			cls_idxs = []

			annot_file = join(annot_dir, basename(image_) + '_forTraining.json')
			
			for confidence, cls_index, rel_box in zip(objectness[0], class_ids[0], boxes[0]): # since we gave it a batch of images, it returns a batch of 1 results thus the 0th indexing

				if confidence > args.obj_threshold:

					_ymin, _xmin, _ymax, _xmax = rel_box # this is the coord order per box used in godapi
					
					xmin = int( _xmin * img_w )
					ymin = int( _ymin * img_h )
					xmax = int( _xmax * img_w )
					ymax = int( _ymax * img_h )

					crops_dst = join(args.output_dir, 'class_crops')
					if not isdir(crops_dst): mkdir(crops_dst)

					_dst_dir_ = join(crops_dst, class_label_map[int(cls_index)])
					if not isdir(_dst_dir_): mkdir(_dst_dir_)

					_crop = img[ymin:ymax, xmin:xmax, ...]
					#imwrite(join(_dst_dir_, str(cnt_) + '_' + basename(image)), _crop)
					imwrite(join(_dst_dir_, basename(image)), _crop)
					cnt_ += 1

					abs_box = [xmin, ymin, xmax, ymax, class_label_map[int(cls_index)], int(cls_index) ] # switching to my standard coord order per box

					relevant_boxes.append(abs_box)

			_dir = ''

			true_pos, gts, false_pos = None, [], None

			if compute_metrics:

				gts = get_annots_from_json(annot_file)

				draw_boxes(img, gts, class_color_map, color = (0, 0, 0))

				true_pos, false_pos, false_neg = get_metrics(gts, deepcopy(relevant_boxes), args.iou_threshold)

				if gts: 
					
					precision = 1 if false_pos == 0 else true_pos / ( true_pos + false_pos )
					precisions.append( precision ) 
 
					recall = 1 if false_neg == 0 else true_pos / ( true_pos + false_neg ) 
					recalls.append( recall )

					#f1_score = true_pos / ( true_pos + ( (false_pos + false_neg) / 2) )
					#f1_scores.append( f1_score )

				if true_pos == len(gts) and false_pos == 0: 

					#print('true pos folder',true_pos, len(gts), false_pos)
					_dir = 'true_pos'

				else: 
					
					#print('false folder ', true_pos, len(gts), false_pos)
					_dir = 'false_pos_false_neg' 

			draw_boxes(img, relevant_boxes, class_color_map )
			_result_dir = join(args.output_dir, _dir)

			if not isdir(_result_dir): mkdir(_result_dir)

			name_ = 'True pos ' + str(true_pos) + ' -- ' + 'Ground truths ' + str(len(gts)) + ' -- ' + 'False pos ' + str(false_pos) + ' -- '
			if _dir == 'false_pos_false_neg':

				imwrite( join(_result_dir, name_ + basename(image) ), img)

	if compute_metrics:

		print('Average precision : ', sum(precisions) / len(precisions))
		print('Average recall : ', sum(recalls) / len(recalls))
		# print('Average F1 score: ', sum(f1_scores) / len(f1_scores))

	print('Average inference time in milliseconds: ', ( total / cnt ) * 1000 )

if __name__ == '__main__':

	main()
