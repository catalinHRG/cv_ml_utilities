from json import load, dump
from glob import glob
from cv2 import imread, rectangle, resize, imwrite, putText, FONT_HERSHEY_SIMPLEX
from os.path import join, basename, splitext, dirname, isfile, isdir
from argparse import ArgumentParser
from datetime import datetime
from os import makedirs, mkdir
from imgaug import augmenters as iaa
from imgaug.augmenters import Affine, Fliplr, Flipud, AverageBlur, AdditiveGaussianNoise, Multiply, ContrastNormalization, Sequential, Invert, Sharpen
from imgaug.augmenters import ChangeColorspace, WithChannels, ChangeColorspace, Add
from imgaug import Keypoint, KeypointsOnImage
from tqdm import tqdm
from math import sqrt

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

from multiprocessing import cpu_count, Process, Manager
from time import time

'''

    use pca to end up with a 2d plot which illustrades the clusters for each class of object that has to be detected
    pca will be applied on the contents of the box for each object


'''

counter = 0


aug_pipe = Sequential([
        
            #ChangeColorspace(from_colorspace="BGR", to_colorspace="HSV"),
            #WithChannels(0, Add((-360, 360))),
            #ChangeColorspace(from_colorspace="HSV", to_colorspace="BGR"),
            #ContrastNormalization((0.9, 1.1))
            AdditiveGaussianNoise( loc=0, scale = (0.0, 0.001 * 255) )


])

#aug_pipe = ContrastNormalization((0.9, 1.1))

#aug_pipe = Sequential([ AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255)), Affine(shear = (-5, 5))])

aug_pipe = AdditiveGaussianNoise(loc=0, scale=(0.0, 0.001*255))

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

def handle_dyve_nonsense(class_name):

    return '0' if class_name == '' else class_name

def _stats_sweep_task(

    image_files, 
    annotation_boxes, 

    output_dir,
    
    class_colors, 
    class_names, 
    
    view_annotations,
    multiscale_anchor_generator,
    
    worker_id,
    return_dict, 
    

    ):

    def _draw_box(image, box, color = (0, 255, 0)):

        xmin, ymin, xmax, ymax, cls_label = box
    
        rectangle(image, (xmin, ymin), (xmax, ymax ), color, 4)
        w = xmax - xmin
        h = ymax - ymin
        putText(image, str(w) + ' ' + str(h), (w // 2, h // 2), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness = 1)
        putText(image, str(cls_label), (xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness = 1) 
            
        return image

    def _get_stats(box, ref_w, ref_h, image_file):

        bad_box = False

        xmin, ymin, xmax, ymax, _ = box

        width = xmax - xmin
        height = ymax - ymin

        ref = min(ref_w, ref_h)

        if width >= 5 and height >= 5:

            #print('@@@@@@@@', image_file)
            aspect_ratio = width / height

            if width == height: square_base_anchor_size = width
                
            else: square_base_anchor_size = sqrt(width * height)

            scale = square_base_anchor_size / ref
                
        else: 

            print(image_file)

            aspect_ratio = 1.0
            square_base_anchor_size = 4
            scale = 0.0001

            bad_box = True

        return aspect_ratio, scale, width, height, square_base_anchor_size, bad_box

    stats = Stats()

    for image_file, boxes in zip( tqdm(image_files), annotation_boxes ):

        per_image_class_instance_counts = dict()

        num_boxes = len(boxes)
        
        image = imread(image_file)
        
        if image is None: continue

        stats.update_box_count_per_image(num_boxes)

        h, w, _ = image.shape

        overlaps = []
        for i in range(num_boxes):

            row = []

            box = boxes[i]

            class_label = handle_dyve_nonsense(box[4])

            per_image_class_instance_counts[class_label] = per_image_class_instance_counts.get(class_label, 0) + 1

            for j in range(num_boxes):

                if i != j :

                    stats.update_overlaps( jaccard_index( box, boxes[j] ) )

            class_color = class_colors[ class_names.index(class_label) ]

            ref_w = w if multiscale_anchor_generator else 256 # for frcnn grid_anchor_generator implicit anchor width
            ref_h = h if multiscale_anchor_generator else 256 # bis anchor height

            aspect_ratio, scale, box_w, box_h, square_base_anchor_size, bad_box_flag = _get_stats(box, ref_w, ref_h, image_file)

            if bad_box_flag: continue

            stats.update_class_instance_count(class_label)
            stats.update_whs((box_w, box_h, class_label))
            stats.update_aspect_ratios(aspect_ratio)
            stats.update_scales(scale)
            stats.update_box_sizes(square_base_anchor_size)

            if view_annotations:  _draw_box(image, box, class_color)


        for class_name, count in per_image_class_instance_counts.items():

            stats.update_box_count_per_image_per_class(count)

        if view_annotations:
        
            output_file = join(output_dir, str(num_boxes) + '_' + worker_id + '_' + basename(image_file))

            imwrite(output_file, image)

    return_dict[worker_id] = stats

class Stats():

    def __init__(self):

        self.box_sizes = []
        self.aspect_ratios = []
        self.scales = []
        self.overlaps = [] 
        self.box_counts_per_image = [] 
        self.box_counts_per_image_per_class = []
        self.whs = []

        self.all_class_names = set()

        self.class_instance_count_per_dataset = dict()
        self.total_num_objects_per_dataset = 0

    def _iou(self, box, clusters):

        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])
        if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
            raise ValueError("Box has no area")

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = intersection / (box_area + cluster_area - intersection)

        return iou_

    def _avg_iou(self, boxes, clusters):

        return np.mean([np.max(self._iou(boxes[i], clusters)) for i in range(boxes.shape[0])])

    def _kmeans(self, num_clusters, dist=np.median):

        num_boxes = len(self.whs)

        _whs = []

        for wh in self.whs:

            w, h, _ = wh

            _whs.append((w,h))

        _whs = np.asarray(_whs)

        distances = np.empty((num_boxes, num_clusters))
        last_clusters = np.zeros((num_boxes,))

        np.random.seed()

        clusters = _whs[np.random.choice(num_boxes, num_clusters, replace=False)]

        while True:
            
            for box_id in range(num_boxes):

                distances[box_id] = 1 - self._iou(_whs[box_id], clusters)

            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                break

            for cluster in range(num_clusters):
                clusters[cluster] = dist(_whs[nearest_clusters == cluster], axis=0)

            last_clusters = nearest_clusters

        return clusters



    def update_whs(self, wh):
        
        self.whs.append(wh)
        self.all_class_names.add(wh[2])

    def update_box_sizes(self, box_size):

        self.box_sizes.append(box_size)

    def update_aspect_ratios(self, value):

        self.aspect_ratios.append(value)

    def update_scales(self, value):

        self.scales.append(value)

    def update_overlaps(self, value):

        self.overlaps.append(value)

    def update_box_count_per_image(self, value):

        self.box_counts_per_image.append(value)

    def update_box_count_per_image_per_class(self, value):

        self.box_counts_per_image_per_class.append(value)

    def update_class_instance_count(self, class_name):

        self.total_num_objects_per_dataset += 1
        self.class_instance_count_per_dataset[class_name] = self.class_instance_count_per_dataset.get(class_name, 0) + 1

    def get_class_instance_count(self):

        return self.class_instance_count_per_dataset


    def accumulate(self, stats):

        if not isinstance(stats, Stats):

            msg = 'stats param has to be an instance of class get_dataset_stats.Stats'
            raise ValueError(msg)

        self.box_sizes += stats.box_sizes
        self.aspect_ratios += stats.aspect_ratios
        self.scales += stats.scales
        self.overlaps += stats.overlaps 
        self.box_counts_per_image += stats.box_counts_per_image 
        self.box_counts_per_image_per_class += stats.box_counts_per_image_per_class
        
        for element in stats.whs: self.update_whs(element)

        for _class, count in stats.class_instance_count_per_dataset.items():

            self.class_instance_count_per_dataset[_class] = self.class_instance_count_per_dataset.get(_class, 0) + count

        self.total_num_objects_per_dataset += stats.total_num_objects_per_dataset
        
    def dump(self, output_dir, scale_bins, aspect_ratio_bins, log_to_stdout, num_box_clusters):

        if log_to_stdout:

            for class_label, count in self.class_instance_count_per_dataset.items():

                print(class_label, 'count:', count, ', fraction of the entire population:', count / self.total_num_objects_per_dataset)

        plt.hist(self.aspect_ratios, bins = aspect_ratio_bins, density = True)
        plt.savefig(join(output_dir, 'aspect_ratios.jpg'))
        plt.clf()

        plt.hist(self.box_sizes, bins = aspect_ratio_bins, density = True)
        plt.savefig(join(output_dir, 'square_box_sizes.jpg'))
        plt.clf()

        plt.hist(self.scales, bins = scale_bins, density = True)
        plt.savefig(join(output_dir, 'scales.jpg'))

        max_overlap = max(self.overlaps) if self.overlaps else 0
        max_box_count_per_image = max(self.box_counts_per_image)
        max_box_count_per_image_per_class = max(self.box_counts_per_image_per_class)
        
        biggest_box_sides, smallest_box_sides = [], []

        for wh in self.whs: 

            w, h, _ = wh

            biggest_box_sides.append( max(w, h) )
            smallest_box_sides.append( min(w, h) )

        biggest_box_side = max(biggest_box_sides)
        smallest_box_side = min(smallest_box_sides)

        clusters = self._kmeans(num_box_clusters)

        _whs = []

        for _wh in self.whs:

            w, h, class_label = _wh
            _whs.append((w,h))

        _whs = np.asarray(_whs)
        avg_iou = self._avg_iou(_whs, clusters)

        _square_box_sizes = []
        _aspect_ratios = []
        _shortest_sides = []
        _biggest_sides = []

        for cluster in clusters:

            w, h = cluster

            _square_box_size = int(sqrt(w * h))
            _aspect_ratio = w / h
            
            _shortest_sides.append(int(min(w,h)))
            _biggest_sides.append(int(max(w,h)))    
                    
            _square_box_sizes.append(_square_box_size)
            _aspect_ratios.append(_aspect_ratio)

        sorted_indices = np.argsort(_square_box_sizes)
        
        sorted_square_box_sizes = []
        sorted_aspect_ratios = []
        sorted_shortest_sides = []
        sorted_biggest_sides = []

        for idx in sorted_indices:

            sorted_square_box_sizes.append( _square_box_sizes[idx] )
            sorted_aspect_ratios.append(_aspect_ratios[idx])

            sorted_shortest_sides.append( _shortest_sides[idx] )
            sorted_biggest_sides.append(_biggest_sides[idx])

        per_class_box_stats = {}

        for class_name in self.all_class_names:

            _dict = {

                'widths': [],
                'heights': [],
                'square_box_sizes': [],
                'aspect_ratios': []

            }

            for wh in self.whs:

                if wh[2] == class_name:

                    _w, _h = wh[0:2]

                    _dict['widths'].append(_w)
                    _dict['heights'].append(_h)
                    _dict['square_box_sizes'].append(sqrt(_w * _h))
                    _dict['aspect_ratios'].append(_w / _h)

                
            _final_dict = {

                'width_mean': np.mean(_dict['widths']),
                'width_std': np.std(_dict['widths']),
                
                'height_mean': np.mean(_dict['heights']),
                'height_std': np.std(_dict['heights']),

                'square_box_size_mean': np.mean(_dict['square_box_sizes']),
                'square_box_size_std': np.std(_dict['square_box_sizes']),

                'aspect_ratio_mean': np.mean(_dict['aspect_ratios']),
                'aspect_ratios_std': np.std(_dict['aspect_ratios'])

            }

            per_class_box_stats[class_name] = _final_dict


        if log_to_stdout:

            print('Max overlap between boxes found in dataset is:', max_overlap)
            print('Max box count per image is:', max_box_count_per_image)
            print('Max box count per image per class is:', max_box_count_per_image_per_class)
            print('Average iou given the clusters of box sizes yielded by', str(num_box_clusters), 'clusters is:', avg_iou)
            print('Square anchor box sizes are:', sorted_square_box_sizes)
            print('Anchor box aspect ratios are:', sorted_aspect_ratios)
            print('Shortest sides associated with the square box anchor sizes are:', sorted_shortest_sides)
            print('Biggest sides associated with the square box anchor sizes are:', sorted_biggest_sides)
            print('Biggest box side is:', biggest_box_side)
            print('Smallest box side is:', smallest_box_side)

            for cls_name, values in per_class_box_stats.items():

                print('For class', cls_name, 'box stats are:')

                for stat_name, stat_value in values.items():

                    print(stat_name, ':', stat_value)

        _result = {}

        _result['population_distribution'] = {}

        with open(join(output_dir, 'box_stats.json'), 'w') as ad:

            for class_label, count in self.class_instance_count_per_dataset.items():

                _result['population_distribution'][class_label] = count / self.total_num_objects_per_dataset

            
            _result['max_box_count_per_image'] = max_box_count_per_image
            _result['max_box_count_per_image_per_class'] = max_box_count_per_image_per_class
            
            _result['average_iou_yielded_by_' + str(num_box_clusters) + '_clusters'] = avg_iou
            _result['max_overlap_between_boxes_found_in_dataset'] = max_overlap
            
            _result['square_anchor_box_sizes'] = sorted_square_box_sizes
            _result['anchor_box_aspect_ratios'] = sorted_aspect_ratios
            
            _result['biggest_box_side'] = biggest_box_side
            _result['smalelst_box_side'] = smallest_box_side

            _result['per_class_box_stats'] = per_class_box_stats

            #print(sorted_square_box_sizes, type(sorted_square_box_sizes))
            #print(sorted_shortest_sides, type(sorted_shortest_sides))
            #print(sorted_biggest_sides, type(sorted_biggest_sides))
            
            _result['shortest_anchor_sides'] = sorted_shortest_sides
            _result['biggest_anchor_sides'] = sorted_biggest_sides

            dump(_result, ad)


class Dataset():

    def __init__(self, root_dir, annot_file_suffix, balance_flag, include_masks):

        print('Creating dataset handle ...')

        self.JSON_ANNOTATION_DIR = 'ForTrainingAnnotations'
        self.JSON_ANNOT_KEY = 'Annotations'
        self.JSON_LABEL_KEY = 'Label'
        self.JSON_TOP_LEFT_KEY = 'PointTopLeft'
        self.JSON_TOP_RIGHT_KEY = 'PointTopRight'
        self.JSON_BOTTOM_LEFT_KEY = 'PointBottomLeft'
        self.JSON_BOTTOM_RIGHT_KEY = 'PointBottomRight'
        self.JSON_IS_BAD_IMAGE_KEY = 'IsBadImage'

        class_names = set()

        self._image_files = []
        self._all_annotations = []

        self._aspect_ratios = []
        self._scales = []

        self._image_formats = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', 'PNG', '.bmp', '.BMP', '.bitmap', '.BITMAP']
        self._annot_file_suffix = annot_file_suffix

        self._include_masks = include_masks

        self._balance_flag = balance_flag

        nodes = glob(join(root_dir, '*'))

        for node in nodes:

            if isfile(node):

                file_, _ext = splitext(node)

                if _ext in self._image_formats:

                    self._image_files.append(node)

                    annot_file = join( join( root_dir , 'ForTrainingAnnotations'), basename(file_) + annot_file_suffix + '.json')

                    _img = imread(node)
                    if _img is None: continue
                    
                    image_annotations = self._get_annots_from_json(_img, annot_file) # for negative examples, be them with missing annotation file or with empty list of annotations in a json file, an empty list will be returned

                    for box in image_annotations:

                        class_names.add( handle_dyve_nonsense(box[4]) )

                    self._all_annotations.append( image_annotations ) # the empty list for negative examples will be also added

        self._class_names = list(class_names)
        self._class_colors = self._get_class_colors( len(self._class_names) )
    
        print('Done !')

    def _distort(self, image, ground_truths, aug_pipe):

        if not ground_truths: return image, ground_truths

        det_aug = aug_pipe.to_deterministic()
                
        image = det_aug.augment_image(image)
                
        image_shape = image.shape
                
        keypoints_on_image = []
        keypoints = []
        bbox_class_labels = []
        result_gts = []

        for label in ground_truths:

            keypoints.append(Keypoint(x=label[0], y=label[1]))  # top left xmin, ymin
            keypoints.append(Keypoint(x=label[2], y=label[3]))  # bottom right xmax, ymax
            keypoints.append(Keypoint(x=label[0], y=label[3]))  # bottom left xmin, ymax
            keypoints.append(Keypoint(x=label[2], y=label[1]))  # top right xmax, ymin

            bbox_class_labels.append(label[4])

            keypoints_on_image.append(KeypointsOnImage(keypoints, shape=image_shape))
        
        keypoints_on_image = det_aug.augment_keypoints(keypoints_on_image)
        
        index = 0

        for keypoint in keypoints_on_image[0].keypoints: 

            if index % 4 == 0:
                        
                x1, y1 = keypoint.x, keypoint.y
                         
            if index % 4 == 1:
                        
                x2, y2 = keypoint.x, keypoint.y
                
            if index % 4 == 2:
                        
                x3, y3 = keypoint.x, keypoint.y
                        
            if index % 4 == 3:
                        
                x4, y4 = keypoint.x, keypoint.y
                result_gts.append( [x1, y1, x2, y2, x3, y3, x4, y4, bbox_class_labels[index // 4] ] ) # top left, bottom right, bottom left, top right and class_name

            index += 1
                    
        return image, result_gts

    def _get_class_colors(self, num_classes):

        ret = []
  
        r = int(random.random() * 256)
        g = int(random.random() * 256)
        b = int(random.random() * 256)

        step = 256 / num_classes
        
        for i in range(num_classes):
    
            r += step
            g += step
            b += step
    
            r = int(r) % 256
            g = int(g) % 256
            b = int(b) % 256
    
            ret.append((r,g,b)) 
  
        return ret

    def _get_annots_from_json(self, image, file_, scale_w = 1, scale_h = 1):

        image_annotations = []

        dst_root_dir = dirname(dirname(file_))

        mask_dir = join(dst_root_dir, 'masks')

        masks = False

        if isdir(mask_dir) and self._include_masks: 

            mask_dst_dir = join(dst_root_dir, 'mask_info')
            if not isdir(mask_dst_dir): mkdir(mask_dst_dir)

            mask_file_dst_dir = join(mask_dst_dir, splitext(basename(file_))[0])
            if not isdir(mask_file_dst_dir): mkdir(mask_file_dst_dir)

            imwrite(join(mask_file_dst_dir, splitext(basename(file_))[0] + '.jpg'), image)

            masks = True

        if not isfile(file_): return image_annotations

        else:

            with open(file_, 'r') as fd:

                json_data = load(fd)

            try:
            
                is_bad_image = json_data[self.JSON_IS_BAD_IMAGE_KEY]
            
            except KeyError:
            
                is_bad_image = False
            
            if is_bad_image:
            
                return None
            
            else:
            
                annot_dicts = json_data[self.JSON_ANNOT_KEY]
            
                for annot_dict in annot_dicts:
            
                    class_label = annot_dict[self.JSON_LABEL_KEY]

                    _detection_rect = annot_dict.get('DetectionRect', None)
                    if _detection_rect is None: 
                        _detection_rect = annot_dict.get('ClassificationRect', None)
                    if _detection_rect is None: 
                        _detection_rect = annot_dict

                    tl_xy_string = _detection_rect[self.JSON_TOP_LEFT_KEY]
                    tr_xy_string = _detection_rect[self.JSON_TOP_RIGHT_KEY]
                    bl_xy_string = _detection_rect[self.JSON_BOTTOM_LEFT_KEY]
                    br_xy_string = _detection_rect[self.JSON_BOTTOM_RIGHT_KEY]
        
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
                
                    _crop = image[int(ymin): int(ymax), int(xmin): int(xmax), ...]
                    
                    dst_dir = join(dst_root_dir, 'info')
                    if not isdir(dst_dir): mkdir(dst_dir)
                    dst_dir = join(dst_dir, class_label)
                    if not isdir(dst_dir): mkdir(dst_dir)

                    global counter
                    
                    if masks:

                        mask_file_name = _detection_rect.get('objMask', None)
                            
                        '''    

                        mask_file_name_split = splitext(mask_file_name)[0].split('_')

                        _mask_file_name = ''

                        num_splits = len(mask_file_name_split)

                        for i in range(num_splits - 1):

                            prefix = '_'
                            if i == 0: prefix = ''
                            
                            _mask_file_name += prefix + mask_file_name_split[i]

                        _mask_file_name += '.png'

                        '''

                        if mask_file_name:

                            mask_file = join(mask_dir, mask_file_name)
                            
                            _mask_file = imread(mask_file)

                            imwrite(join(mask_file_dst_dir, mask_file_name), _mask_file * 255)
                            
                    imwrite(join(dst_dir, splitext(basename(file_))[0] + '_' + str(counter) + '.jpg'), _crop)
                    counter += 1

                    image_annotations.append( [ int(scale_w * xmin), int(scale_h * ymin), int(scale_w * xmax), int(scale_h * ymax), class_label ] ) # obj_prob filled in there for consistency
    
            return image_annotations

    def _draw_box(self, image, box, color = (0, 255, 0)):

        xmin, ymin, xmax, ymax, cls_label = box
    
        rectangle(image, (xmin, ymin), (xmax, ymax ), color, 4)
        w = xmax - xmin
        h = ymax - ymin
        putText(image, str(w) + ' ' + str(h), (w // 2, h // 2), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness = 1)
        putText(image, str(cls_label), (xmin, ymin), FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness = 1) 
            
        return image

    def _get_stats(self, box, ref_w, ref_h):

        bad_box = False

        xmin, ymin, xmax, ymax, _ = box

        width = xmax - xmin
        height = ymax - ymin

        ref = min(ref_w, ref_h)

        if width >= 5 and height >= 5:

            aspect_ratio = width / height

            if width == height: square_base_anchor_size = width
                
            else: square_base_anchor_size = sqrt(width * height)

            scale = square_base_anchor_size / ref
                
        else: bad_box = True

        return aspect_ratio, scale, width, height, square_base_anchor_size, bad_box 

    def _balance_dataset(self, aug_pipe, class_instance_occurrences, balance_dataset_output_dir):

        balance_dataset_output_dir = join(balance_dataset_output_dir, 'balance')
        if not isdir(balance_dataset_output_dir): makedirs(balance_dataset_output_dir)

        max_count = 0
        max_count_class_name = 'bitches'

        workloads, class_baselines = {}, {}

        '''

            Workloads dict layout:

                {
    
                    "class1_name": 

                        {
                            
                            amount_to_balance": int,
                            "baseline": {'images': list, 'boxes': list}
                            "aug_pipe": maybe as a feature to be implementated later on, an imgaug pipe describing how the augmentations should be carried out for this particular class
                        },

                    "class2_name": 

                        {
                            
                            "amount_to_balance": int,
                            "baseline": {'images': list, 'boxes': list}
                            "aug_pipe": maybe as a feature to be implementated later on, an imgaug pipe describing how the augmentations should be carried out for this particular class
                        },

                    etc...

                }

        '''

        for class_name, count in class_instance_occurrences.items():

            if count > max_count: 

                max_count = count
                max_count_class_name = class_name

        for image_file, boxes in zip( self._image_files, self._all_annotations ): # compile a list with all images with ONE instance of a particular class

            if len(boxes) == 1:

                box_class = boxes[0][4]

                # dunno if dict.get() returns a reference to the content found at 'key', if it does, there is no need for the temp_var shinanigan

                class_baseline = class_baselines.get(box_class, {}) 
                
                _image_files_so_far = class_baseline.get('images', [])
                _image_files_so_far.append(image_file)

                class_baseline['images'] = _image_files_so_far
                
                _image_file_boxes_so_far = class_baseline.get('boxes', [])
                _image_file_boxes_so_far.append(boxes) # keeping [ box ] per image, for consistency, even tho it's just one

                class_baseline['boxes'] = _image_file_boxes_so_far

                class_baselines[box_class] = class_baseline
                
        for class_name, count in class_instance_occurrences.items():

            if class_name == max_count_class_name: continue

            workload = {}
            
            workload['amount_to_balance'] = int((max_count - count) * 1.0) # amount needed to balance dataset with respect to this class
            
            _baseline = class_baselines.get(class_name, None) # images with respective boxes, that will be used for balancing

            if _baseline is None:

                print('There was no image containing exactly one instance of class', class_name, 'in the dataset. For the moment this class will not be balanced out')
                continue

            workload['baseline'] = _baseline
            workloads[class_name] = workload

        for class_name, workload in workloads.items():

            amount_to_balance = workload['amount_to_balance']
            
            images = workload['baseline']['images']
            boxes = workload['baseline']['boxes']

            num_images = len(images)

            amount_per_image = max(amount_to_balance // num_images, 1) # no need for exact quantity

            file_counter = 0

            for image_file, boxes in zip(images, boxes):

                image = imread(image_file)

                if image is None: continue

                for i in range(amount_per_image):

                    augmented_image, adjusted_boxes = self._distort(image, boxes, aug_pipe) 
    
                    annotations = []

                    # each box is defined by xy coords for each corner --> TL, BR, BL, TR

                    for box in adjusted_boxes:

                        box_dict = {

                            'Type': 'manual',
                            "Angle": '0',
                            "RealAngle": '0',
                            self.JSON_LABEL_KEY: handle_dyve_nonsense(box[8]),
                            self.JSON_TOP_LEFT_KEY: str(box[0]) + ',' + str(box[1]),
                            self.JSON_BOTTOM_RIGHT_KEY: str(box[2]) + ',' + str(box[3]),
                            self.JSON_BOTTOM_LEFT_KEY: str(box[4]) + ',' + str(box[5]),
                            self.JSON_TOP_RIGHT_KEY: str(box[6]) + ',' + str(box[7])

                        }

                        annotations.append(box_dict)

                    annot_dict = {

                        'EndOfAction': False,
                        self.JSON_IS_BAD_IMAGE_KEY: False,
                        self.JSON_ANNOT_KEY: annotations

                    }

                    base_ = "balance_" + str(file_counter)
                
                    _image_file = base_ + '.jpg'
                    _annot_file = base_ + self._annot_file_suffix + '.json'

                    prefix = str(datetime.now())

                    balance_image_file = join(balance_dataset_output_dir, prefix + '_' + _image_file)

                    annot_dir = join(balance_dataset_output_dir, 'ForTrainingAnnotations')
                
                    if not isdir(annot_dir): makedirs(annot_dir)

                    annot_file = join(annot_dir, prefix + '_' + _annot_file)

                    imwrite(balance_image_file, augmented_image)

                    with open(annot_file, 'w') as hd:

                        dump(annot_dict, hd)

                    file_counter += 1


    def query_dataset(

        self, 
        output_dir, 
        view_annotations, 
        multiscale_anchor_generator, 
        scale_bins, 
        aspect_ratio_bins, 
        log_to_stdout,
        num_box_clusters

        ):

        print('Running sweep ...')
        
        stats = Stats()

        max_workers = cpu_count()

        image_workloads, annotation_workloads = [ [] for _ in range(max_workers) ], [ [] for _ in range(max_workers) ]
        worker_id = 0

        for image_file, boxes in zip( tqdm(self._image_files), self._all_annotations ):

            image_workloads[worker_id].append(image_file)
            annotation_workloads[worker_id].append(boxes)
            worker_id += 1

            if worker_id == max_workers: worker_id = 0

        manager = Manager()
        return_dict = manager.dict()

        workers = []
        
        for idx, workload_tuple in enumerate(zip(image_workloads, annotation_workloads)):

            image_workload, annotation_workload = workload_tuple

            workers.append(

                Process(

                    target = _stats_sweep_task, 
                    
                    args = (

                        image_workload, 
                        annotation_workload,
                        output_dir,
                        self._class_colors,
                        self._class_names,
                        view_annotations, 
                        multiscale_anchor_generator,
                        'Worker ' + str(idx),
                        return_dict
                    )
                )
            )

        start = time()
        for worker in workers: worker.start()

        for worker in workers: worker.join()
        print('Done running stats sweep in', time() - start, 'seconds.')

        if self._balance_flag: 

            global aug_pipe
            self._balance_dataset(aug_pipe, stats.get_class_instance_count(), output_dir)

        for worker_id, partial_stats in return_dict.items(): 

            #print( len(partial_stats.box_sizes) )
            stats.accumulate(partial_stats)
            #print( len(stats.box_sizes) )

        stats.dump(output_dir, scale_bins, aspect_ratio_bins, log_to_stdout, num_box_clusters)

        print('Done !')
        
        return stats


def run(args):

    input_dir = args.dataset_input_dir

    if not isdir(input_dir): 
        
        print('Input directory', input_dir, 'does not exist, aborting ...')
        exit()

    output_dir = join(args.dataset_input_dir, 'info')
    if not isdir(output_dir): mkdir(output_dir)

    dataset_handle = Dataset(args.dataset_input_dir, args.annot_file_suffix, args.balance, args.include_masks)

    _ = dataset_handle.query_dataset(

        output_dir, 
        args.view_annots, 
        args.multiscale_grid_anchor_generator, 
        args.scale_bins, 
        args.aspect_ratios_bins, 
        args.log,
        args.num_box_clusters
    )

def main():

    argp = ArgumentParser(description = 'Python3 utility to visualize dataset annotations or get statistics about them')
    
    argp.add_argument('--dataset_input_dir', '-i', required = True, help = 'path to root dir of DVnnotationTool dataset format')
    argp.add_argument('--view_annots', '-v', action = 'store_true', help = 'mention this arg in order to get annotations, ommiting it will dump only the statistics for the dataset')
    argp.add_argument('--scale_bins', type = int, default = 5, help = 'Number of bins for the scale histogram')
    argp.add_argument('--aspect_ratios_bins', type = int, default = 5, help = 'Number of bins for the aspect ratios histogram')
    argp.add_argument('--balance', action = 'store_true', help = 'mention this arg if you intend to balance dataset in terms of class occurences using additive gaussian noise for augamentations')
    argp.add_argument('--multiscale_grid_anchor_generator', '-mgag', action = 'store_true', help = 'wether anchor stats should be computed relative to GODAPI MGAG i.e. SSD, else for GODAPI GAG i.e. FRCNN ')
    argp.add_argument('--log', '-l', action = 'store_true', help = 'whether to log to console')
    argp.add_argument('--num_box_clusters', '-k', type = int, default = 1, help = 'number of box clusters to use when computing kmeans for anchor boxes')
    argp.add_argument('--annot_file_suffix', '-as', default = '_forTraining', help = 'suffix for annot file to look for')
    argp.add_argument('--include_masks', '-m', action = 'store_true', help = 'if present, it will include masks in the info section of this utilities output, for reviewing them visually')

    args = argp.parse_args()

    run(args)

if __name__ == '__main__':

    main()
