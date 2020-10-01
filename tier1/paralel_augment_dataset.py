from imgaug.augmenters import Sequential, Affine, Invert, Fliplr, Flipud
from datetime import datetime
from glob import glob
from os import makedirs
from argparse import ArgumentParser
from json import load, dump
from tqdm import tqdm
from time import time
from sys import path as PYTHONPATH

from imgaug import Keypoint, KeypointsOnImage

from cv2 import imread, imwrite, IMREAD_GRAYSCALE, IMREAD_COLOR

from os.path import join, isfile, isdir, splitext, basename

from multiprocessing import cpu_count, Process

import numpy as np

JSON_ANNOTATION_DIR = 'ForTrainingAnnotations'
JSON_ANNOT_KEY = 'Annotations'
JSON_LABEL_KEY = 'Label'
JSON_TOP_LEFT_KEY = 'PointTopLeft'
JSON_TOP_RIGHT_KEY = 'PointTopRight'
JSON_BOTTOM_LEFT_KEY = 'PointBottomLeft'
JSON_BOTTOM_RIGHT_KEY = 'PointBottomRight'
JSON_IS_BAD_IMAGE_KEY = 'IsBadImage'


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
                
                image_annotations.append( [ int(xmin), int(ymin), int(xmax), int(ymax), class_label ] )
    
        return image_annotations

def distort(image, ground_truths, aug_pipe):

    truncated_box =  False
        
    det_aug = aug_pipe.to_deterministic()
            
    image = det_aug.augment_image(image)
            
    if not ground_truths: return image, ground_truths, truncated_box
    
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

    image_h, image_w = image_shape[0:2]

    for keypoint in keypoints_on_image[0].keypoints: 

        if index % 4 == 0:
                    
            x1, y1 = keypoint.x, keypoint.y
                     
        if index % 4 == 1:
                    
            x2, y2 = keypoint.x, keypoint.y
            
        if index % 4 == 2:
                    
            x3, y3 = keypoint.x, keypoint.y
                    
        if index % 4 == 3:
                    
            x4, y4 = keypoint.x, keypoint.y

            _x1 = max(x1, 0)
            _x1 = min(image_w, _x1)

            if _x1 != x1: truncated_box = True

            _x2 = max(x2, 0)
            _x2 = min(image_w, _x2)

            if _x2 != x2: truncated_box = True

            _x3 = max(x3, 0)
            _x3 = min(image_w, _x3)

            if _x3 != x3: truncated_box = True

            _x4 = max(x4, 0)
            _x4 = min(image_w, _x4)

            if _x4 != x4: truncated_box = True

            _y1 = max(y1, 0)
            _y1 = min(_y1, image_h)

            if _y1 != y1: truncated_box = True

            _y2 = max(y2, 0)
            _y2 = min(_y2, image_h)

            if _y2 != y2: truncated_box = True

            _y3 = max(y3, 0)
            _y3 = min(_y3, image_h)

            if _y3 != y3: truncated_box = True

            _y4 = max(y4, 0)
            _y4 = min(_y4, image_h)

            if _y4 != y4: truncated_box = True

            xmin, ymin, xmax, ymax = min(_x1, _x2, _x3, _x4), min(_y1, _y2, _y3, _y4), max(_x1, _x2, _x3, _x4), max(_y1, _y2, _y3, _y4)

            box_width = xmax - xmin
            box_height = ymax - ymin

            box_area = box_width * box_height

            #if box_area < ((image_w * image_h) * 0.01): 

                #print('Found a box with less than 0.01 of the image area... skipping')
                #continue

            result_gts.append( [_x1, _y1, _x2, _y2, _x3, _y3, _x4, _y4, bbox_class_labels[index // 4] ] ) # top left, bottom right, bottom left, top right and class_name

        index += 1
              
    return image, result_gts, truncated_box

def _task(image_files, image_annotations, aug_pipe, image_file_prefix, output_dir, num_augs_per_image, read_gray_scale):

    file_counter = 0

    for image_file, image_annotation in zip(tqdm(image_files), image_annotations):

        read_mode = IMREAD_GRAYSCALE if read_gray_scale else IMREAD_COLOR # BGR
        image = imread(image_file, read_mode)

        max_scale = 0.1
        min_scale = max_scale / num_augs_per_image

        scales = [

            ( 

                min_scale * (i + 1),
                - ( min_scale * (i + 1) ), 

            ) for i in range(num_augs_per_image)


        ]

        for i in range(num_augs_per_image):

            _scales = scales[i]

            #for _scale in _scales:
            for i in range(1):
                
                augmented_image, adjusted_boxes, truncated_box_flag = distort(image, image_annotation, aug_pipe) # boxes are also cliped to image

                #if not adjusted_boxes: continue
                
                #if truncated_box_flag: continue # atleast one box to be truncated is needed for this to trigger

                # each box is defined by xy coords for each corner --> TL, BR, BL, TR

                annotations = []
                
                for box in adjusted_boxes:

                    box_dict = {

                        'Type': 'manual',
                        "Angle": '0',
                        "RealAngle": '0',
                        JSON_LABEL_KEY: box[8],
                        JSON_TOP_LEFT_KEY: str(box[0]) + ',' + str(box[1]),
                        JSON_BOTTOM_RIGHT_KEY: str(box[2]) + ',' + str(box[3]),
                        JSON_BOTTOM_LEFT_KEY: str(box[4]) + ',' + str(box[5]),
                        JSON_TOP_RIGHT_KEY: str(box[6]) + ',' + str(box[7])

                    }

                    annotations.append(box_dict)
                    
                annot_dict = {

                    'EndOfAction': False,
                    JSON_IS_BAD_IMAGE_KEY: False,
                    JSON_ANNOT_KEY: annotations

                }

                base_ = image_file_prefix + "_" + str(file_counter)
            
                _image_file = base_ + '.jpg'
                _annot_file = base_ + '_forTraining.json'

                prefix = str(datetime.now())

                image_file = join(output_dir, prefix + '_' + _image_file)

                annot_dir = join(output_dir, 'ForTrainingAnnotations')
            
                if not isdir(annot_dir): makedirs(annot_dir)

                annot_file = join(annot_dir, prefix + '_' + _annot_file)

                imwrite(image_file, augmented_image)

                with open(annot_file, 'w') as hd:

                    dump(annot_dict, hd)

                file_counter += 1


### basic augs

def _rotate(n):

    pipe = Sequential( [ Affine( rotate = n ) ] )
    
    return pipe

def _negative():

    pipe = Sequential( [ Invert( 1.0, per_channel = 1.0 ) ] )

    return pipe

def _flip_lr():

    pipe = Sequential( [ Fliplr(1.0) ] )

    return pipe

def _flip_ud():

    pipe = Sequential( [ Flipud(1.0) ] )

    return pipe

###

def run(args):

    _args = [args.rotate_n != 0, args.negative_filter, args.flip_lr, args.flip_ud]

    counter = 0

    for _arg in _args:

        if _arg: counter += 1

    if counter > 1:

        msg = 'One base augmentation strategy can be specified at a time, ' + str(counter) + ' were given.'
        raise ValueError(msg)

    rotate_n = args.rotate_n != 0
    
    base_augs = rotate_n or args.negative_filter or args.flip_lr or args.flip_ud

    if base_augs:

        if rotate_n:

            pipe = _rotate(args.rotate_n)

        elif args.negative_filter:

            pipe = _negative()

        elif args.flip_lr:

            pipe = _flip_lr()

        elif args.flip_ud:

            pipe = _flip_ud()

        else:

            msg = 'wtf happened ?!?!'
            raise ValueError(msg)

    else:

        if not isdir(args.augmentation_pipe_root_dir):

            msg = args.augmentation_pipe_root_dir + ' does not exist or is not a directory.'
            raise IOError(msg)

        AUG_PIPE_FILE_NAME = 'aug_pipe.py'

        aug_pipe_file_path = join(args.augmentation_pipe_root_dir, AUG_PIPE_FILE_NAME)

        if not isfile(aug_pipe_file_path):

            msg = aug_pipe_file_path + ' does not exist or is not a file.'
            raise IOError(msg)

        PYTHONPATH.insert(0, args.augmentation_pipe_root_dir)

        try:

            from aug_pipe import pipe

        except ImportError:

            msg = 'There should be a variable called \'pipe\' in \'aug_pipe.py\', also there might be syntax issues or some import errors in your file that would trigger import errors within the utility as a consequence.'
            raise ImportError(msg)

        if not isinstance(pipe, Sequential):

            msg = 'Make sure you wrap your augmentation pipeline in a Sequential instance, this is a requirement to simplify type checking which is not going to affect the way the augmentation process is going to work.'
            raise TypeError(msg)

    annot_dir = join(args.output_dir, 'ForTrainingAnnotations')
            
    if not isdir(annot_dir): makedirs(annot_dir)

    image_formats = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', 'PNG', '.bmp', '.BMP', '.bitmap', '.BITMAP']

    nodes = glob(join(args.root_dir, '*'))
    
    image_files = []
    image_annotations = []

    for node in nodes:

        if isfile(node):

            file_, _ext = splitext(node)

            if _ext in image_formats:

                #boxes = get_annots_from_json(annot_file)
                #_label = boxes[0][4]
                #if _label = '00-7040-068-00': continue

                image_files.append(node)

                annot_file = join( join( args.root_dir , 'ForTrainingAnnotations'), basename(file_) + '_forTraining.json')

                image_annotations.append( get_annots_from_json(annot_file) )

    max_workers = cpu_count()

    image_workloads, annotation_workloads = [[] for _ in range( max_workers )], [[] for _ in range( max_workers )]
    worker_id = 0

    for image_file, image_annotation in zip( tqdm( image_files ), image_annotations ):

        image_workloads[worker_id].append(image_file)
        annotation_workloads[worker_id].append(image_annotation)
        worker_id += 1

        if worker_id == max_workers: worker_id = 0

    workers = []

    for idx, workload_tuple in enumerate(zip(image_workloads, annotation_workloads)):

        image_workload, annotation_workload = workload_tuple

        workers.append(

            Process(

                target = _task, 
                args = (

                    image_workload,
                    annotation_workload,
                    pipe,
                    'worker_nr_' + str(idx + 1) + '_' + args.image_file_prefix, 
                    args.output_dir, 
                    args.num_augs_per_image,
                    args.grayscale 
                )
            )
        )

    start = time()
    for worker in workers: worker.start()

    for worker in workers: worker.join()
    print('Job done in', time() - start)

def main():

    argp = ArgumentParser( description = 'Offline parallel augmentations for dataset' )
    
    argp.add_argument('--root_dir', '-i', required = True, help = 'AnnotationTool dataset format root directory')
    argp.add_argument('--output_dir', '-o', required = False, help = 'Output dir for augmented images, for subsequent inspection before merging with original dataset. Defaults to roo_dir/augs.')
    argp.add_argument('--num_augs_per_image', '-n', required = False, type = int, default = 1, help = 'how many augmented versions for each image you should get')
    argp.add_argument('--image_file_prefix', '-p', required = False, default = 'generated_file', help = '@@')
    argp.add_argument('--augmentation_pipe_root_dir', '-ad', required = False, default = '/media/sml/catalinh/misc/utils/exe/tier1/samples', help = 'abs path to root direcyory where a python module with the name aug_pipe containing a variable called pipe which stores an instance of imgaug Augmenter is to be used for augmenting.')
    argp.add_argument('--grayscale', '-gs', action = 'store_true', help = '@@')
    
    argp.add_argument('--negative_filter', '-neg', action = 'store_true', help = 'overides any specified aug pipe and just applies a negative filter to each image in root dir')
    argp.add_argument('--rotate_n', '-deg', required = False, type = int, default = 0, help = 'overides any specified aug pipe and just rotate by n degrees all images in root dir')
    argp.add_argument('--flip_lr', '-flr', action = 'store_true', help = '@@')
    argp.add_argument('--flip_ud', '-fud', action = 'store_true', help = '@@')

    args = argp.parse_args()

    run(args)

if __name__ == '__main__': main()