import os
import sys
import argparse
import cv2
import json
import math
from enum import Enum

from os.path import isfile
from json import load

def get_pad_wh_amount(in_w, in_h, desirable_ratio):
    
    pad_w = pad_h = 0
    
    if in_w / in_h > desirable_ratio:
        
        pad_h = int( in_w / desirable_ratio - in_h )
        
    else:
        
        pad_w = int( desirable_ratio * in_h - in_w )

    return (pad_w, pad_h)

def dir_exists(dir_path):
    return os.path.isdir(dir_path) and os.path.exists(dir_path)


def file_exists(file_path):
    return os.path.isfile(file_path) and os.path.exists(file_path)


def get_files(dir_path, extensions):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.lower().endswith(extensions)]
    return files


def create_directory(dir_path):
    if not dir_exists(dir_path):
        os.makedirs(dir_path)


class AnnPointPosition(Enum):
    TLX = 1 #TopLeftX
    TRX = 2 #TopRightX
    BLX = 3 #BottomLeftX
    BRX = 4 #BottomRightX
    TLY = 5 #TopLeftY
    TRY = 6 #TopRightY
    BLY = 7 #BottomLeftY
    BRY = 8 #BottomRightY


def add_operation(pt, value):
    return pt + value


def substract_operation(pt, value):
    return pt - value


def modify_annotation(pt, value_x, value_y, ann_point_position, operation_type):
    if operation_type.lower() == "none":
        res = pt
    if operation_type.lower() == "shrink":
        res = {
            AnnPointPosition.TLX: lambda pt, value_x, value_y: add_operation(pt, value_x),
            AnnPointPosition.TLY: lambda pt, value_x, value_y: add_operation(pt, value_y),
            AnnPointPosition.TRX: lambda pt, value_x, value_y: substract_operation(pt, value_x),
            AnnPointPosition.TRY: lambda pt, value_x, value_y: add_operation(pt, value_y),     
            AnnPointPosition.BLX: lambda pt, value_x, value_y: add_operation(pt, value_x),
            AnnPointPosition.BLY: lambda pt, value_x, value_y: substract_operation(pt, value_y),
            AnnPointPosition.BRX: lambda pt, value_x, value_y: substract_operation(pt, value_x),
            AnnPointPosition.BRY: lambda pt, value_x, value_y: substract_operation(pt, value_y)
        }[ann_point_position](pt, value_x, value_y)
    elif operation_type.lower() == "enlarge":
        res = {
            AnnPointPosition.TLX: lambda pt, value_x, value_y: substract_operation(pt, value_x),
            AnnPointPosition.TLY: lambda pt, value_x, value_y: substract_operation(pt, value_y),
            AnnPointPosition.TRX: lambda pt, value_x, value_y: add_operation(pt, value_x),
            AnnPointPosition.TRY: lambda pt, value_x, value_y: substract_operation(pt, value_y),                                        
            AnnPointPosition.BLX: lambda pt, value_x, value_y: substract_operation(pt, value_x),
            AnnPointPosition.BLY: lambda pt, value_x, value_y: add_operation(pt, value_y),
            AnnPointPosition.BRX: lambda pt, value_x, value_y: add_operation(pt, value_x),
            AnnPointPosition.BRY: lambda pt, value_x, value_y: add_operation(pt, value_y)
        }[ann_point_position](pt, value_x, value_y)
    else:
        res = pt
    return res

def parse_annotation(

    group_flag,
    ignore_is_background,
    pad_w, 
    pad_h, 
    annotation_path, 
    annotation_basename, 
    annotations_structure, 
    image_height, 
    image_width, 
    resize_factor_x, 
    resize_factor_y, 
    alter_ann_value, 
    operation_type,
    label_config

    ):
    
    annotation_data = []
    selected_annotations = []

    image_height *= resize_factor_y
    image_width *= resize_factor_x

    _labels = label_config['label_mappings']
    ignore_images = label_config['ignore_images_with_no_explicit_mappings']

    good_image = True

    # account for evenly split padding, top -- bottom and left -- right, by shifting the box points xy coords accordingly

    shift_x = 0 if pad_w == 0 else pad_w // 2
    shift_y = 0 if pad_h == 0 else pad_h // 2

    with open(annotation_path) as annotation_file:
        
        annotation_data = json.load(annotation_file)
        boxes = annotation_data['Annotations']

        _boxes = []

        label_key = 'Group' if group_flag else 'Label'

        for box in boxes:

            is_background = box.get('IsBackground', None)

            if is_background is not None: 

                if is_background and not ignore_is_background: continue

            ptb_xy = box.get('PointBottomLeft', None)

            if ptb_xy is not None: 

                _boxes.append( box ) # standard annotation tool box dict

            detection_rect = box.get('DetectionRect', None)
            if detection_rect is not None: 

                detection_rect['Label'] = box.get(label_key)

                _boxes.append( detection_rect ) # biomet detection rect dict

            if label_key == 'Label':

                classification_rect = box.get('ClassificationRect', None)
                if classification_rect is not None: 

                    classification_rect['Label'] = box.get(label_key)
                    
                    _boxes.append( classification_rect ) # -- classification --

                first_classification_rect = box.get('FirstClassificationRect', None)
                if first_classification_rect is not None: 

                    first_classification_rect['Label'] = box.get(label_key)

                    _boxes.append( first_classification_rect ) # -- classification --

                second_classification_rect = box.get('SecondClassificationRect', None)
                if second_classification_rect is not None: 

                    second_classification_rect['Label'] = box.get(label_key)
                    
                    _boxes.append( second_classification_rect ) # -- classification --

                classification_rects = box.get('ClassificationRects', None)
                if classification_rects is not None:

                    for _classification_rect in classification_rects:

                        tag_label = _classification_rect.get('Label', None)

                        if tag_label is None:

                            _classification_rect['Label'] = box.get(label_key)

                        _boxes.append(_classification_rect) 


        for _box in _boxes:

                ptrV = _box['PointTopRight'].split(',')
                #print("Before TRX: " + str(ptrV[0]))
                ptrV[0] = modify_annotation(float(ptrV[0]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.TRX, operation_type)
                #print("After TRX: " + str(ptrV[0]))
                #print("Before TRY: " + str(ptrV[1]))
                ptrV[1] = modify_annotation(float(ptrV[1]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.TRY, operation_type)
                #print("After TRY: " + str(ptrV[1]))

                ptrX = (float(ptrV[0]) + shift_x) * resize_factor_x
                ptrX = 0 if ptrX < 0 else image_width if ptrX > image_width else ptrX
                ptrY = (float(ptrV[1]) + shift_y) * resize_factor_y
                ptrY = 0 if ptrY < 0 else image_height if ptrY > image_height else ptrY

                ptlV = _box['PointTopLeft'].split(',')
                #print("Before TLX: " + str(ptlV[0]))
                ptlV[0] = modify_annotation(float(ptlV[0]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.TLX, operation_type)
                #print("After TLX: " + str(ptlV[0]))
                #print("Before TLY: " + str(ptlV[1]))
                ptlV[1] = modify_annotation(float(ptlV[1]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.TLY, operation_type)
                #print("After TLY: " + str(ptlV[1]))
                ptlX = (float(ptlV[0]) + shift_x) * resize_factor_x
                ptlX = 0 if ptlX < 0 else image_width if ptlX > image_width else ptlX
                ptlY = (float(ptlV[1]) + shift_y) * resize_factor_y
                ptlY = 0 if ptlY < 0 else image_height if ptlY > image_height else ptlY

                pblV = _box['PointBottomLeft'].split(',')
                #print("Before BLX: " + str(pblV[0]))
                pblV[0] = modify_annotation(float(pblV[0]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.BLX, operation_type)
                #print("After BLX: " + str(pblV[0]))
                #print("Before BLY: " + str(pblV[1]))
                pblV[1] = modify_annotation(float(pblV[1]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.BLY, operation_type)
                #print("After BLY: " + str(pblV[1]))
                pblX = (float(pblV[0]) + shift_x) * resize_factor_x
                pblX = 0 if pblX < 0 else image_width if pblX > image_width else pblX
                pblY = (float(pblV[1]) + shift_y) * resize_factor_y
                pblY = 0 if pblY < 0 else image_height if pblY > image_height else pblY

                pbrV = _box['PointBottomRight'].split(',')
                #print("Before BRX: " + str(pbrV[0]))
                pbrV[0] = modify_annotation(float(pbrV[0]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.BRX, operation_type)
                #print("After BRX: " + str(pbrV[0]))
                #print("Before BRY: " + str(pbrV[1]))
                pbrV[1] = modify_annotation(float(pbrV[1]), alter_ann_value["x"], alter_ann_value["y"], AnnPointPosition.BRY, operation_type)
                #print("After BRY: " + str(pbrV[1]))
                pbrX = (float(pbrV[0]) + shift_x) * resize_factor_x
                pbrX = 0 if pbrX < 0 else image_width if pbrX > image_width else pbrX
                pbrY = (float(pbrV[1]) + shift_y) * resize_factor_y
                pbrY = 0 if pbrY < 0 else image_height if pbrY > image_height else pbrY
                
                #global _labels
                #global ignore_images

                _label = _box['Label']

                if _labels:
                
                    _label = _labels.get(_label, None)
                   
                if _label is None: 

                    good_image = False or ( not ignore_images )
                    continue

                selected_annotation = {}
                selected_annotation['Label'] = _label
                selected_annotation['Type'] = 'Manual'
                selected_annotation['RealAngle'] = 0.0
                selected_annotation['Angle'] = 0.0
                selected_annotation['PointTopRight'] = str(ptrX) + "," + str(ptrY)
                selected_annotation['PointTopLeft'] = str(ptlX) + "," + str(ptlY)
                selected_annotation['PointBottomLeft'] = str(pblX) + "," + str(pblY)
                selected_annotation['PointBottomRight'] = str(pbrX) + "," + str(pbrY)
                selected_annotation['objMask'] = _box.get('objMask', None)
                selected_annotations.append(selected_annotation)

    if good_image:

        # Replace old boxes with the new ones and write them
        annotation_data['Annotations'] = selected_annotations

        forTraining_output_ann_path = os.path.join(annotations_structure["output_paths"]["fortraining"], annotation_basename + annotations_structure["suffixes"]["fortraining"])
        with open(forTraining_output_ann_path, "w") as ft: json.dump(annotation_data, ft)

    return good_image


def run(args):

    squish_flag = args.squish

    group_label_flag = args.group
    ignore_is_background = args.phase_1_boxes

    log_flag = args.log

    input_dir = args.input_dir

    if args.label_config is not None:

        if not isfile(args.label_config):

            msg = label_config + ' does not exist or is not a file.'
            raise IOError(msg)

        with open(args.label_config, 'r') as fd:

            label_config = load(fd)

        # first element
        target_key = 'label_mappings'

        entry = label_config.get(target_key, None)
        
        if entry is None:

            msg = 'There should be an entry called \'' + target_key + '\''
            raise ValueError(msg)

        if not isinstance(entry, dict):

            msg = target_key + ' should be type dictionary.'
            raise TypeError(msg)

        # second element
        target_key = 'ignore_images_with_no_explicit_mappings'

        entry = label_config.get(target_key, None)
        
        if entry is None:

            msg = 'There should be an entry called \'' + target_key + '\''
            raise ValueError(msg)

        if not isinstance(entry, bool):

            msg = target_key + ' should be type boolean.'
            raise TypeError(msg)

    else:

        label_config = {

            "label_mappings": {},
            "ignore_images_with_no_explicit_mappings": False

        }

    if args.output_dir is None: 

        output_dir = os.path.join(args.input_dir, 'resized', str(args.new_width) + 'x' + str(args.new_height))

    else: 

        output_dir = args.output_dir

    new_width = args.new_width
    new_height = args.new_height

    alter_ann_value = {}
    alter_ann_value["x"] = int(args.alter_ann_width)
    alter_ann_value["y"] = int(args.alter_ann_height)
    operation_type = args.modify_ann_operation

    if not dir_exists(input_dir):
        if log_flag: print("Input directory doesn't exist!")
        return

    input_annotations_dir_path = os.path.join(input_dir, "ForTrainingAnnotations")
    if not dir_exists(input_annotations_dir_path):
        if log_flag: print("No annotations to load!")
        return

    annotations_structure = {}
    annotations_structure["suffixes"] = {}
    annotations_structure["output_paths"] = {}
    annotations_structure["suffixes"]["fortraining"] = args.ann_file_suffix + ".json"
    annotations_structure["output_paths"]["fortraining"] = os.path.join(output_dir, "ForTrainingAnnotations")
    
    create_directory(output_dir)
    for key,_ in annotations_structure["output_paths"].items():
        create_directory(annotations_structure["output_paths"][key])

    training_images = get_files(input_dir, ('.png', '.jpg', '.jpeg', '.bmp'))
    num_of_images = len(training_images)
    current_image_index = 1

    for training_image in training_images:
        if log_flag: print("[" + str(current_image_index) + "/" + str(num_of_images) + "] Processing: " + training_image)
        current_image_index += 1

        filename_without_extension, _ = os.path.splitext(training_image)

        input_img_path = os.path.join(input_dir, training_image)
        input_img_annotation_path = os.path.join(input_annotations_dir_path, filename_without_extension + annotations_structure["suffixes"]["fortraining"])
        
        if not file_exists(input_img_annotation_path):
            if log_flag: print("[!!!] Cannot find the annotation file for the current image: " + input_img_path)
            continue

        source_image = cv2.imread(input_img_path)

        h, w, _ = source_image.shape
           
        pad_w, pad_h = get_pad_wh_amount(w, h, new_width / new_height)

        if squish_flag: pad_w, pad_h = 0, 0
        
        pad_top, pad_bottom, pad_left, pad_right = 0, 0, 0, 0

        if pad_w > 0: 

            _split_amount = pad_w // 2

            pad_left, pad_right = _split_amount, _split_amount

        if pad_h > 0: 

            _split_amount = pad_h // 2

            pad_top, pad_bottom = _split_amount, _split_amount

        source_image = cv2.copyMakeBorder(source_image, top = pad_top, bottom = pad_bottom, left = pad_left, right = pad_right, borderType = cv2.BORDER_CONSTANT, value = 0)

        resize_factor_x = (new_width / (w + pad_w ))
        resize_factor_y = (new_height / (h + pad_h ))
        image_height, image_width = source_image.shape[0:2]


        parse_result = parse_annotation(

            group_label_flag,
            ignore_is_background,
            pad_w, 
            pad_h, 
            input_img_annotation_path, 
            filename_without_extension, 
            annotations_structure, 
            image_height, 
            image_width, 
            resize_factor_x, 
            resize_factor_y, 
            alter_ann_value, 
            operation_type,
            label_config

        )
        
        if parse_result:

            output_image_path = os.path.join(output_dir, filename_without_extension + '.jpg')

            result_image = cv2.resize(source_image, (new_width, new_height) )
            cv2.imwrite(output_image_path, result_image)

    return 0

def main(argv):
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate a new dataset from an existing one.')
    parser.add_argument('--input_dir', '-i', required=True, help='Input directory generated by ImageAnnotationTool')
    parser.add_argument('--output_dir', '-o', required=False, default = None, help='Output directory')
    parser.add_argument('--new_width', '-nw', type=int, required=True, help='New image width')
    parser.add_argument('--new_height', '-nh', type=int, required=True, help='New image height')
    parser.add_argument('--alter_ann_width', '-anw', type=int, default=0, help='Pixels to alter annotations width')
    parser.add_argument('--alter_ann_height', '-anh', type=int, default=0, help='Pixels to alter annotations height')
    parser.add_argument('--modify_ann_operation', default='none', help='Modify annotations boxes method. Can be none, shrink or enlarge. Default is: none') 
    parser.add_argument('--log', '-l', action = 'store_true', help = 'Whether to log to stdout info regarding process')
    parser.add_argument('--squish', '-s', action = 'store_true', help = 'Whether to ignore aspect ratio')
    parser.add_argument('--group', '-g', action = 'store_true', help = 'Whether to choose the "Group" key for the class label or not, the later case being the "Label" key for the class label')
    parser.add_argument('--ann_file_suffix', '-as', default = '_forTraining', help = 'suffix for json annot file')
    parser.add_argument('--phase_1_boxes', '-p1', action = 'store_true', help = 'this will bypass is_background flag, which is to be used only for p2 boxes')
    parser.add_argument('--label_config', '-lc', required = False, help = 'The configuration related to how labels are to be changed if desired.')
    args = parser.parse_args()

    # Run the script
    run(args)

if __name__ == "__main__":
    main(sys.argv[1:])
