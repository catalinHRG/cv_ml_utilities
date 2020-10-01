from sys import path as PYTHONPATH
from argparse import ArgumentParser, Namespace
from os.path import isdir, join, isfile, dirname, basename
from os import listdir, mkdir
from json import load, dump
from math import log
from time import time, sleep
from copy import deepcopy
from subprocess import Popen, PIPE

from psutil import virtual_memory

from glob import glob

from tensorflow.errors import DataLossError, ResourceExhaustedError
from tensorflow.summary import FileWriterCache

'''

    TOOD:

        have a json configurable with both root dir for godapi and root dir for tier exes

'''

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/AnnotationConverter/AnnotationConverter')

from to_google_obj_detect import run as create_tfr

PYTHONPATH.insert(0, "/media/sml/catalinh/misc/godapi/research/object_detection")

from model_main import run as model_dataset

from utils.config_util import get_configs_from_pipeline_file as create_config
from utils.config_util import create_pipeline_proto_from_configs as create_pipeline
from utils.config_util import save_pipeline_config as save_pipeline

from object_detection.protos import input_reader_pb2, preprocessor_pb2

def _check_make(dir_name):

    if not isdir(dir_name): mkdir(dir_name)

def _update_config( config, dataset_root_dir, max_number_of_boxes, original_image_height, original_image_width ):
    
    # updating tfr data source paths

    new_tfr_input_reader = input_reader_pb2.TFRecordInputReader()
    new_tfr_input_reader.input_path.extend( [ join( dataset_root_dir, 'train*.tfr') ] ) # regex to acount for shards with naming convention of '_n' where n is an integer denoting the n th shard
    
    config['train_input_config'].tf_record_input_reader.CopyFrom(new_tfr_input_reader)
    
    new_tfr_input_reader = input_reader_pb2.TFRecordInputReader()
    new_tfr_input_reader.input_path.extend( [ join( dataset_root_dir, 'eval*.tfr') ] ) # regex to acount for shards with naming convention of '_n' where n is an integer denoting the n th shard
    
    config['eval_input_config'].tf_record_input_reader.CopyFrom(new_tfr_input_reader)

    # updating label map path

    lmap_file_path = join( dataset_root_dir, 'lmap.pbtxt')

    config['train_input_config'].label_map_path = lmap_file_path
    config['eval_input_config'].label_map_path = lmap_file_path

    # updating input box buffer size

    config['train_input_config'].max_number_of_boxes = max_number_of_boxes
    config['eval_input_config'].max_number_of_boxes = max_number_of_boxes

    # updating train configuration

    config['train_config'].max_number_of_boxes = max_number_of_boxes
    
    new_random_crop_image = preprocessor_pb2.RandomCropImage()

    aspect_ratio = original_image_width / original_image_height
    
    low_end_ar = aspect_ratio * 0.9
    high_end_ar = aspect_ratio * 1.1

    new_random_crop_image.min_object_covered = 0.800000011920929
    new_random_crop_image.min_aspect_ratio = low_end_ar
    new_random_crop_image.max_aspect_ratio = high_end_ar
    new_random_crop_image.min_area = 0.9900000095367432
    new_random_crop_image.max_area = 1.0
    new_random_crop_image.overlap_thresh = 0.30000001192092896

    new_procesing_step = preprocessor_pb2.PreprocessingStep()
    new_procesing_step.random_crop_image.CopyFrom(new_random_crop_image)

    config['train_config'].data_augmentation_options.extend([new_procesing_step])

    return config

def _get_current_status(datasubset, workbenches):

    datasubset_workspace = dirname(dirname(workbenches[0]))

    current_best_test_recall = 0
    current_best_test_precision = 0

    current_best_train_recall = 0
    current_best_train_precision = 0

    datasubset_status_file = join(datasubset_workspace, 'status.json')

    parse_datasubset = True

    if isfile(datasubset_status_file):

        with open(datasubset_status_file, 'r') as fd:

            datasubset_status = load(fd)

        _current_best_test_recall = datasubset_status['test_recall']
        _current_best_test_precision = datasubset_status['test_precision']

        _current_best_train_recall = datasubset_status['train_recall']
        _current_best_train_precision = datasubset_status['train_precision']

        if _current_best_test_recall is None or _current_best_test_precision is None or _current_best_train_recall is None or _current_best_train_precision is None:

            parse_datasubset = False # out of memory or code broke for some reason at the previous attempt, pass

        elif (
                
                _current_best_test_recall == 1.0 and 
                _current_best_test_precision == 1.0 and 
                _current_best_train_recall == 1.0 and 
                _current_best_train_precision == 1.0 

            ):

                parse_datasubset = False # job done, pass

        else:

            current_best_test_recall = _current_best_test_recall
            current_best_test_precision = _current_best_test_precision

            current_best_train_recall = _current_best_train_recall
            current_best_train_precision = _current_best_train_precision

    return parse_datasubset, current_best_test_recall, current_best_test_precision, current_best_train_recall, current_best_train_precision

def _parse_workbenches( datasubset, workbenches, checkpoint_frequency_in_steps, final ):

    err = None

    parse, current_best_test_recall, current_best_test_precision, current_best_train_recall, current_best_train_precision = _get_current_status(datasubset, workbenches)

    if not parse: return final, err

    print('Working on', datasubset, '...')

    datasubset_workspace = dirname(dirname(workbenches[0]))

    workbench_test_set_metrics = dict()
    workbench_train_set_metrics = dict()

    def _train_and_evaluate(workbench, checkpoint_frequency_in_steps, workbench_test_set_metrics, workbench_train_set_metrics):

        try:

            train_stamp = join(workbench, 'base', 'done_training')

            _workbench = basename(workbench)

            print('working workbench', _workbench, '...')

            if not isfile(train_stamp):

                godapi_train_args = Namespace(

                    workspace = workbench,
                    checkpoint_frequency = checkpoint_frequency_in_steps,
                    keep_checkpoint_max = 50,
                    evaluate = False,
                    training_data_eval = False,
                    sample_1_of_n_eval_examples = 1,
                    sample_1_of_n_eval_on_train_examples = 1,
                    hparams_overrides = None

                )

                _ = model_dataset( godapi_train_args )

                with open(train_stamp, 'w') as fd: pass # stamp workbench as being finished with training

            else:

                print('skipping training step for', workbench, 'since there is a training stamp ...')

            eval_stamp = join(workbench, 'base', 'done_evaluating')

            if not isfile(eval_stamp): 
                
                godapi_eval_on_testset_args = Namespace(

                    workspace = workbench,
                    checkpoint_frequency = checkpoint_frequency_in_steps,
                    keep_checkpoint_max = 50,
                    evaluate = True,
                    training_data_eval = False,
                    sample_1_of_n_eval_examples = 1,
                    sample_1_of_n_eval_on_train_examples = 1,
                    hparams_overrides = None

                )

                print('evaluating workbench', _workbench, ' on test set ...')
                workbench_test_set_metrics = model_dataset( godapi_eval_on_testset_args )

                # each call to model_dataset for evaluation will stamp the workbench as finished, therefore removing it allows for the checkpoints to be evaluated a second time, on the train set

                proc = Popen( ['rm', eval_stamp], bufsize = 2048, stdin = PIPE ) 
                proc.wait()

                print('evaluating workbench', _workbench, ' on train set ...')
                godapi_eval_on_trainset_args = Namespace(

                    workspace = workbench,
                    checkpoint_frequency = checkpoint_frequency_in_steps,
                    keep_checkpoint_max = 50,
                    evaluate = True,
                    training_data_eval = True,
                    sample_1_of_n_eval_examples = 1,
                    sample_1_of_n_eval_on_train_examples = 1,
                    hparams_overrides = None

                )

                workbench_train_set_metrics = model_dataset( godapi_eval_on_trainset_args )
            
            else:

                print('skipping', workbench, 'since there is an evaluation stamp ...')

            memory_usage_percentage = virtual_memory().percent

            if memory_usage_percentage > 80.0: 

                print('Memory usage exceeds hardcoded threshold of 80 percent, aborting ...')
                exit() 

                '''

                    Since this utility is invoked in an infinite loop to account for these shutdowns when the memory leak from godapi eval starts ramping up, 
                    it will pick up from where the shutdown happened with the leaked memory released.


                    TODO: have the script invoking this utility stop the infinite loop when this utility has finished with it's workload entirely, i.e. invoking the utility only takes several seconds to run through and see that there is nothing left o do.
                
                '''

        except DataLossError:

            print('Found corrupt tfr or checkpoint files for', datasubset, 'cleaning workbench and reserializing tfrs ...')

            datasubset_root_dir = join( dirname( dirname( dirname( dirname( workbench ) ) ) ), datasubset )

            tfr_args = Namespace(

                input_dir = datasubset_root_dir,
                labelmap_file = join(datasubset_root_dir, 'lmap.pbtxt'),

                json_dir = "ForTrainingAnnotations",
                annot_file_suffix = "_forTraining",
                masks_dir = None,
                num_shards = 1,
                blacklist = [],
                class_weights = None

            )

            _ = create_tfr(tfr_args)

            cache = FileWriterCache()
            cache.clear()

            workbench_dump_dir = join(workbench, 'base', 'dump')

            if isdir(workbench_dump_dir):

            	proc = Popen( [ rm, '-r', workbench_dump_dir ], bufsize = 2048, stdin = PIPE )
            	proc.wait()

            workbench_train_stamp = join(workbench, 'base', 'done_training')
            workbench_eval_stamp = join(workbench, 'base', 'done_evaluating')

            if isfile(workbench_train_stamp):

                proc = Popen( [rm, workbench_train_stamp], bufsize = 2048, stdin = PIPE )
                proc.wait()
            
            if isfile(workbench_eval_stamp):

                proc = Popen( [ rm, workbnch_eval_stamp ], bufsize = 2048, stdin = PIPE )
                proc.wait()
            	
            print('Resuming train and test workloads with newly serialized tfrs after cleaning workbench', workbench, 'for', datasubset, '...')

            workbench_test_set_metrics, workbench_train_set_metrics = _train_and_evaluate(workbench, checkpoint_frequency_in_steps, workbench_test_set_metrics, workbench_train_set_metrics) # recursively attempt to fix broken tfrs and resume train / test workloads, this acctualy ends up resuming train from latest checkpoint

        return workbench_test_set_metrics, workbench_train_set_metrics

    for workbench in workbenches:

        try: 

            workbench_test_set_metrics, workbench_train_set_metrics = _train_and_evaluate(workbench, checkpoint_frequency_in_steps, workbench_test_set_metrics, workbench_train_set_metrics) # DataLossError exception raise due to tfr corruptin will be caught inside and recursivly attempt to fix them by rewriting tfrs and resuming train / test workloads, while exceptions like OOM will any other exception will be caught at this level, outside the call
            
        except KeyboardInterrupt: 

            print('Keyboard interrupt, aborting ...')
            exit()

        except Exception as _err: # generic
        
            err = _err
            break # this is mostly for out of memory exceptions, so that the rest of the subsets can be attempted

        workbench_base = join(workbench, 'base')

        eval_set_metrics_on_a_checkpoint_basis = workbench_test_set_metrics.get( workbench_base, {} )
        train_set_metrics_on_a_checkpoint_basis = workbench_train_set_metrics.get( workbench_base, {} )

        finished_subset = False

        for checkpoint, eval_set_metrics in eval_set_metrics_on_a_checkpoint_basis.items():

            '''

                TODO:

                    include both cls loss and loc loss, maybe also mean average precision, and have different degrees of importance for each of the individual metrics in
                    concluding whether a checkpoint is better. 

            '''

            test_recall = eval_set_metrics['PascalBoxes_Recall/recall@0.5IOU']
            test_precision = eval_set_metrics['PascalBoxes_Precision/precision@0.5IOU']

            train_set_metrics = train_set_metrics_on_a_checkpoint_basis[ checkpoint ]

            train_recall = train_set_metrics['PascalBoxes_Recall/recall@0.5IOU']
            train_precision = train_set_metrics['PascalBoxes_Precision/precision@0.5IOU']

            if ( 

                    test_recall >= current_best_test_recall and 
                    test_precision >= current_best_test_precision and 
                    train_recall >= current_best_train_recall and
                    train_precision >= current_best_train_precision

                ):

                current_best_test_recall = test_recall
                current_best_test_precision = test_precision
                current_best_train_recall = train_recall
                current_best_train_precision = train_precision

                solution_pointer = { 

                    "workbench": workbench_base, 
                    "checkpoint": checkpoint,

                    'test_recall': float(test_recall), # cannot serialize numpy float scalar, so casting to python float
                    'test_precision': float(test_precision), # cannot serialize numpy float scalar, so casting to python float
                    'train_recall': float(train_recall), # cannot serialize numpy float scalar, so casting to python float
                    'train_precision': float(train_precision) # cannot serialize numpy float scalar, so casting to python float

                }

                final[ datasubset ] = solution_pointer
                
                with open(join(datasubset_workspace, 'status.json'), 'w') as fd:

                    dump(solution_pointer, fd, indent = 3, sort_keys = True)

                if test_recall == 1.0 and test_precision == 1.0 and train_recall == 1.0 and train_precision == 1.0:

                    finished_subset = True
                
                continue # instead of breaking, this actually leads to having the LAST checkpoint ( curtosy of 'w' flag for opening a file) which meets the requirement to be marked as an optimal solution

        if finished_subset: break

    return final, err

def _clean_subset_workspace(datasubset_meta_arch_workbenches, datasubset_status_file):

    cache = FileWriterCache()
    cache.clear()

    best_workbench_candidate_so_far, best_checkpoint_candidate_so_far = None, None

    if isfile(datasubset_status_file):

        with open(datasubset_status_file, 'r') as fd:

            solution_pointer = load(fd)

        best_workbench_candidate_so_far = solution_pointer['workbench']
        best_checkpoint_candidate_so_far = solution_pointer['checkpoint']

    for workbench in datasubset_meta_arch_workbenches:

        workbench_base = join(workbench, 'base')
        
        if best_workbench_candidate_so_far is not None and best_checkpoint_candidate_so_far is not None:

            if workbench_base == best_workbench_candidate_so_far: 

                workbench_dump_dir = join(workbench_base, 'dump')

                content = listdir(workbench_dump_dir)

                for element in content:

                    if element == 'checkpoint': 

                        continue

                    if 'model.ckpt' in element:

                        _best_checkpoint_candidate_so_far = str(best_checkpoint_candidate_so_far)

                        if _best_checkpoint_candidate_so_far in element: 

                            continue

                    _path = join(workbench_dump_dir, element)
                
                    proc = Popen( ['rm', '-r', _path], bufsize = 2048, stdin = PIPE ) # might be a good idea to let the unix rm util internally deal with a list of elements to remove, thus eliminating the overhead introduced by invoking the rm util on each entry at a time
                    proc.wait()

                continue

        train_stamp = join( workbench_base, 'done_training' )
        eval_stamp = join( workbench_base, 'done_evaluating' )

        eligible_for_cleaning = True # might be a good idea to clean anyway, given the fact that there is a valid solution pointer which constitutes the best one so far, will have to investigate

        # eligible_for_cleaning = isfile(train_stamp) and isfile(eval_stamp) 
        # eligible_for_cleaning = isfile(eval_stamp) # will only purge workbench if it underperforms with respect to the best so far
        
        if eligible_for_cleaning:

            workbench_dump_dir = join(workbench_base, 'dump')
            
            if isdir(workbench_dump_dir):

                proc = Popen( ['rm', '-r', workbench_dump_dir], bufsize = 2048, stdin = PIPE ) # might be a good idea to let the unix rm util internally deal with a list of elements to remove, thus eliminating the overhead introduced by invoking the rm util on each entry at a time
                proc.wait()

            
def run(args):

    unmatched_threshold_values = [ 0.5, 0.45, 0.4, 0.35, 0.3 ]
    max_steps_per_attempt = 2500
    num_interpolated_convs_between_encoding_and_predictor = 2

    retinanet_flag = True

    ssd_flag = False

    faster_rcnn_flag = True
    striped_faster_rcnn_flag = True

    blacklist, target = [], []

    if not isdir(args.dataset_root_dir):

        msg = args.dataset_root_dir + ' does not exist or is not a directory'
        raise IOError(msg)

    if args.config is not None:

        if not isfile(args.config):

            msg = args.config + ' does not exist or is not a file'
            raise IOError(msg)

        with open(args.config, 'r') as fd:
        
            config = load(fd)

    else:

        config_file = join(args.dataset_root_dir, 'model_config.json')

        with open(config_file, 'r') as fd:
        
            config = load(fd)
        
    unmatched_threshold_values = config['unmatched_threshold_values']
    max_steps_per_attempt = config['max_steps_per_attempt']
    num_interpolated_convs_between_encoding_and_predictor = config['num_interpolated_convs_between_encoding_and_predictor']
        
    retinanet_flag = config['retinanet_flag']

    # ssd_flag = config['ssd_flag'] # currently ssd is going to be attempted only if there is not enought memory to try retinanet

    faster_rcnn_flag = config['faster_rcnn_flag']
    striped_faster_rcnn_flag = config['striped_faster_rcnn_flag']

    blacklist = config['blacklist']
    target = config['target']

    if not ( retinanet_flag or ssd_flag or faster_rcnn_flag or striped_faster_rcnn_flag ):

        msg = 'The resulting configuration implies that none of the available model meta architectures is to be used. At least one of them has to have a flag for retinanet, ssd or faster rcnn has to be True'
        raise ValueError(msg)

    workspace_dir = join(args.dataset_root_dir, 'workspace')
    _check_make(workspace_dir)

    __dataset_root_dir_content = listdir(args.dataset_root_dir)
    __dataset_root_dir_content.remove("workspace")
    
    _dataset_root_dir_content = []

    for entry in __dataset_root_dir_content:

        _path = join(args.dataset_root_dir, entry)

        if isdir(_path): _dataset_root_dir_content.append(entry)

    dataset_root_dir_content = []

    if target:

        for entry in target:

            dataset_root_dir_content.append(entry)

    else:

        for entry in blacklist:

            if entry in _dataset_root_dir_content:

                _dataset_root_dir_content.remove(entry)

        dataset_root_dir_content = _dataset_root_dir_content
        
    retinanet_workload, ssd_workload, frcnn_workload = dict(), dict(), dict()

    checkpoint_frequencies = { 'retinanet':{}, 'ssd': {}, 'frcnn': {} }

    LOOP_OVER_TRAIN_SET = 90
    
    STEPS_PER_ATTEMPT_LOWER_LIMIT = 750

    for element in dataset_root_dir_content:

        workspace_equivalent = join(workspace_dir, element)
        _check_make(workspace_equivalent)

        preped_dataset_dir = join(args.dataset_root_dir, element)

        # fetching dataset metadata

        box_stats_file = join(preped_dataset_dir, 'info', 'box_stats.json')
        
        with open(box_stats_file, 'r') as fd:

            box_stats = load(fd)
            
            max_number_of_boxes = box_stats['max_box_count_per_image']
            max_box_counter_per_class_per_iamge = box_stats['max_box_count_per_image_per_class']

            num_classes = len(list(box_stats['population_distribution'].keys())) 

        stats_file = join(preped_dataset_dir, 'info', 'stats.json')

        if not isfile(stats_file):

            msg = stats_file + ' is missing or is not a file. Most likely the datasets in this workspace were not standardized using prep_dataset.py yet.'
            raise IOError(msg)

        with open(stats_file, 'r') as fd:

            stats = load(fd)

            optimal_image_heights = stats['optimal_image_heights']
            optimal_image_widths = stats['optimal_image_widths']

            original_image_height = stats['original_image_height']
            original_image_width = stats['original_image_width']

            highest_multiple = stats['highest_multiple']

        if len(optimal_image_heights) != len(optimal_image_widths):

            msg = 'there is an inconsistency in the way the optimal resolutions have been computed'
            raise ValueError(msg)

        datasubset_train_dir = join(preped_dataset_dir, 'train')

        if not isdir(datasubset_train_dir):

            msg = datasubset_train_dir + ' does not exist or is not a directory'
            raise IOError(msg)

        datasubset_num_train_images = len( glob(join(datasubset_train_dir, '*.jpg')) ) 

        if datasubset_num_train_images == 0:

            msg = element + ' found at ' + args.dataset_root_dir + ' does not have any train images, or so it appears. The expected jpg images might have been removed even tho the train set tfr contains them.'
            raise IOErorr(msg)

        # the following 3 if branches are subject to refactoring due to the present almost common blocks found throughout

        if retinanet_flag:

            retinanet_workbench = join(workspace_equivalent, 'retinanet')
            _check_make(retinanet_workbench)

            retinanet_template_config = create_config( join(args.base_configuration_directory, 'retinanet_pipeline.config') )

            retinanet_template_config = _update_config( # helper, update the common parts of retinanet, sdd and frcnn configuration

                retinanet_template_config, 
                preped_dataset_dir, 
                max_number_of_boxes, 
                original_image_height, 
                original_image_width

            )

            retinanet_template_config['model'].ssd.num_classes = num_classes

            batch_size_16_max_input_image_area = 1024 * 384 # for retinanet -- mobilenet_v1 1.0 depth -- fpn min == 3 fpn max == 6

            _batch_size = retinanet_template_config['train_config'].batch_size # currently the default is 16 for retinanet

            workbench_priorities = []

            for optimal_image_height, optimal_image_width in zip(optimal_image_heights, optimal_image_widths):

                max_steps_per_attempt_multiplier = 1
                
                if optimal_image_height * optimal_image_width >= batch_size_16_max_input_image_area: # for a gpu with 8 gb memory

                    ssd_flag = True

                    _batch_size = 8

                    max_steps_per_attempt_multiplier = 2

                _max_steps_per_attempt = max_steps_per_attempt
                
                if max_steps_per_attempt == -1:

                    _max_steps_per_attempt = int( datasubset_num_train_images * LOOP_OVER_TRAIN_SET / _batch_size ) # i.e. loop over train set LOOP_OVER_TRAIN_SET times, which seems ok since there is a 0.5 % chance that at any given time an image can be slightly altered in one of 4 ways

                #print(datasubset_num_train_images);exit()
                _max_steps_per_attempt *= max_steps_per_attempt_multiplier
                
                _max_steps_per_attempt = max( STEPS_PER_ATTEMPT_LOWER_LIMIT, _max_steps_per_attempt )

                _max_steps_per_attempt = _max_steps_per_attempt - (_max_steps_per_attempt % 3) # rounding to the nearest multiple of 3 in order to get exactly 3 checkpoints

                checkpoint_frequency_in_steps = int( _max_steps_per_attempt / 3 ) # will only make 3 checkpoints in total, which seems optimal, also, after each checkpoint, the learning rate will be halfed 

                retinanet_template_config['train_config'].optimizer.rms_prop_optimizer.learning_rate.exponential_decay_learning_rate.decay_steps = checkpoint_frequency_in_steps

                checkpoint_frequencies ['retinanet'] [ element ] = checkpoint_frequency_in_steps

                #print('for retinanet, there are', datasubset_num_train_images, 'and going over the dataset 50 times with a batch size of', str(_batch_size), 'adds up to', str(_max_steps_per_attempt), 'steps')

                retinanet_template_config['train_config'].batch_size = _batch_size
                retinanet_template_config['train_config'].num_steps = _max_steps_per_attempt

                # retinanet_template_config['train_config'].shuffle_buffer_size = min(datasubset_num_train_images, _batch_size * 3)

                _optimal_image_height = optimal_image_height
                _optimal_image_width = optimal_image_width

                if _optimal_image_height * _optimal_image_width > 1605632 * 0.9: # (3 channel input) if the input area is bigger than that value, it leads to some unpredictable behaviours where oom exceptions are not even yhrown anymore, so there has to be some sort of cap that allows stability

                    _optimal_image_height = int(_optimal_image_height *  0.8)
                    _optimal_image_width = int(_optimal_image_width *  0.8)

                    _optimal_image_height = 64 * round(_optimal_image_height / 64)
                    _optimal_image_width = 64 * round(_optimal_image_width / 64)

                retinanet_template_config['model'].ssd.image_resizer.fixed_shape_resizer.height = max(_optimal_image_height, 128)
                retinanet_template_config['model'].ssd.image_resizer.fixed_shape_resizer.width = max(_optimal_image_width, 128)

                max_fpn_level = int(log(highest_multiple, 2))

                retinanet_template_config['model'].ssd.feature_extractor.fpn.max_level = max_fpn_level

                retinanet_template_config['model'].ssd.anchor_generator.multiscale_anchor_generator.max_level = max_fpn_level
                
                for i in range( num_interpolated_convs_between_encoding_and_predictor ):

                    '''

                    TODO: 

                        add another loop for matched_threshold_values i.e. the overlap threshold above which an anchor is considered positive, 
                        relative to a set of ground truths. This way a more extensive set of configurations are yielded that will ultimatly lead to better models.

                    '''

                    for unmatched_threshold_value in unmatched_threshold_values:

                        retinanet_template_config['model'].ssd.box_predictor.weight_shared_convolutional_box_predictor.num_layers_before_predictor = i + 1
                        retinanet_template_config['model'].ssd.matcher.argmax_matcher.unmatched_threshold = unmatched_threshold_value

                        destination_folder_name = str(_optimal_image_height) + 'x' + str(_optimal_image_width) + '_ic3x3_' + str(i + 1) + '_' + 'matcher_1.0_' + str(unmatched_threshold_value) 
                        
                        destination_folder = join(retinanet_workbench, destination_folder_name)
                        _check_make(destination_folder)
                        
                        base_folder = join(destination_folder, 'base')
                        _check_make( base_folder )

                        workbench_priorities.append( destination_folder )

                        pipeline = create_pipeline( retinanet_template_config )

                        if not args.skip_config_generation:

                            save_pipeline( pipeline, base_folder, 'train.config' )
                            save_pipeline( pipeline, base_folder, 'eval.config' )

            retinanet_workload[ element ] = workbench_priorities

        if ssd_flag: 
            
            ssd_workbench = join(workspace_equivalent, 'ssd')
            _check_make(ssd_workbench)

            ssd_template_config = create_config( join(args.base_configuration_directory, 'ssd_pipeline.config') )
            
            ssd_template_config = _update_config( # helper, update the common parts of retinanet, ssd and frcnn configuration

                ssd_template_config, 
                preped_dataset_dir, 
                max_number_of_boxes,  
                original_image_height, 
                original_image_width

            )

            ssd_template_config['model'].ssd.num_classes = num_classes

            workbench_priorities = []

            for optimal_image_height, optimal_image_width in zip(optimal_image_heights, optimal_image_widths):

                ssd_template_config['model'].ssd.image_resizer.fixed_shape_resizer.height = optimal_image_height
                ssd_template_config['model'].ssd.image_resizer.fixed_shape_resizer.width = optimal_image_width

                _max_steps_per_attempt = max_steps_per_attempt
                
                if max_steps_per_attempt == -1:

                    _max_steps_per_attempt = int( datasubset_num_train_images * LOOP_OVER_TRAIN_SET / ssd_template_config['train_config'].batch_size ) # i.e. loop over train set LOOP_OVER_TRAIN_SET times, which seems ok since there is a 0.5 % chance that at any given time an image can be slightly altered in one of 4 ways

                _max_steps_per_attempt *= 2

                _max_steps_per_attempt = max( STEPS_PER_ATTEMPT_LOWER_LIMIT, _max_steps_per_attempt )

                _max_steps_per_attempt = _max_steps_per_attempt - (_max_steps_per_attempt % 3) # rounding to the nearest multiple of 3 in order to get exactly 3 checkpoints
                
                checkpoint_frequency_in_steps = int( _max_steps_per_attempt / 3 ) # will only make 3 checkpoints in total, which seems optimal, also, after each checkpoint, the learning rate will be halfed, which ends up being a well rounded approach 

                #print('for ssd, there are ', str(datasubset_num_train_images), 'and going over the dataset 50 times with a batch size of', str(ssd_template_config['train_config'].batch_size), 'adds up to', _max_steps_per_attempt, 'steps')
                ssd_template_config['train_config'].optimizer.rms_prop_optimizer.learning_rate.exponential_decay_learning_rate.decay_steps = checkpoint_frequency_in_steps

                checkpoint_frequencies ['ssd'] [ element ] = checkpoint_frequency_in_steps

                ssd_template_config['train_config'].num_steps = _max_steps_per_attempt

                # ssd_template_config['train_config'].shuffle_buffer_size = min(datasubset_num_train_images, ssd_template_config['train_config'].batch_size * 3)

                for i in range( num_interpolated_convs_between_encoding_and_predictor ):

                    '''

                    TODO: 

                        add another loop for matched_threshold_values i.e. the overlap threshold above which an anchor is considered positive, 
                        relative to a set of ground truths. This way a more extensive set of configurations are yielded that will ultimatly lead to better models.

                    '''

                    for unmatched_threshold_value in unmatched_threshold_values:

                        ssd_template_config['model'].ssd.box_predictor.weight_shared_convolutional_box_predictor.num_layers_before_predictor = i + 1
                        ssd_template_config['model'].ssd.matcher.argmax_matcher.unmatched_threshold = unmatched_threshold_value

                        destination_folder_name = str(optimal_image_height) + 'x' + str(optimal_image_width) + '_ic3x3_' + str(i + 1) + '_' + 'matcher_1.0_' + str(unmatched_threshold_value) 
                        
                        destination_folder = join(ssd_workbench, destination_folder_name)
                        _check_make(destination_folder)
                        
                        base_folder = join(destination_folder, 'base')
                        _check_make( base_folder )

                        workbench_priorities.append( destination_folder )

                        pipeline = create_pipeline( ssd_template_config )

                        if not args.skip_config_generation:

                            save_pipeline( pipeline, base_folder, 'train.config' )
                            save_pipeline( pipeline, base_folder, 'eval.config' )

            ssd_workload[ element ] = workbench_priorities

        if faster_rcnn_flag or striped_faster_rcnn_flag:

            frcnn_workbench = join( workspace_equivalent, 'frcnn' )
            _check_make( frcnn_workbench )

            frcnn_template_config = create_config( join(args.base_configuration_directory, 'frcnn_pipeline.config'))

            frcnn_template_config = _update_config( # helper, update the common parts of retinanet, ssd and frcnn configuration

                frcnn_template_config, 
                preped_dataset_dir, 
                max_number_of_boxes,  
                original_image_height, 
                original_image_width

            )

            frcnn_template_config['model'].faster_rcnn.num_classes = num_classes

            batch_size_32_max_input_image_area = 1024 * 256 # for frcnn -- inception v2 1.0 depth -- second stage crop size == 10 and max pool kernel == 2x2 and stride == 2 -- 32 mini batch size per stage

            standard_frcnn_stride_8_priorities, striped_frcnn_stride_16_priorities = [], []

            _batch_size = frcnn_template_config['train_config'].batch_size # default is 32
            
            for optimal_image_height, optimal_image_width in zip(optimal_image_heights, optimal_image_widths):

                max_steps_per_attempt_multiplier = 1

                if optimal_image_height * optimal_image_width >= batch_size_32_max_input_image_area: # for a gpu with 8 gb memory

                    _batch_size = 16

                    max_steps_per_attempt_multiplier = 2

                _max_steps_per_attempt = max_steps_per_attempt
                
                if max_steps_per_attempt == -1:

                    _max_steps_per_attempt = int( datasubset_num_train_images * LOOP_OVER_TRAIN_SET / _batch_size ) # i.e. loop over train set LOOP_OVER_TRAIN_SET times, which seems ok since there is a 0.5 % chance that at any given time an image can be slightly altered in one of 4 ways

                _max_steps_per_attempt *= max_steps_per_attempt_multiplier

                _max_steps_per_attempt = max( STEPS_PER_ATTEMPT_LOWER_LIMIT, _max_steps_per_attempt )

                _max_steps_per_attempt = _max_steps_per_attempt - (_max_steps_per_attempt % 3) # rounding to the nearest multiple of 3 in order to get exactly 3 checkpoints

                checkpoint_frequency_in_steps = int( _max_steps_per_attempt / 3 ) # will only make 3 checkpoints in total, which seems optimal, also, after each checkpoint, the learning rate will be halfed, which ends up being a well rounded approach 

                frcnn_template_config['train_config'].optimizer.rms_prop_optimizer.learning_rate.exponential_decay_learning_rate.decay_steps = checkpoint_frequency_in_steps

                checkpoint_frequencies ['frcnn'] [ element ] = checkpoint_frequency_in_steps

                #print('for frcnn, there are', datasubset_num_train_images, 'and going over the dataset 50 times with a batch size of', _batch_size, 'adds up to', _max_steps_per_attempt, 'steps')

                frcnn_template_config['train_config'].batch_size = _batch_size
                frcnn_template_config['train_config'].num_steps = _max_steps_per_attempt

                # frcnn_template_config['train_config'].shuffle_buffer_size = min(datasubset_num_train_images, _batch_size * 3)

                _optimal_image_height = optimal_image_height # imposing minimum image height of 128, otherwise, second stage crop would reach out of bounds, given a network stride of 8
                _optimal_image_width = optimal_image_width # imposing minimum image height of 128, otherwise, second stage crop would reach out of bounds, given a network stride of 8

                if _optimal_image_height * _optimal_image_width > 1605632 * 0.9: # (3 channel input) if the input area is bigger than of that value, it leads to some unpredictable behaviours where oom exceptions are not even yhrown anymore, so there has to be some sort of cap that allows stability

                    _optimal_image_height = int( _optimal_image_height * 0.8 )
                    _optimal_image_width = int( _optimal_image_width * 0.8 )

                    _optimal_image_height = 16 * round(_optimal_image_height / 16)
                    _optimal_image_width = 16 * round(_optimal_image_width / 16)

                frcnn_template_config['model'].faster_rcnn.image_resizer.fixed_shape_resizer.height = max(_optimal_image_height, 128)
                frcnn_template_config['model'].faster_rcnn.image_resizer.fixed_shape_resizer.width = max(_optimal_image_width, 128)

                mini_batch_size = max_number_of_boxes * 4 # 3 negative regions for each positive region

                frcnn_template_config['model'].faster_rcnn.first_stage_minibatch_size = mini_batch_size
                frcnn_template_config['model'].faster_rcnn.first_stage_max_proposals = mini_batch_size
                frcnn_template_config['model'].faster_rcnn.second_stage_batch_size = mini_batch_size

                if faster_rcnn_flag:

                    destination_folder = join( frcnn_workbench, str(optimal_image_height) + 'x' + str(optimal_image_width) + '_standard_stride_8' )
                    _check_make( destination_folder )
                    
                    base_folder = join( destination_folder, 'base' )
                    _check_make( base_folder )

                    standard_frcnn_stride_8_priorities.append(destination_folder)

                    pipeline = create_pipeline( frcnn_template_config )

                    if not args.skip_config_generation:

                        save_pipeline( pipeline, base_folder, 'train.config' )
                        save_pipeline( pipeline, base_folder, 'eval.config' )

                if striped_faster_rcnn_flag:

                    striped_frcnn_template_config = deepcopy(frcnn_template_config)
                    
                    #striped_frcnn_template_config['train_config'].batch_size = 32
                    #striped_frcnn_template_config['train_config'].num_steps = max_steps_per_attempt
                    
                    striped_frcnn_template_config['model'].faster_rcnn.feature_extractor.type = 'faster_rcnn_striped_inception_v2'
                    striped_frcnn_template_config['model'].faster_rcnn.feature_extractor.first_stage_features_stride = 16

                    striped_frcnn_template_config['model'].faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.height_stride = 16
                    striped_frcnn_template_config['model'].faster_rcnn.first_stage_anchor_generator.grid_anchor_generator.width_stride = 16

                    destination_folder = join( frcnn_workbench, str(optimal_image_height) + 'x' + str(optimal_image_width) + '_striped_stride_16' )
                    _check_make( destination_folder )
                    
                    base_folder = join( destination_folder, 'base' )
                    _check_make( base_folder )

                    striped_frcnn_stride_16_priorities.append(destination_folder)

                    pipeline = create_pipeline( striped_frcnn_template_config )

                    if not args.skip_config_generation:

                        save_pipeline( pipeline, base_folder, 'train.config' )
                        save_pipeline( pipeline, base_folder, 'eval.config' )

            frcnn_workload[ element ] = striped_frcnn_stride_16_priorities + standard_frcnn_stride_8_priorities

    final = dict()

    datasubsets1 = retinanet_workload.keys()
    datasubsets2 = ssd_workload.keys()
    datasubsets3 = frcnn_workload.keys()

    datasubsets = []

    datasubsets += list(datasubsets1) # in case there is some inconsistency with what was intended via the config file and what ened up on disc being attempted
    datasubsets += list(datasubsets2) # in case there is some inconsistency with what was intended via the config file and what ened up on disc being attempted
    datasubsets += list(datasubsets3) # in case there is some inconsistency with what was intended via the config file and what ened up on disc being attempted

    datasubsets = set(datasubsets)

    # TODO: log _parse_workbench() potential exceptions returned in a separate file than workbench status file

    for datasubset in datasubsets:

        retinanet_datasubset_workbenches = retinanet_workload.get(datasubset, None)

        if retinanet_datasubset_workbenches:

            checkpoint_frequency_in_steps = checkpoint_frequencies ['retinanet'] [ datasubset ]
            
            #print('Parsing retinanet workload ...')
            final, _ = _parse_workbenches( datasubset, retinanet_datasubset_workbenches, checkpoint_frequency_in_steps, final )

            datasubset_status_file = join(args.dataset_root_dir, 'workspace', datasubset, 'status.json') 

            #print('Cleaning retinanet workspace of redundant training and evaluation specific files for disk space optimization ...')
            
            _clean_subset_workspace(retinanet_datasubset_workbenches, datasubset_status_file)

        ssd_datasubset_workbenches = ssd_workload.get(datasubset, None)

        if ssd_datasubset_workbenches:

            checkpoint_frequency_in_steps = checkpoint_frequencies ['ssd'] [ datasubset ]

            #print('Parsing ssd workload ...')
            final, _ = _parse_workbenches( datasubset, ssd_datasubset_workbenches, checkpoint_frequency_in_steps, final )

            datasubset_status_file = join(args.dataset_root_dir, 'workspace', datasubset, 'status.json') 
            
            #print('Cleaning ssd workspace of redundant training and evaluation specific files for disk space optimization ...')
            
            _clean_subset_workspace(ssd_datasubset_workbenches, datasubset_status_file)

        frcnn_datasubset_workbenches = frcnn_workload.get(datasubset, None)

        if frcnn_datasubset_workbenches:

            checkpoint_frequency_in_steps = checkpoint_frequencies ['frcnn'] [ datasubset ]

            # err = None

            #print('Parsing frcnn workload ...')
            final, err = _parse_workbenches( datasubset, frcnn_datasubset_workbenches, checkpoint_frequency_in_steps, final )

            datasubset_status_file = join(args.dataset_root_dir, 'workspace', datasubset, 'status.json') 
            
            #print('Cleaning frcnn workspace of redundant training and evaluation specific files for disk space optimization ...')

            _clean_subset_workspace(frcnn_datasubset_workbenches, datasubset_status_file)

            if err is not None: # only stamp a certain workspace associated with a subset containing an array of workbenches after all other of the meta architectures have been attempted and if no pointer to an optimal solution is available

                datasubset_workspace = dirname(dirname(frcnn_datasubset_workbenches[0]))

                subset_status_file = join(datasubset_workspace, 'status.json')

                if not isfile(subset_status_file):
                    
                    solution_pointer = { 

                        "workbench": None, 
                        "checkpoint": None, 
                        
                        "details": str(err),

                        'test_recall': None,
                        'test_precision': None,
                        'train_recall': None,
                        'train_precision': None

                    }

                    # final[ datasubset ] = solution_pointer # avoid overiding any valid pointer to optimal solution potentially found by parsing previous meta architectures

                    with open(subset_status_file, 'w') as fd:

                        dump(solution_pointer, fd, indent = 3, sort_keys = True)


    print('Finished')

    return final

def main():

    parser = ArgumentParser( description = '' )
    
    parser.add_argument( '--dataset_root_dir', '-i', required = True, help = 'Root directory path for standardized dataset layout.' )
    parser.add_argument( '--config', '-c', required = False, help = 'Allows for in depth custumization of the inner workings for the utility and ultimatly the heuristic that is employed to find the optimal solution for the underlying grid search problem.')
    parser.add_argument( '--skip_config_generation', '-s', action = 'store_true', help = 'skips the pipeline config files generation. this is mainly to be used in case there was not enough memory for the baseline configurations to run.')
    parser.add_argument( '--base_configuration_directory', '-bc', required = False, default = '/media/sml/catalinh/misc/utils/exe/tier1/samples/godapi_configs/phase1', help = 'root directory for frcnn and retinanet base configuration files')

    args = parser.parse_args()

    status = run(args)

if __name__ == '__main__': main()