from sys import path as PYTHONPATH

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier1')

from get_dataset_stats import run as get_stats
from get_random_eval_selection import run as get_random_split
from resize_annotated_images import run as resize_dataset

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/AnnotationConverter/AnnotationConverter')

from to_google_obj_detect import run as create_tfr

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier2')

from find_optimal_resolution import run as find_optimal_image_shape

from argparse import ArgumentParser, Namespace
from json import load, dump
from os.path import isdir, isfile, join, dirname
from os import system, mkdir
from time import time
from subprocess import Popen, PIPE

def put_quotes(string):

	return '"' + string + '"'

def run(args):

	start = time()
	
	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)
 
	tray_workspace_dir = join(args.root_dir, 'workspace')

	if not isdir(tray_workspace_dir): mkdir(tray_workspace_dir)

	phase1_workspace_dir = join(tray_workspace_dir, 'phase1_standardized_dataset')

	if not isdir(phase1_workspace_dir): mkdir(phase1_workspace_dir)

	resize_args = Namespace(

		input_dir = args.root_dir, 
		output_dir = phase1_workspace_dir,
		new_width = args.base_image_width,
		new_height = args.base_image_height,
		alter_ann_width = 0,
		alter_ann_height = 0,
		modify_ann_operation = 'none',
		log = False,
		squish = False,
		group = True,
		ann_file_suffix = '_forTraining',
		phase_1_boxes = True,
		label_config = None
	)

	resize_dataset(resize_args)

	stats_dir = join(phase1_workspace_dir, 'info')

	if not isdir(stats_dir):

		get_stats_args = Namespace(

			dataset_input_dir = phase1_workspace_dir,
			view_annots = True,
			scale_bins = 5,
			aspect_ratios_bins = 5,
			balance = False,
			multiscale_grid_anchor_generator = False,
			log = False,
			num_box_clusters = 1,
			annot_file_suffix = '_forTraining',
			include_masks = False

		)

		_ = get_stats(get_stats_args)

	box_stats_file = join(stats_dir, 'box_stats.json')

	with open(box_stats_file, 'r') as fd:

		box_stats = load(fd)

	label_groups = {}

	if args.label_groups is not None:

		if isinstance(args.label_groups, str):			

			if isfile(args.label_groups):

				with open(args.label_groups, 'r') as fd:

					label_groups = load(fd)

		elif isinstance(args.label_groups, dict):

			label_groups = args.label_groups

		else:

			msg = 'Expected ' + type(str) + ' or ' + type(dict) + ' got ' + type(args.label_groups)

	marked = []
	targets = {}

	for group_label, components in label_groups.items():

		_target = {}

		for component in components:

			_target[ component ] = component
			marked.append( component )

		target = {

			"label_mappings": _target,
			"ignore_images_with_no_explicit_mappings": False

		}

		targets[ group_label ] = target

	_population_distribution = box_stats['population_distribution']

	class_labels = list( _population_distribution.keys() )

	for class_label in class_labels:

		if class_label in marked: continue

		_target = { class_label: class_label }

		target = {

			"label_mappings": _target,
			"ignore_images_with_no_explicit_mappings": False
		}		

		targets[ class_label ] = target

	output_dir = join( phase1_workspace_dir, 'subsets' )

	aug_config = dict()

	for identifier, grouping in targets.items():

		if args.candidates:

			if identifier not in args.candidates: continue

		aug_config[ identifier ] = {

			"rotate_up_to_360_by": 0,
			"rotate_180": False,
			"negative_filter": False,
			"flip_up_down": True,
			"flip_left_right": True,
			"baseline_augs_per_image": 0

		}

		print('Fetching', identifier, 'datasubset ...')

		label_mapping_file = join(phase1_workspace_dir, identifier + '.json')

		with open(label_mapping_file, 'w') as fd:

			dump(grouping, fd)

		subset_output_dir = join( output_dir, identifier )

		resize_args = Namespace(

			input_dir = args.root_dir, 
			output_dir = subset_output_dir,
			new_width = args.base_image_width,
			new_height = args.base_image_height,
			alter_ann_width = 0,
			alter_ann_height = 0,
			modify_ann_operation = 'none',
			log = False,
			squish = False,
			group = True,
			ann_file_suffix = '_forTraining',
			phase_1_boxes = True,
			label_config = label_mapping_file
		)

		resize_dataset( resize_args )

		#command = 'cp ' + put_quotes( label_mapping_file ) + ' ' + put_quotes( subset_output_dir )
		#system( command )

		proc = Popen( [ 'cp', label_mapping_file, subset_output_dir ], bufsize = 2048, stdin = PIPE)
		proc.wait()

		print('Running datasubset sweep for stats ...')

		get_stats_args = Namespace(

			dataset_input_dir = subset_output_dir,
			view_annots = True,
			scale_bins = 5,
			aspect_ratios_bins = 5,
			balance = False,
			multiscale_grid_anchor_generator = False,
			log = False,
			num_box_clusters = 1,
			annot_file_suffix = '_forTraining',
			include_masks = False

		)

		_ = get_stats(get_stats_args)

		'''

		_base = 64
		max_obj_size = 50
		min_obj_size = 45

		find_optimal_resolution_args = Namespace(

			root_dir = subset_output_dir,
			base = _base,
			max_object_target_size = max_obj_size,
			min_object_target_size = min_obj_size

		)

		_results = find_optimal_image_shape(find_optimal_resolution_args)

		info_dir = join(subset_output_dir, 'info')

		metadata = {

			"optimal_image_heights": [ _results['square_boxes'][0] ],
			"optimal_image_widths": [ _results['square_boxes'][1] ],
			"original_image_height": base_image_height,
			"original_image_width": base_image_width,
			"highest_multiple": _base,
			"average_object_size": (max_obj_size + min_obj_size) / 2

		}

		with open(join(info_dir, 'stats.json'), 'w') as fd:

			dump(metadata, fd)

		'''

		lmap_content_lines = []
		counter = 1

		_grouping = grouping['label_mappings']

		elements = list( _grouping.keys() )

		for element in elements:

			lmap_content_line = "item{name: \"" + element + "\", id: " + str(counter) + "}" + '\n'
			lmap_content_lines.append(lmap_content_line)
			counter += 1

		lmap_file = join(subset_output_dir, 'lmap.pbtxt')

		with open(lmap_file, 'w') as fd:

			for lmap_content_line in lmap_content_lines:

				fd.write(lmap_content_line)

	dataset_aug_config_file = join(output_dir, 'aug_config.json')

	with open(dataset_aug_config_file, 'w') as fd:

		dump(aug_config, fd, indent = 3, sort_keys = True)

	workspace_dir = join( output_dir, 'workspace' )

	if not isdir( workspace_dir ): mkdir( workspace_dir )

	#command = 'cp ' + put_quotes( join( args.root_dir , "groups_lmap.pbtxt" ) ) + ' ' + put_quotes( workspace_dir )
	#system(command)

	proc = Popen( [ 'cp', join( args.root_dir , "groups_lmap.pbtxt" ), workspace_dir ], bufsize = 2048, stdin = PIPE)
	proc.wait()

	model_config_file = join(output_dir, 'model_config.json')

	p1_meta_dev_config = {

		"retinanet_flag": True,
		"num_interpolated_convs_between_encoding_and_predictor": 2,
		"unmatched_threshold_values": [0.5, 0.45, 0.4, 0.35, 0.3],

		"faster_rcnn_flag": True,

		"max_steps_per_attempt": 3000,

		"blacklist": [],
		"target": []

	}

	with open(model_config_file, 'w') as fd:

		dump(p1_meta_dev_config, fd, indent = 3, sort_keys = True)

	print('Elapsed time:', (time() - start) / 60., 'minutes.')
		
	return output_dir

def main():

	parser = ArgumentParser(description = 'prep phase 1 workspace, where one can manually intervent for aditional modifications before the dataset can be preped for training')
	
	parser.add_argument('--root_dir', '-i', required = True, help = 'Root directory path for AnnotationTool format dataset')
	parser.add_argument('--label_groups', '-lg', required = False, help = 'Path to a json file that illustrates how certain labels should be bundled together if needed. If an element is present in a grouping found here and is also blacklisted for some reason, the blacklisting for that element will not occur.')
	parser.add_argument('--candidates', '-c', nargs = '*', default = [], dest = 'candidates', help = 'List of p1 class labels or the identifiers for groups of class labels intended for training.')
	parser.add_argument('--base_image_height', required = False, type = int, default = 512, help = '@@')
	parser.add_argument('--base_image_width', required = False, type = int, default = 1024, help = '@@')

	args = parser.parse_args()

	_ = run(args)

if __name__ == '__main__': main()