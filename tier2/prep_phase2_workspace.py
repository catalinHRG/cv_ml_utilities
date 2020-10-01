from argparse import ArgumentParser, Namespace
from sys import path as PYTHONPATH
from glob import glob
from cv2 import imread
from os.path import join, isdir, basename
from os import mkdir
from tqdm import tqdm
from subprocess import Popen, PIPE
from json import dump

PYTHONPATH.insert(0, '/media/sml/catalinh/misc/utils/exe/tier1')

from resize_annotated_images import run as resize_dataset
from get_dataset_stats import run as get_stats

def run(args):

	if not isdir(args.root_dir):

		msg = args.root_dir + ' does not exist or is not a directory.'
		raise IOError(msg)

	p2_dataset_dir = join(args.root_dir, 'Crops')

	if not isdir(p2_dataset_dir):

		msg = 'Aparently there is no directroy called Crops, which was supposed to contain the phase2 dataset'
		raise IOError(msg)

	if not args.candidates:

		msg = 'There should be at least one candidate intended for phase2 training if this utility is to be invoked.'
		raise ValueError(msg)

	workspace = join(args.root_dir, 'workspace')

	tray_id = basename(args.root_dir)

	if not isdir(workspace): mkdir(workspace)

	p2_workspace_directory = join(workspace, 'phase2_standardized_dataset')

	if not isdir(p2_workspace_directory): mkdir(p2_workspace_directory)

	candidate_base_resolution_mapping = dict()

	aug_config = dict()

	for candidate in tqdm(args.candidates):

		print('Preping', candidate, '...')

		datasubset_input_dir = join(p2_dataset_dir, candidate)

		if not isdir(datasubset_input_dir):

			msg = candidate + ' does not exist within the phase2 dataset for tray ' + tray_id
			raise ValueError(msg)

		all_subset_images = glob(join(datasubset_input_dir, '*.jpg'))

		if not all_subset_images:

			msg = 'Aparently there are no image files with the .jpg extension within ' + candidate + ' subset directory.'
			raise ValueError(msg)

		aug_config[ candidate ] = {

			"rotate_up_to_360_by": 0,
			"rotate_180": False,
			"negative_filter": True,
			"flip_up_down": False,
			"flip_left_right": False,
			"baseline_augs_per_image": 5

		}

		max_h, max_w = 0, 0

		print('Sweaeping dataset images for max sides ...')
		
		for image_path in tqdm(all_subset_images):

			img = imread(image_path)

			h, w, _ = img.shape

			if h > max_h: max_h = h
			if w > max_w: max_w = w

		max_h = args.base * round(max_h / args.base)
		max_w = args.base * round(max_w / args.base)

		if args.square_images:

			max_h = max(max_h, max_w)
			max_w = max(max_h, max_w)

		candidate_base_resolution_mapping[candidate] = { 'height': max_h, 'width': max_w }

		print('done!')

		datasubset_output_dir = join(p2_workspace_directory, candidate)

		if not isdir(datasubset_output_dir): mkdir(datasubset_output_dir)

		print('Resizing ...')

		resize_dataset_args = Namespace(

			input_dir = datasubset_input_dir, 
			output_dir = datasubset_output_dir,
			new_height = max_h,
			new_width = max_w,
			alter_ann_width = 0,
			alter_ann_height = 0,
			modify_ann_operation = 'none',
			log = False,
			squish = False,
			group = False,
			ann_file_suffix = '_forTraining',
			phase_1_boxes = False,
			label_config = None
		)

		_ = resize_dataset( resize_dataset_args )

		print('done!')
		
		print('Fetching dataset stats ...')

		get_stats_args = Namespace(

			dataset_input_dir = datasubset_output_dir,
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

		_ = get_stats( get_stats_args )


		proc = Popen( 
		
			['cp', join(datasubset_input_dir, 'lmap.pbtxt'), datasubset_output_dir], 
			bufsize = 2048, 
			stdin = PIPE 

		)
		
		proc.wait()

		print('done!')

	dataset_aug_config_file = join(p2_workspace_directory, 'aug_config.json')

	with open(dataset_aug_config_file, 'w') as fd:

		dump(aug_config, fd, indent = 3, sort_keys = True)

	model_config_file = join(p2_workspace_directory, 'model_config.json')

	p2_meta_dev_config = {

		"retinanet_flag": False,
		"num_interpolated_convs_between_encoding_and_predictor": 2,
		"unmatched_threshold_values": [0.5, 0.45, 0.4, 0.35, 0.3],

		"faster_rcnn_flag": True,

		"max_steps_per_attempt": 4000,

		"blacklist": [],
		"target": []

	}

	with open(model_config_file, 'w') as fd:

		dump(p2_meta_dev_config, fd, indent = 3, sort_keys = True)

	return p2_workspace_directory

def main():

    argp = ArgumentParser( description = 'Designate phase2 candidates for training and imposes a standard workspace layout for the subsequent utilities to work with' )
    
    argp.add_argument('--root_dir', '-i', required = True, help = 'Biomet annotation tool export Crops folder directory which contains the various datasubsets intended for phase2')
    argp.add_argument('--candidates', '-c', nargs = '*', default = [], dest = 'candidates', help = 'list of tool_categories intended for p2 training')
    argp.add_argument('--base', '-b', required = False, type = int, default = 16, help = 'value used for rounding up resolution to the nearest multiple of this base.')
    argp.add_argument('--square_images', '-s', action = 'store_true', help = 'wether to pad to max side in order to end up with a square image, which is necesary when one intendes to rotated these images up to 360 degrees, so that no matter how it will be rotated the information fits within the canvas.')

    args = argp.parse_args()

    _ = run(args)

if __name__ == '__main__': main()
